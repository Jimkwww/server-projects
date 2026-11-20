"""Custom discrete complex diagonal MIMO SSM, not faithful S5 or Mamba."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def serial_scan(a, b):
    h = torch.zeros_like(b[:, 0])
    out = []
    for n in range(b.shape[1]):
        h = a[:, n] * h + b[:, n]
        out.append(h)
    return torch.stack(out, 1)


def parallel_scan(a, b):
    """Inclusive affine prefix; zero initial state. O(N log N) work, NOT fused."""
    if b.shape[1] == 1:
        return b + a * 0
    step = 1
    while step < b.shape[1]:
        b = torch.cat((b[:, :step], b[:, step:] + a[:, step:] * b[:, :-step]), 1)
        a = torch.cat((a[:, :step], a[:, step:] * a[:, :-step]), 1)
        step *= 2
    return b


# --- Q_corr physical-gate round variant groups ---------------------------------------
# c_n = |r_n| (local correlation strength) -> write gate; d_n = |r_n - r_{n-1}| (local
# correlation change) -> retention gate. gate_* modulators live on the S1 ComplexSSM; the
# token_* variants instead add the standardized scalar(s) to the input token once, so the
# scalar can only reach the gates indirectly through the existing selector.
GATE_C_VARIANTS = ('Q_corr_gate_c', 'Q_corr_gate_cd')
GATE_D_VARIANTS = ('Q_corr_gate_d', 'Q_corr_gate_cd')
TOKEN_C_VARIANTS = ('Q_corr_token_c', 'Q_corr_token_cd')
TOKEN_D_VARIANTS = ('Q_corr_token_d', 'Q_corr_token_cd')
SCALAR_VARIANTS = ('Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                   'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd')


def raw_iq_scalar_features(x, patch, eps=1e-8):
    """Per-token scalar descriptors of the raw (normalized) IQ local correlation.

    r_n is the same normalized lag-1 complex correlation S1_corr uses (7 within-patch
    adjacent pairs, no circular wrap): r_n = sum_m s[m] conj(s[m-1]) / sqrt(sum|s[m]|^2 *
    sum|s[m-1]|^2). Returns c = |r_n| [B,N] and d = |r_n - r_{n-1}| [B,N] with d[:,0]=0
    (no previous patch). c/d are computed from the complex r BEFORE any scalar
    standardization, per the round spec (standardizing Re/Im first would change |r|)."""
    s = torch.complex(x[:, 0], x[:, 1]).reshape(x.shape[0], -1, patch)
    cur, prev = s[..., 1:], s[..., :-1]
    r = (cur * prev.conj()).sum(-1) / (
        (cur.abs().square().sum(-1) * prev.abs().square().sum(-1)).sqrt() + eps)
    c = r.abs()
    d = (r[:, 1:] - r[:, :-1]).abs()
    d = torch.cat((torch.zeros_like(d[:, :1]), d), 1)
    return c, d


def physical_features(x, patch, eps=1e-8):
    """x [B,2,L]; within-token adjacent pairs only (no circular/sample wrap)."""
    if patch < 2 or x.shape[-1] % patch:
        raise ValueError('physical features require patch>=2 and L divisible by patch')
    s = torch.complex(x[:, 0], x[:, 1]).reshape(x.shape[0], -1, patch)
    energy = s.abs().square().mean(-1)
    cur, prev = s[..., 1:], s[..., :-1]
    r = (cur * prev.conj()).sum(-1) / (
        (cur.abs().square().sum(-1) * prev.abs().square().sum(-1)).sqrt() + eps)
    return torch.stack((torch.log(energy + eps), r.real, r.imag), -1)


def sinusoidal_positions(n, d, device, dtype):
    """Absolute sinusoidal token positions; matches the historical Global reference."""
    t = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000) / d))
    p = torch.zeros(n, d, device=device, dtype=torch.float32)
    p[:, 0::2] = torch.sin(t * freq)
    p[:, 1::2] = torch.cos(t * freq[:d // 2])
    return p.to(dtype)


def perturb(x, frequency=0., phase=0.):
    """frequency in cycles/raw sample, phase in radians; scalar or [B]."""
    f = torch.as_tensor(frequency, device=x.device, dtype=x.dtype).reshape(-1, 1)
    p = torch.as_tensor(phase, device=x.device, dtype=x.dtype).reshape(-1, 1)
    angle = 2 * math.pi * f * torch.arange(x.shape[-1], device=x.device) + p
    co, si = angle.cos(), angle.sin()
    return torch.stack((x[:, 0]*co-x[:, 1]*si, x[:, 0]*si+x[:, 1]*co), 1)


def max_abs_norm(x):
    """Per-sample joint max |component| over I and Q; matches data.py __getitem__.

    Idempotent on already-max_abs-normalized input (divides by 1.0)."""
    m = x.abs().reshape(x.shape[0], -1).max(dim=1, keepdim=True).values  # [B,1]
    return x / m.clamp_min(1e-8).unsqueeze(-1)


class PatchEmbedding(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(2*patch, dim)

    def forward(self, x):
        # [I0,Q0,I1,Q1,...] ordering: server must compare with historical code.
        b, c, length = x.shape
        if c != 2 or length % self.patch:
            raise ValueError('expected [B,2,L] with L divisible by patch')
        return self.proj(x.transpose(1, 2).reshape(b, length//self.patch, 2*self.patch))


class ComplexSSM(nn.Module):
    def __init__(self, dim, state, variant, hidden, scan='parallel', gain=1.):
        super().__init__()
        self.variant, self.scan = variant, scan
        self.B = nn.Parameter(torch.randn(state, dim, dtype=torch.cfloat)/math.sqrt(dim))
        self.C = nn.Parameter(torch.randn(dim, state, dtype=torch.cfloat)/math.sqrt(state))
        self.direct = nn.Parameter(torch.ones(dim))
        # decay times log-uniform from 2 to 128 tokens; trainable positive alpha.
        alpha = 1/torch.exp(torch.linspace(math.log(2), math.log(128), state))
        self.raw_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, state+1)[:-1])
        self.raw_delta = nn.Parameter(torch.full((state,), math.log(math.expm1(1.))))
        self.selector = None
        if variant in ('S1', 'S2', 'S2_concat', 'S1_energy', 'S1_corr', 'B_front',
                       'R0', 'C0', 'Rb', 'Cb', 'Q_linear', 'Q_corr', 'R_corr', 'Q_corr_frozen',
                       'Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                       'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd'):
            # S1 uses zero physical slots: exactly same parameter count as S2.
            self.selector = nn.Sequential(nn.Linear(dim+3, hidden), nn.SiLU(), nn.Linear(hidden, 2*state))
            nn.init.normal_(self.selector[-1].weight, std=1e-3)
            nn.init.zeros_(self.selector[-1].bias)
        # Physical scalar gate modulators (per-layer, per-state, zero-init). write_w shifts
        # the write gate logit by w_k * c_tilde; retention_v shifts the decay argument by
        # v_k * d_tilde. One scalar per sample, read by every layer.
        self.write_w = nn.Parameter(torch.zeros(state)) if variant in GATE_C_VARIANTS else None
        self.retention_v = nn.Parameter(torch.zeros(state)) if variant in GATE_D_VARIANTS else None
        self.gain = gain
        self.collect_stats = False
        self.last_stats = {}
        self.collect_gate_detail = False
        self.last_gate_detail = {}

    def forward(self, u, p, gate=None):
        v = F.linear(u.to(self.B.dtype), self.B)
        base_delta = self.raw_delta.view(1, 1, -1)
        if self.selector is None:
            delta = F.softplus(base_delta)
            g = torch.ones_like(v.real)
        else:
            aux = p if self.variant == 'S2' else torch.zeros_like(p)
            d_raw, g_raw = self.selector(torch.cat((u, aux), -1)).chunk(2, -1)
            g_logit = g_raw
            delta_arg = base_delta + d_raw  # z^delta already carries the base bias
            if gate is not None:
                c_tilde, d_tilde = gate
                if self.write_w is not None and c_tilde is not None:
                    g_logit = g_logit + self.write_w * c_tilde.unsqueeze(-1)
                if self.retention_v is not None and d_tilde is not None:
                    delta_arg = delta_arg + self.retention_v * d_tilde.unsqueeze(-1)
            delta = F.softplus(delta_arg)
            # Starts near 0.5; this initialization differs from S0, recorded explicitly.
            g = torch.sigmoid(g_logit)
        rho = torch.exp(-F.softplus(self.raw_alpha) * delta)
        a = torch.polar(rho.expand_as(v.real), self.theta.expand_as(v.real))
        fn = parallel_scan if self.scan == 'parallel' else serial_scan
        h = fn(a, g*v)
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats = dict(state_rms=h.abs().square().mean().sqrt().item(),
                    state_max=h.abs().max().item(), rho_mean=rho.mean().item(),
                    rho_near_one=(rho > .999).float().mean().item(), gate_mean=g.mean().item(),
                    gate_saturation=((g < .01) | (g > .99)).float().mean().item())
        if self.collect_gate_detail:
            with torch.no_grad():
                self.last_gate_detail = dict(g=g.detach(), rho=rho.detach())
        # gain is an explicit common initial scaling to limit residual branch variance.
        return self.gain * F.linear(h, self.C).real + self.direct*u


# --- Round-4 structure modules (bidirectional context + complex input structure) ---

class ComplexLinear(nn.Module):
    """No-bias complex linear: z = W @ x, W complex [out,in].

    Exactly equals the real weight binding z_R=W_R I - W_I Q, z_I=W_I I + W_R Q.
    Complex entries are initialized real,imag iid N(0,1/in), matching the ComplexSSM
    B/C init convention (NOT nn.Linear's Xavier-uniform; scale difference recorded)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.cfloat)
                              / math.sqrt(in_features))

    def forward(self, x):
        return F.linear(x, self.W)


class ComplexRMSNorm(nn.Module):
    """Complex RMS normalization over channels: rms=sqrt(mean(|z|^2)+eps); out=gamma*z/rms.
    gamma is a real [dim] scale init 1; no shift. eps fixed at 1e-6 (recorded)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, z):
        rms = z.abs().square().mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * z / rms


class RadialAct(nn.Module):
    """Radial (phase-preserving) nonlinearity: z * sigmoid(|z|)."""
    def forward(self, z):
        return z * torch.sigmoid(z.abs())


class ComplexPatchEmbedding(nn.Module):
    """Round-4 complex frontend for B_front/C_bi_front. Each patch of `patch` IQ points
    is one complex vector; ComplexLinear(patch -> complex_dim), then packed to 2*complex_dim
    reals in interleaved [Re0,Im0,Re1,Im1,...] order (mirrors the [I0,Q0,I1,Q1,...] convention)."""
    def __init__(self, patch, complex_dim):
        super().__init__()
        self.patch = patch
        self.complex_dim = complex_dim
        self.proj = ComplexLinear(patch, complex_dim)

    def forward(self, x):
        b, c, length = x.shape
        if c != 2 or length % self.patch:
            raise ValueError('expected [B,2,L] with L divisible by patch')
        s = torch.complex(x[:, 0], x[:, 1]).reshape(b, length // self.patch, self.patch)
        z = self.proj(s)
        return torch.stack((z.real, z.imag), -1).reshape(b, length // self.patch, 2 * self.complex_dim)


class _Direction(nn.Module):
    """Single-direction input-driven (S1-style) selective complex diagonal SSM core.

    Independent parameters per direction: B [state,dim], C [dim,state], decay alpha/theta/
    delta, and the selector net (reads u + 3 zero physical slots, exactly the S1 setup)."""
    def __init__(self, dim, state, hidden, scan):
        super().__init__()
        self.state = state
        self.scan = scan
        self.B = nn.Parameter(torch.randn(state, dim, dtype=torch.cfloat) / math.sqrt(dim))
        self.C = nn.Parameter(torch.randn(dim, state, dtype=torch.cfloat) / math.sqrt(state))
        alpha = 1 / torch.exp(torch.linspace(math.log(2), math.log(128), state))
        self.raw_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, state + 1)[:-1])
        self.raw_delta = nn.Parameter(torch.full((state,), math.log(math.expm1(1.))))
        self.selector = nn.Sequential(nn.Linear(dim + 3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * state))
        nn.init.normal_(self.selector[-1].weight, std=1e-3)
        nn.init.zeros_(self.selector[-1].bias)

    def forward(self, u):
        # u: real [B,N,dim]. Returns (h, rho, g).
        v = F.linear(u.to(self.B.dtype), self.B)
        d_raw, g_raw = self.selector(torch.cat((u, torch.zeros_like(u[..., :3])), -1)).chunk(2, -1)
        delta = F.softplus(self.raw_delta.view(1, 1, -1) + d_raw)
        g = torch.sigmoid(g_raw)
        rho = torch.exp(-F.softplus(self.raw_alpha) * delta)
        a = torch.polar(rho, self.theta.expand_as(rho))
        fn = parallel_scan if self.scan == 'parallel' else serial_scan
        return fn(a, g * v), rho, g


class BidirectionalSSM(nn.Module):
    """Two independent S1-style directions: forward + backward. Backward is implemented by
    flipping the token dim, scanning forward, then flipping the output back (no patch-internal
    reversal, no IQ conjugation; zero boundary states both directions). The direct (skip) term
    is added exactly once, after summing both readouts. No unrecorded averaging or extra gain."""
    def __init__(self, dim, state, hidden, scan='parallel', gain=1.):
        super().__init__()
        self.gain = gain
        self.forward_dir = _Direction(dim, state, hidden, scan)
        self.backward_dir = _Direction(dim, state, hidden, scan)
        self.direct = nn.Parameter(torch.ones(dim))
        self.collect_stats = False
        self.last_stats = {}

    def forward(self, u, p, gate=None):
        hf, rhof, gf = self.forward_dir(u)
        hb, rhob, gb = self.backward_dir(u.flip(1))
        hb = hb.flip(1)
        y = self.gain * (F.linear(hf, self.forward_dir.C) + F.linear(hb, self.backward_dir.C)).real + self.direct * u
        if self.collect_stats:
            with torch.no_grad():
                h = torch.cat((hf, hb), -1)
                rho = torch.cat((rhof, rhob), -1)
                g = torch.cat((gf, gb), -1)
                self.last_stats = dict(state_rms=h.abs().square().mean().sqrt().item(),
                    state_max=h.abs().max().item(), rho_mean=rho.mean().item(),
                    rho_near_one=(rho > .999).float().mean().item(), gate_mean=g.mean().item(),
                    gate_saturation=((g < .01) | (g > .99)).float().mean().item())
        return y


class ComplexSelectiveSSM(nn.Module):
    """D_complex unidirectional selective complex diagonal SSM with complex feature I/O.

    Reads the current normalized complex feature: selector input = packed [Re,Im] (2*dim)
    + 3 zero slots, net (2*dim+3) -> hidden -> 2*state. Readout keeps the full complex C h
    (no Re()); the direct term is a real [dim] scale init 1 applied to the complex feature.
    No separate readout gain (equivalent to gain=1; recorded)."""
    def __init__(self, dim, state, hidden, scan='parallel'):
        super().__init__()
        self.state = state
        self.B = nn.Parameter(torch.randn(state, dim, dtype=torch.cfloat) / math.sqrt(dim))
        self.C = nn.Parameter(torch.randn(dim, state, dtype=torch.cfloat) / math.sqrt(state))
        self.direct = nn.Parameter(torch.ones(dim))
        alpha = 1 / torch.exp(torch.linspace(math.log(2), math.log(128), state))
        self.raw_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, state + 1)[:-1])
        self.raw_delta = nn.Parameter(torch.full((state,), math.log(math.expm1(1.))))
        self.selector = nn.Sequential(nn.Linear(2 * dim + 3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * state))
        nn.init.normal_(self.selector[-1].weight, std=1e-3)
        nn.init.zeros_(self.selector[-1].bias)
        self.scan = scan
        self.collect_stats = False
        self.last_stats = {}

    def forward(self, z):
        v = F.linear(z, self.B)
        packed = torch.cat((z.real, z.imag), -1)
        sel_in = torch.cat((packed, torch.zeros_like(packed[..., :3])), -1)
        d_raw, g_raw = self.selector(sel_in).chunk(2, -1)
        delta = F.softplus(self.raw_delta.view(1, 1, -1) + d_raw)
        g = torch.sigmoid(g_raw)
        rho = torch.exp(-F.softplus(self.raw_alpha) * delta)
        a = torch.polar(rho, self.theta.expand_as(rho))
        fn = parallel_scan if self.scan == 'parallel' else serial_scan
        h = fn(a, g * v)
        if self.collect_stats:
            with torch.no_grad():
                self.last_stats = dict(state_rms=h.abs().square().mean().sqrt().item(),
                    state_max=h.abs().max().item(), rho_mean=rho.mean().item(),
                    rho_near_one=(rho > .999).float().mean().item(), gate_mean=g.mean().item(),
                    gate_saturation=((g < .01) | (g > .99)).float().mean().item())
        # Full complex readout (no Re()), real direct scale.
        return F.linear(h, self.C) + self.direct.view(1, 1, -1) * z


class ComplexBlock(nn.Module):
    """D_complex block: complex norm -> selective SSM -> complex residual -> complex norm ->
    complex FFN (no-bias ComplexLinear x2 with radial act) -> complex residual. Dropout=0."""
    def __init__(self, dim, state, hidden, ffn_ratio, dropout):
        super().__init__()
        self.norm1 = ComplexRMSNorm(dim)
        self.ssm = ComplexSelectiveSSM(dim, state, hidden)
        self.norm2 = ComplexRMSNorm(dim)
        self.ffn = nn.Sequential(ComplexLinear(dim, int(dim * ffn_ratio)), RadialAct(),
            ComplexLinear(int(dim * ffn_ratio), dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        z = z + self.drop(self.ssm(self.norm1(z)))
        return z + self.drop(self.ffn(self.norm2(z)))


# --- Round-5 modules (complex frontend fair comparison + IQ auxiliary branch) ---

def round5_front_weights(in_c, out_c, generator):
    """Generate the common round-5 complex frontend weights W_R, W_I so that the equivalent
    real matrix M=[[W_R,-W_I],[W_I,W_R]] (interleaved I/Q) has per-entry variance matching
    nn.Linear(2*in_c -> 2*out_c) default init: U(-1/sqrt(2*in_c), 1/sqrt(2*in_c)), variance
    1/(3*2*in_c)=1/(6*in_c). This is NOT the ComplexLinear N(0,1/in) convention (Round-4):
    the Round-4 equivalent-real variance was 1/in, i.e. 6x nn.Linear's 1/(6*in)."""
    bound = 1.0 / math.sqrt(2 * in_c)
    W_R = torch.empty(out_c, in_c).uniform_(-bound, bound, generator=generator)
    W_I = torch.empty(out_c, in_c).uniform_(-bound, bound, generator=generator)
    return W_R, W_I


def complex_to_real_interleaved(W_R, W_I):
    """Expand complex W (out x in) to the equivalent real matrix M (2*out x 2*in) for the
    interleaved [I0,Q0,I1,Q1,...] -> [Re0,Im0,Re1,Im1,...] convention. Row 2k = Re_k, row
    2k+1 = Im_k; col 2j = I_j, col 2j+1 = Q_j. z=W@s equals M @ [I0,Q0,...]."""
    out_c, in_c = W_R.shape
    M = torch.zeros(2 * out_c, 2 * in_c, device=W_R.device, dtype=W_R.dtype)
    M[0::2, 0::2] = W_R
    M[0::2, 1::2] = -W_I
    M[1::2, 0::2] = W_I
    M[1::2, 1::2] = W_R
    return M


class FairRealFront(nn.Module):
    """Round-5 R0/Rb frontend: plain real Linear(2*patch -> dim) initialized to the
    equivalent real matrix M of the matched complex front. bias optional (init zero)."""
    def __init__(self, patch, dim, bias, M):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(2 * patch, dim, bias=bias)
        with torch.no_grad():
            self.proj.weight.copy_(M)
            if bias:
                self.proj.bias.zero_()

    def forward(self, x):
        b, c, length = x.shape
        if c != 2 or length % self.patch:
            raise ValueError('expected [B,2,L] with L divisible by patch')
        return self.proj(x.transpose(1, 2).reshape(b, length // self.patch, 2 * self.patch))


class FairComplexFront(nn.Module):
    """Round-5 C0/Cb frontend: bound complex weight W = W_R + j W_I, optional bias (init
    zero). Output packed to interleaved [Re0,Im0,...]. Identical at init to FairRealFront
    built from the same W_R,W_I."""
    def __init__(self, patch, complex_dim, bias, W_R, W_I):
        super().__init__()
        self.patch = patch
        self.complex_dim = complex_dim
        self.W = nn.Parameter(torch.complex(W_R, W_I))
        self.bias = nn.Parameter(torch.zeros(2 * complex_dim)) if bias else None

    def forward(self, x):
        b, c, length = x.shape
        if c != 2 or length % self.patch:
            raise ValueError('expected [B,2,L] with L divisible by patch')
        s = torch.complex(x[:, 0], x[:, 1]).reshape(b, length // self.patch, self.patch)
        z = F.linear(s, self.W)
        out = torch.stack((z.real, z.imag), -1).reshape(b, length // self.patch, 2 * self.complex_dim)
        if self.bias is not None:
            out = out + self.bias
        return out


def state_mean_pack(q, patch):
    """q [B,L,S] complex -> per-patch mean over the S states, packed interleaved [Re0,Im0,...]
    -> [B, L//patch, 2S] real (Q_linear readout)."""
    b, length, s = q.shape
    mean = q.reshape(b, length // patch, patch, s).mean(2)
    return torch.stack((mean.real, mean.imag), -1).reshape(b, length // patch, 2 * s)


def state_corr_pack(q, patch, eps=1e-8):
    """q [B,L,S] complex -> per-channel patch-internal normalized adjacent complex
    correlation over the 7 within-patch pairs (no cross-patch, no circular), packed
    interleaved [Re0,Im0,...] -> [B, L//patch, 2S] real (Q_corr/R_corr readout).
    r[k,n] = sum_m q[k,m] conj(q[k,m-1]) / (sqrt(sum|q[m]|^2 * sum|q[m-1]|^2) + eps)."""
    b, length, s = q.shape
    qp = q.reshape(b, length // patch, patch, s)
    cur, prev = qp[:, :, 1:, :], qp[:, :, :-1, :]
    num = (cur * prev.conj()).sum(2)
    den = (cur.abs().square().sum(2) * prev.abs().square().sum(2)).sqrt() + eps
    r = num / den
    return torch.stack((r.real, r.imag), -1).reshape(b, length // patch, 2 * s)


class ComplexRecurBranch(nn.Module):
    """Round-5 B-line auxiliary complex recurrence over the full raw sample. NOT the S1
    backbone (explicitly not called S1). q[m] = a*q[m-1] + b*s[m], s=I+jQ, per-channel
    scalar a,b: a = exp(-softplus(raw_alpha))*exp(j*theta). q is 8-dim complex state,
    q[-1]=0 per sample, continuous over all L=1024 points (no per-patch reset), computed
    once per input. tau init log-uniform 2..128 samples (alpha=1/tau), theta deterministic
    uniform coverage [-pi,pi), b zero-mean complex init real/imag each N(0, b_std)."""
    def __init__(self, state=8, tau_min=2, tau_max=128, seed=None, b_std=0.1):
        super().__init__()
        self.state = state
        alpha = 1 / torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), state))
        self.raw_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))
        self.theta = nn.Parameter(torch.linspace(-math.pi, math.pi, state + 1)[:-1])
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        b = torch.empty(state, dtype=torch.cfloat)
        if gen is not None:
            b.real.normal_(0.0, b_std, generator=gen)
            b.imag.normal_(0.0, b_std, generator=gen)
        else:
            b.real.normal_(0.0, b_std)
            b.imag.normal_(0.0, b_std)
        self.b = nn.Parameter(b)

    def forward(self, x):
        s = torch.complex(x[:, 0], x[:, 1])  # [B,L]
        a = torch.exp(-F.softplus(self.raw_alpha)) * torch.exp(1j * self.theta)  # [S]
        b_s = s.unsqueeze(-1) * self.b  # [B,L,S]
        a_exp = a.expand(x.shape[0], x.shape[2], self.state)  # [B,L,S]
        return parallel_scan(a_exp, b_s)  # [B,L,S] complex


class RealRecurBranch(nn.Module):
    """Round-5 R_corr: 8 groups of 2-dim real state. x_k[m] = A_k x_k[m-1] + B_k [I[m],Q[m]]
    with A_k a general 2x2 real matrix (spectral norm < 1 enforced by forward scaling, not
    post-hoc clipping) and B_k a free 2x2 real input map. Initialized from the Q_corr complex
    recurrence expansion so the initial state sequence matches Q_corr: A_k = rho_k*R(theta_k),
    B_k = [[b_R,-b_I],[b_I,b_R]]. Parameterization: rho_k = sigmoid(logit_k); forward
    A_k = rho_k * A_raw_k / max(1, ||A_raw_k||_2), so ||A_k||_2 <= rho_k < 1 always."""
    def __init__(self, state=8, tau_min=2, tau_max=128, seed=None, b_std=0.1):
        super().__init__()
        self.state = state
        alpha = 1 / torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), state))
        rho_init = torch.exp(-F.softplus(torch.log(torch.expm1(alpha))))  # = exp(-1/tau)
        theta_init = torch.linspace(-math.pi, math.pi, state + 1)[:-1]
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        b = torch.empty(state, dtype=torch.cfloat)
        if gen is not None:
            b.real.normal_(0.0, b_std, generator=gen)
            b.imag.normal_(0.0, b_std, generator=gen)
        else:
            b.real.normal_(0.0, b_std)
            b.imag.normal_(0.0, b_std)
        # A_raw init = R(theta) (rotation, spectral norm 1); forward scales by rho.
        cos, sin = theta_init.cos(), theta_init.sin()
        A_raw = torch.stack((torch.stack((cos, -sin), -1), torch.stack((sin, cos), -1)), -2)
        B_init = torch.stack((torch.stack((b.real, -b.imag), -1), torch.stack((b.imag, b.real), -1)), -2)
        self.logit = nn.Parameter(torch.logit(rho_init.clamp(1e-6, 1 - 1e-6)))
        self.A_raw = nn.Parameter(A_raw)
        self.B = nn.Parameter(B_init)

    def forward(self, x):
        iq = x.transpose(1, 2)  # [B,L,2]
        rho = torch.sigmoid(self.logit)  # [S]
        anorm = torch.linalg.matrix_norm(self.A_raw, ord=2)  # [S] spectral norm
        A = rho[:, None, None] * self.A_raw / anorm.clamp_min(1.0)[:, None, None]  # [S,2,2]
        # O(L) sequential scan, vectorized over batch and groups (explicit, no Python-level
        # per-sample loop). q[-1]=0 per sample; continuous across patches.
        state_vec = iq.new_zeros(x.shape[0], self.state, 2)
        xs = []
        for m in range(x.shape[2]):
            u_m = iq[:, m]  # [B,2]
            state_vec = torch.einsum('gij,bgj->bgi', A, state_vec) + \
                torch.einsum('gij,bj->bgi', self.B, u_m)
            xs.append(state_vec)
        return torch.stack(xs, 1)  # [B,L,S,2] real


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads, self.dropout = heads, dropout
        self.qkv, self.out = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)

    def forward(self, x, p, gate=None):
        b, n, d = x.shape
        q,k,v = self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        # Manual softmax attention, matching the historical Global reference (no padding
        # mask is needed for fixed-length RML2018A). Avoids non-deterministic SDPA kernels.
        scores = (q @ k.transpose(-2, -1)) * ((d // self.heads) ** -0.5)
        weights = F.dropout(scores.float().softmax(-1).to(x.dtype), self.dropout, self.training)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['dim']
        v = cfg['variant']
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if v in ('G','G_full','G_corr','G_Q_corr','G_Q_corr_frozen'):
            self.mix = Attention(d, cfg['heads'], cfg['dropout'])
        elif v in ('A64','A128','C_bi_front'):
            state = 64 if v == 'A64' else 128
            self.mix = BidirectionalSSM(d, state, cfg['selector_hidden'], cfg['scan'], cfg['ssm_gain'])
        else:
            self.mix = ComplexSSM(d, cfg['state'], v, cfg['selector_hidden'], cfg['scan'], cfg['ssm_gain'])
        self.ffn = nn.Sequential(nn.Linear(d, int(d*cfg['ffn_ratio'])), nn.GELU(),
            nn.Dropout(cfg['dropout']), nn.Linear(int(d*cfg['ffn_ratio']),d))
        self.drop = nn.Dropout(cfg['dropout'])

    def forward(self, x, p, gate=None):
        x = x + self.drop(self.mix(self.norm1(x),p,gate))
        return x + self.drop(self.ffn(self.norm2(x)))


class Classifier(nn.Module):
    def __init__(self, cfg, seed=None):
        super().__init__()
        v = cfg['variant']
        if v not in ('G','S0','S1','S2','S2_concat','S1_energy','S1_corr','G_full','G_corr',
                     'A64','A128','B_front','C_bi_front','D_complex',
                     'R0','C0','Rb','Cb','Q_linear','Q_corr','R_corr','Q_corr_frozen',
                     'G_Q_corr','G_Q_corr_frozen',
                     'Q_corr_gate_c','Q_corr_gate_d','Q_corr_gate_cd',
                     'Q_corr_token_c','Q_corr_token_d','Q_corr_token_cd'):
            raise ValueError('unsupported variant')
        if cfg['direction'] != 'forward':
            raise ValueError('v1 implements forward scan only; do not silently change direction')
        self.cfg = dict(cfg)
        if v == 'D_complex':
            # Full complex path. Complex width = dim/2 real components; no position encoding;
            # bidirectionality is NOT encoded via the direction flag (backward is internal flip).
            if cfg['dim'] % 2:
                raise ValueError('D_complex requires even dim')
            cd = cfg['dim'] // 2
            self.front = ComplexLinear(cfg['patch'], cd)
            self.fusion = None
            self.blocks = nn.ModuleList([ComplexBlock(cd, cfg['state'], cfg['selector_hidden'],
                int(cfg['ffn_ratio']), cfg['dropout']) for _ in range(cfg['depth'])])
            self.norm = ComplexRMSNorm(cd)
            self.head = nn.Linear(cfg['dim'], cfg['classes'])
        else:
            round5 = v in ('R0', 'C0', 'Rb', 'Cb', 'Q_linear', 'Q_corr', 'R_corr', 'Q_corr_frozen',
                           'G_Q_corr', 'G_Q_corr_frozen',
                           'Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                           'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd')
            if round5 and seed is None:
                raise ValueError('round-5 variants require an explicit seed for matched init')
            if v in ('R0', 'Rb'):
                # A-line real front: equivalent real matrix M of the common complex weights.
                gen = torch.Generator().manual_seed(seed + 70001)
                W_R, W_I = round5_front_weights(cfg['patch'], cfg['dim'] // 2, gen)
                self.front = FairRealFront(cfg['patch'], cfg['dim'], bias=(v == 'Rb'),
                                           M=complex_to_real_interleaved(W_R, W_I))
            elif v in ('C0', 'Cb'):
                gen = torch.Generator().manual_seed(seed + 70001)
                W_R, W_I = round5_front_weights(cfg['patch'], cfg['dim'] // 2, gen)
                self.front = FairComplexFront(cfg['patch'], cfg['dim'] // 2, bias=(v == 'Cb'),
                                              W_R=W_R, W_I=W_I)
            elif v in ('B_front', 'C_bi_front'):
                self.front = ComplexPatchEmbedding(cfg['patch'], cfg['dim'] // 2)
            else:
                self.front = PatchEmbedding(cfg['patch'], cfg['dim'])
            self.fusion = nn.Linear(3, cfg['dim'], bias=False) if v in ('S2_concat','S1_energy','S1_corr','G_full','G_corr') else None
            # B-line IQ auxiliary branch (free real main path; only models raw-IQ complex
            # temporal structure via an auxiliary branch). W_f init zero so full model == S1.
            if v in ('Q_linear', 'Q_corr', 'Q_corr_frozen', 'G_Q_corr', 'G_Q_corr_frozen',
                     'Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                     'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd'):
                self.aux_branch = ComplexRecurBranch(seed=seed + 70002)
                self.aux_readout = 'mean' if v == 'Q_linear' else 'corr'
                self.aux_fusion = nn.Linear(16, cfg['dim'], bias=False)
                nn.init.zeros_(self.aux_fusion.weight)
                if v in ('Q_corr_frozen', 'G_Q_corr_frozen'):
                    # Freeze the aux branch (raw_alpha/theta/b) from its initial (pre-training)
                    # values; main path + W_f + head still train. Same seed+70002 init as the
                    # learnable counterpart (Q_corr / G_Q_corr).
                    for p in self.aux_branch.parameters():
                        p.requires_grad_(False)
            elif v == 'R_corr':
                self.aux_branch = RealRecurBranch(seed=seed + 70002)
                self.aux_readout = 'corr'
                self.aux_fusion = nn.Linear(16, cfg['dim'], bias=False)
                nn.init.zeros_(self.aux_fusion.weight)
            else:
                self.aux_branch = None
                self.aux_fusion = None
                self.aux_readout = None
            # Token controls: add the standardized scalar(s) to the input token once (zero-init
            # no-bias Linear). Same information as the gate_* variants but reaching the gates
            # only indirectly through the selector. gate_* variants leave token_proj = None.
            if v in ('Q_corr_token_c', 'Q_corr_token_d'):
                self.token_proj = nn.Linear(1, cfg['dim'], bias=False)
                nn.init.zeros_(self.token_proj.weight)
            elif v == 'Q_corr_token_cd':
                self.token_proj = nn.Linear(2, cfg['dim'], bias=False)
                nn.init.zeros_(self.token_proj.weight)
            else:
                self.token_proj = None
            self.needs_gate_scalars = v in SCALAR_VARIANTS
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['depth'])])
            self.norm = nn.LayerNorm(cfg['dim'])
            self.head = nn.Linear(cfg['dim'], cfg['classes'])
        self.register_buffer('phys_mean', torch.zeros(3))
        self.register_buffer('phys_std', torch.ones(3))
        # 16-dim aux descriptor standardization (frozen after fit_aux; Q_corr/R_corr share
        # identical init so their initial stats match; Q_linear differs).
        self.register_buffer('aux_mean', torch.zeros(16))
        self.register_buffer('aux_std', torch.ones(16))
        # Frozen scalar standardization for c=|r| and d=|r_n-r_{n-1}| (only the new scalar
        # variants carry these buffers; Q_corr and the historical baselines do not, so their
        # checkpoints stay key-compatible). fit on 8192 train samples (d excludes position 0).
        if v in SCALAR_VARIANTS:
            self.register_buffer('gate_c_mean', torch.zeros(1))
            self.register_buffer('gate_c_std', torch.ones(1))
            self.register_buffer('gate_d_mean', torch.zeros(1))
            self.register_buffer('gate_d_std', torch.ones(1))
        # Historical Global reference uses absolute sinusoidal positions (audited from the
        # frozen SwinE/MultToken Global). The recurrent SSM variants use no explicit positions.
        pos_mode = 'sinusoidal' if v in ('G','G_full','G_corr','G_Q_corr','G_Q_corr_frozen') else cfg['position']
        if pos_mode == 'learned':
            self.pos = nn.Parameter(torch.zeros(1, cfg['input_length'] // cfg['patch'], cfg['dim']))
            nn.init.normal_(self.pos, std=.02)
        elif pos_mode in ('none', 'sinusoidal'):
            self.pos = None
        else:
            raise ValueError('position must be none, learned or sinusoidal')
        self.pos_mode = pos_mode

    def forward(self, x):
        v = self.cfg['variant']
        if v == 'D_complex':
            b, c, length = x.shape
            if c != 2 or length % self.cfg['patch']:
                raise ValueError('expected [B,2,L] with L divisible by patch')
            s = torch.complex(x[:, 0], x[:, 1]).reshape(b, length // self.cfg['patch'], self.cfg['patch'])
            z = self.front(s)
            for block in self.blocks:
                z = block(z)
            z = self.norm(z).mean(1)
            return self.head(torch.cat((z.real, z.imag), -1))
        u = self.front(x)
        if self.pos_mode == 'learned':
            if u.shape[1] != self.pos.shape[1]:
                raise ValueError('learned position requires training length; use separate length models')
            u = u + self.pos
        elif self.pos_mode == 'sinusoidal':
            u = u + sinusoidal_positions(u.shape[1], u.shape[2], u.device, u.dtype)
        if self.aux_branch is not None:
            # u_new = u + W_f * f_standardized; W_f init zero, grad flows to branch params.
            f = (self.aux_descriptor(x) - self.aux_mean) / self.aux_std
            u = u + self.aux_fusion(f)
        gate = None
        if self.needs_gate_scalars:
            c_tilde, d_tilde = self._std_gate_scalars(x)
            if self.token_proj is not None:
                if v == 'Q_corr_token_c':
                    s = c_tilde.unsqueeze(-1)
                elif v == 'Q_corr_token_d':
                    s = d_tilde.unsqueeze(-1)
                else:
                    s = torch.stack((c_tilde, d_tilde), -1)
                u = u + self.token_proj(s)
            if v in GATE_C_VARIANTS or v in GATE_D_VARIANTS:
                gate = (c_tilde if v in GATE_C_VARIANTS else None,
                        d_tilde if v in GATE_D_VARIANTS else None)
        if v in ('S2','S2_concat','S1_energy','S1_corr','G_full','G_corr'):
            p = (physical_features(x, self.cfg['patch']) - self.phys_mean) / self.phys_std
            if v == 'S1_energy':
                p[..., 1:] = 0
            elif v in ('S1_corr', 'G_corr'):
                p[..., 0] = 0
        else:
            p = u.new_zeros(u.shape[0], u.shape[1], 3)
        if self.fusion is not None:
            u = u + self.fusion(p)
        for block in self.blocks:
            u = block(u, p, gate)
        return self.head(self.norm(u).mean(1))

    def _std_gate_scalars(self, x):
        """Standardized c_tilde, d_tilde [B,N] from the (already normalized) input IQ, using
        the frozen scalar statistics. d first position is forced to 0 after standardization
        (no previous patch), matching the fit which excluded position 0."""
        c, d = raw_iq_scalar_features(x, self.cfg['patch'])
        c_tilde = (c - self.gate_c_mean) / self.gate_c_std
        d_tilde = ((d - self.gate_d_mean) / self.gate_d_std).clone()
        d_tilde[:, 0] = 0.0
        return c_tilde, d_tilde

    def aux_descriptor(self, x):
        """Raw 16-dim aux descriptor (before standardization) for the B-line variants.
        Reads the same augmented+normalized raw IQ the model receives. No detach, so
        gradient flows to the branch parameters."""
        q = self.aux_branch(x)
        if self.aux_readout == 'mean':
            return state_mean_pack(q, self.cfg['patch'])
        qc = q if q.is_complex() else torch.complex(q[..., 0], q[..., 1])
        return state_corr_pack(qc, self.cfg['patch'])


def build_round5_model(cfg, seed):
    """Round-5 model builder. The S1 main path is initialized identically across seeds for
    matched comparison: A-line (R0/C0/Rb/Cb) shares only the backbone (blocks/norm/head);
    B-line (Q_linear/Q_corr/R_corr) shares the full S1 main path (front + backbone). The
    matched complex/real front and the aux branch use dedicated per-seed generators."""
    v = cfg['variant']
    ref = Classifier({**cfg, 'variant': 'S1'})
    model = Classifier(cfg, seed=seed)
    if v in ('R0', 'C0', 'Rb', 'Cb'):
        sd = {k: val for k, val in ref.state_dict().items()
              if k.startswith(('blocks.', 'norm.', 'head.'))}
    else:
        sd = {k: val for k, val in ref.state_dict().items()
              if k.startswith(('front.', 'blocks.', 'norm.', 'head.'))}
    model.load_state_dict(sd, strict=False)
    return model


def build_global_qcorr_model(cfg, seed):
    """G_Q_corr / G_Q_corr_frozen builder. The Global backbone (front PatchEmbedding, 6
    attention blocks, LayerNorm, head) is initialized identically to the historical G
    reference (constructed here under the same seed_all(seed) RNG state the engine sets
    before building), so the two new models reproduce G exactly at init. The aux branch
    (ComplexRecurBranch, per-seed seed+70002) + zero-init Linear(16,dim) fusion are added
    on top; there is NO G_corr raw-IQ physical-feature fusion (the original correlation is
    replaced by the 8-channel filtered correlation)."""
    ref = Classifier({**cfg, 'variant': 'G'})
    model = Classifier(cfg, seed=seed)
    sd = {k: val for k, val in ref.state_dict().items()
          if k.startswith(('front.', 'blocks.', 'norm.', 'head.'))}
    model.load_state_dict(sd, strict=False)
    return model


def parameter_count(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
