#!/usr/bin/env python
"""Q_corr physical-gate round pre-training acceptance (5 checks).

Runs at the real production config (patch=8, dim=176, depth=6, heads=4, L=1024, 24 classes)
and saves per-seed numerical results to reports/qcorr_gates/acceptance/.

  Check 1  Hand-calculated descriptors: a constant complex exponential gives c~=1, d~=0;
           two patches with phase increments 0 and pi/2 give r0=1, r1=j, d1=sqrt(2), d0=0.
  Check 2  CFO transform: r_n rotates by the common CFO phase, c=|r| and d=|r_n-r_{n-1}|
           are preserved exactly for any signal (perturb is sample-wise e^{j 2pi f m}).
  Check 3  Zero-init equivalence: all 6 new variants at init produce logits identical to
           Q_corr on the same batch; records real-scalar param increments and token shape
           [batch,128,176] for every variant.
  Check 4  Gate wiring: write_w changes only g (exact sigmoid sign+amplitude); retention_v
           changes only delta/rho (exact softplus/exp form). Cross-talk-free in _cd.
  Check 5  Gradient + resume: new params receive grad and update; token shape; param
           increments. Save/restore consistency delegated to tests/test_qcorr_gates_resume.py.

No test split is touched. This is a verification script, not a research run.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_exp.model import (ComplexSSM, build_round5_model, parameter_count, perturb,
                           raw_iq_scalar_features)
from ssm_exp.utils import seed_all, save_json

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(variant='Q_corr', patch=8, dim=176, depth=6, heads=4, state=128,
           selector_hidden=64, classes=24, input_length=1024, ffn_ratio=4, dropout=0.0,
           position='none', direction='forward', scan='parallel', ssm_gain=1.0)
VARIANTS = ['Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
            'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd']


def _r_of(x, patch):
    s = torch.complex(x[:, 0], x[:, 1]).reshape(x.shape[0], -1, patch)
    cur, prev = s[..., 1:], s[..., :-1]
    return (cur * prev.conj()).sum(-1) / (
        (cur.abs().square().sum(-1) * prev.abs().square().sum(-1)).sqrt() + 1e-8)


# --- check 1: hand-calculated descriptors ---------------------------------------------
def check1():
    patch, L = CFG['patch'], CFG['input_length']
    omega = 0.3
    t = torch.arange(L).float()
    s = torch.exp(1j * omega * t)
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)
    c, d = raw_iq_scalar_features(x, patch)
    c_max_dev = float((c - 1.0).abs().max())
    d_max = float(d.abs().max())

    # two patches: phase increment 0 -> r0=1; increment pi/2 -> r1=j
    t0 = torch.arange(patch).float()
    tok0 = torch.exp(1j * 0.0 * t0)          # r0 = 1
    tok1 = torch.exp(1j * (math.pi / 2) * t0)  # r1 = j
    s2 = torch.cat([tok0, tok1])
    x2 = torch.stack((s2.real, s2.imag), 0).unsqueeze(0)
    r2 = _r_of(x2, patch)
    c2, d2 = raw_iq_scalar_features(x2, patch)
    return {
        'const_exp_c_max_dev': c_max_dev,
        'const_exp_d_max_abs': d_max,
        'const_exp_d0_exact_zero': bool((d[:, 0] == 0).all()),
        'two_token_r0': [float(r2[0, 0].real), float(r2[0, 0].imag)],
        'two_token_r1': [float(r2[0, 1].real), float(r2[0, 1].imag)],
        'two_token_c': [float(v) for v in c2[0].tolist()],
        'two_token_d0': float(d2[0, 0]),
        'two_token_d1': float(d2[0, 1]),
        'two_token_d1_expected_sqrt2': math.sqrt(2.0),
    }


# --- check 2: CFO transform ------------------------------------------------------------
def check2():
    patch, L = CFG['patch'], CFG['input_length']
    t = torch.arange(L).float()
    s = 0.3 * torch.exp(1j * 0.2 * t) + 0.7 * torch.exp(1j * 0.5 * t) + 0.1 * torch.randn(L)
    x = torch.stack((s.real, s.imag), 0).unsqueeze(0)
    freq = 0.001
    r0 = _r_of(x, patch)
    xp = perturb(x, freq)
    r1 = _r_of(xp, patch)
    rot = torch.tensor(math.e ** (1j * 2 * math.pi * freq), dtype=torch.complex64)
    r_rotate_err = float((r1 - r0 * rot).abs().max())
    c0, d0 = raw_iq_scalar_features(x, patch)
    c1, d1 = raw_iq_scalar_features(xp, patch)
    return {
        'cfo': freq,
        'r_common_rotation_max_abs_err': r_rotate_err,
        'c_preserved_max_abs_err': float((c0 - c1).abs().max()),
        'd_preserved_max_abs_err': float((d0 - d1).abs().max()),
    }


# --- check 3: zero-init equivalence + param counts + token shape -----------------------
def check3(seed):
    x = torch.randn(8, 2, 1024)
    seed_all(seed)
    qc = build_round5_model(dict(CFG, variant='Q_corr'), seed).eval()
    base_params = parameter_count(qc)
    with torch.no_grad():
        qc_logits = qc(x)

    variants = {}
    for v in VARIANTS:
        seed_all(seed)
        m = build_round5_model(dict(CFG, variant=v), seed).eval()
        with torch.no_grad():
            l = m(x)
        captured = {}

        def hook(module, inp):
            captured['u'] = inp[0]

        h = m.blocks[0].register_forward_pre_hook(hook)
        with torch.no_grad():
            m(x)
        h.remove()
        variants[v] = {
            'max_abs_logit_diff_vs_Q_corr': float((l - qc_logits).abs().max().item()),
            'params': parameter_count(m),
            'param_increment': parameter_count(m) - base_params,
            'token_shape': list(captured['u'].shape),
        }
    return {'base_params_Q_corr': base_params, 'variants': variants}


# --- check 4: gate wiring --------------------------------------------------------------
def _ssm(variant):
    seed_all(0)
    return ComplexSSM(CFG['dim'], CFG['state'], variant, CFG['selector_hidden'], 'parallel', 1.0)


def _gate4(variant, w, v, c_tilde, d_tilde):
    s = _ssm(variant)
    if w is not None:
        s.write_w.data = torch.tensor(w)
    if v is not None:
        s.retention_v.data = torch.tensor(v)
    u = torch.randn(2, 3, CFG['dim'])
    p = torch.zeros(2, 3, 3)
    s.collect_gate_detail = True
    s(u, p, gate=None)
    g0 = s.last_gate_detail['g'].clone()
    rho0 = s.last_gate_detail['rho'].clone()
    s(u, p, gate=(c_tilde, d_tilde))
    g1 = s.last_gate_detail['g'].clone()
    rho1 = s.last_gate_detail['rho'].clone()
    return s, g0, rho0, g1, rho1


def check4():
    # write gate: w>0 & c>0 -> g up; exact sigmoid formula; rho untouched.
    s, g0, rho0, g1, rho1 = _gate4('Q_corr_gate_c', [1.0, -0.5, 2.0] + [0.0] * 125,
                                    None, torch.full((2, 3), 0.5), None)
    logit0 = torch.logit(g0.clamp(1e-7, 1 - 1e-7))
    g_expected = torch.sigmoid(logit0 + s.write_w * 0.5)
    write = {
        'g_formula_max_abs_err': float((g1 - g_expected).abs().max()),
        'g_increases_where_wc_positive': bool((g1[..., 0] > g0[..., 0]).all()),
        'g_decreases_where_wc_negative': bool((g1[..., 1] < g0[..., 1]).all()),
        'g_unchanged_where_w_zero': bool((g1[..., 3] == g0[..., 3]).all()),
        'rho_unchanged': bool((rho1 == rho0).all()),
    }

    # retention gate: v>0 & d>0 -> delta up -> rho down; exact exp/softplus form; g untouched.
    s, g0, rho0, g1, rho1 = _gate4('Q_corr_gate_d', None,
                                    [1.0, -0.5, 2.0] + [0.0] * 125,
                                    None, torch.full((2, 3), 0.5))
    alpha = F.softplus(s.raw_alpha).view(1, 1, -1)
    delta_base = -torch.log(rho0.clamp_min(1e-12)) / alpha
    delta_arg_base = torch.log(torch.expm1(delta_base))
    rho_expected = torch.exp(-alpha * F.softplus(delta_arg_base + s.retention_v * 0.5))
    retention = {
        'rho_formula_max_abs_err': float((rho1 - rho_expected).abs().max()),
        'rho_decreases_where_vd_positive': bool((rho1[..., 0] < rho0[..., 0]).all()),
        'rho_increases_where_vd_negative': bool((rho1[..., 1] > rho0[..., 1]).all()),
        'rho_unchanged_where_v_zero': bool((rho1[..., 3] == rho0[..., 3]).all()),
        'g_unchanged': bool((g1 == g0).all()),
    }

    # cd variant: write_w channel 0 only -> g changes, rho unchanged; retention_v channel 2
    # only -> rho changes, g unchanged (no cross-talk).
    s = _ssm('Q_corr_gate_cd')
    s.write_w.data = torch.tensor([1.0] + [0.0] * 127)
    s.retention_v.data = torch.tensor([0.0, 0.0, 1.0] + [0.0] * 125)
    u = torch.randn(2, 3, CFG['dim']); p = torch.zeros(2, 3, 3)
    s.collect_gate_detail = True
    s(u, p, gate=(torch.full((2, 3), 1.0), torch.full((2, 3), 1.0)))
    ga, ra = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    s(u, p, gate=None)
    gb, rb = s.last_gate_detail['g'].clone(), s.last_gate_detail['rho'].clone()
    cd = {
        'g_changes_on_channel0': bool((ga[..., 0] != gb[..., 0]).any()),
        'rho_unchanged_on_channel0': bool((ra[..., 0] == rb[..., 0]).all()),
        'g_unchanged_on_channel2': bool((ga[..., 2] == gb[..., 2]).all()),
        'rho_changes_on_channel2': bool((ra[..., 2] != rb[..., 2]).any()),
    }
    return {'write_gate': write, 'retention_gate': retention, 'cd_no_crosstalk': cd}


# --- check 5: gradient + param update + token shape + resume delegation ----------------
def check5(seed):
    x = torch.randn(4, 2, 1024); y = torch.arange(4) % 24
    gate = {}
    for v in ('Q_corr_gate_cd', 'Q_corr_token_cd'):
        seed_all(seed)
        m = build_round5_model(dict(CFG, variant=v), seed)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        opt.zero_grad(); F.cross_entropy(m(x), y).backward()
        new_names = ([n for n, _ in m.named_parameters()
                      if n.endswith(('.write_w', '.retention_v', 'token_proj.weight'))])
        grads = {n: float(p.grad.abs().sum().item())
                 for n, p in m.named_parameters() if n in new_names}
        before = {n: p.detach().clone() for n, p in m.named_parameters() if n in new_names}
        opt.step()
        updated = {n: bool((p - before[n]).abs().max() > 0) for n, p in m.named_parameters()
                   if n in new_names}
        gate[v] = {'new_param_grad_abs_sum': grads, 'new_params_updated': updated,
                   'n_new_params': len(new_names)}
    return gate


def check5b():
    r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_qcorr_gates_resume.py',
                        '-q'], capture_output=True, text=True, cwd=ROOT)
    return {'returncode': r.returncode, 'stdout_tail': r.stdout.strip().splitlines()[-3:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='17,42,2026')
    p.add_argument('--out', default=str(ROOT / 'reports' / 'qcorr_gates' / 'acceptance'))
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for seed in [int(s) for s in a.seeds.split(',')]:
        res = {
            'seed': seed,
            'check1_hand_calc_descriptors': check1(),
            'check2_cfo_transform': check2(),
            'check3_zero_init_and_params': check3(seed),
            'check4_gate_wiring': check4(),
            'check5_gradient_and_update': check5(seed),
        }
        save_json(out / f'seed{seed}.json', res)
        print(f'seed {seed}: check1 c_dev={res["check1_hand_calc_descriptors"]["const_exp_c_max_dev"]:.2e} '
              f'd0_zero={res["check1_hand_calc_descriptors"]["const_exp_d0_exact_zero"]}, '
              f'check2 r_rot_err={res["check2_cfo_transform"]["r_common_rotation_max_abs_err"]:.2e}, '
              f'check3 max_logit_diff='
              f'{max(v["max_abs_logit_diff_vs_Q_corr"] for v in res["check3_zero_init_and_params"]["variants"].values()):.2e}, '
              f'check4 g_err={res["check4_gate_wiring"]["write_gate"]["g_formula_max_abs_err"]:.2e} '
              f'rho_err={res["check4_gate_wiring"]["retention_gate"]["rho_formula_max_abs_err"]:.2e}',
              flush=True)

    b = check5b()
    save_json(out / 'check5b_resume.json', b)
    print('check5b resume:', 'PASS' if b['returncode'] == 0 else 'FAIL', flush=True)
    print('ACCEPTANCE DONE', flush=True)


if __name__ == '__main__':
    main()
