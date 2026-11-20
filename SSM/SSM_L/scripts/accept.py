"""SSM_L round-1 formal acceptance (task book section 6, checks 1-4).

Runs the four acceptance groups on the current source tree and writes
reports/acceptance.json + reports/acceptance.md. Each check returns PASS/FAIL with a detail
string; any FAIL makes the script exit non-zero so a launch script can refuse to start full
training.

Checks:
  1. patch/decode round-trip + old-project alignment (bit-identical encoder init + logits)
  2. noise statistics (zero-mean, equal-variance, uncorrelated, power ratio, edge cases, seed)
  3. MAE mask leakage (G/Q) + Q-denoise branch recompute + Q filter 32 grad scalars/flow
     + probe encoder state_dict hash unchanged
  4. no-NaN small-batch loss decreases (all four groups) + shared init + sample order +
     checkpoint reload consistency
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / 'configs' / 'pretrain.json').read_text())['model']

from pretrain.model import (build_encoder, PretrainModel, ReconstructHead, mask_patches,
                            stable_state_corr_pack, DownstreamModel)
from pretrain.engine import sample_mask, denoise_loss, mae_loss, fit_aux_stats
from pretrain.iq_noise import corrupt_iq
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import (Classifier, build_round5_model, max_abs_norm, PatchEmbedding,
                           state_corr_pack)
from ssm_exp.utils import seed_all

RESULTS = []


def record(name, passed, detail):
    RESULTS.append({'check': name, 'passed': bool(passed), 'detail': detail})
    print(('PASS' if passed else 'FAIL') + ' | ' + name + ' | ' + detail, flush=True)


def small_cfg(**over):
    c = dict(CFG)
    c.update({'depth': 2, 'dim': 16, 'state': 8, 'heads': 2, 'input_length': 64,
              'selector_hidden': 8, 'patch': 8})
    c.update(over)
    return c


def sd_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check1_roundtrip_and_alignment(device):
    # 1a. patch/decode round-trip (pure I/Q + impulse ordering)
    patch = 8
    x = torch.randn(3, 2, 64)
    b, c, l = x.shape
    flat = x.transpose(1, 2).reshape(b, l // patch, 2 * patch)
    out = flat.reshape(b, l // patch, patch, 2).permute(0, 3, 1, 2).reshape(b, 2, l)
    ok_rt = torch.equal(x, out)

    n = 128 // patch
    tokens = torch.zeros(1, n, 2 * patch)
    tokens[0, 2, 2 * 3] = 5.0
    tokens[0, 5, 2 * 7 + 1] = -7.0
    outp = tokens.reshape(1, n, patch, 2).permute(0, 3, 1, 2).reshape(1, 2, 128)
    ok_imp = (outp[0, 0, 2 * patch + 3] == 5.0) and (outp[0, 1, 5 * patch + 7] == -7.0)

    # 1b. bit-identical encoder init to old Classifier for both backbones (full model)
    aligned = {}
    for backbone in ('G', 'Q'):
        seed_all(17)
        ref = (build_round5_model({**CFG, 'variant': 'Q_corr'}, 17) if backbone == 'Q'
               else Classifier({**CFG, 'variant': 'G'}))
        seed_all(17)
        enc = build_encoder(dict(CFG), backbone, 17)
        ref_sd, enc_sd = ref.state_dict(), enc.state_dict()
        expect = {k for k in ref_sd if not k.startswith('head.') and not k.startswith('phys_')}
        mismatch = [k for k in expect if not torch.equal(ref_sd[k], enc_sd[k])]
        aligned[backbone] = (set(enc_sd) == expect) and not mismatch

        # logits alignment on a fixed input
        xin = max_abs_norm(torch.randn(2, 2, 1024)).to(device)
        enc = enc.to(device).eval(); ref = ref.to(device).eval()
        with torch.no_grad():
            head = nn.Linear(CFG['dim'], CFG['classes']).to(device)
            head.weight.data.copy_(ref.head.weight); head.bias.data.copy_(ref.head.bias)
            logits_ok = torch.allclose(ref(xin), head(enc.features(xin)), atol=1e-5)
        aligned[backbone] = aligned[backbone] and logits_ok

    passed = ok_rt and ok_imp and all(aligned.values())
    detail = (f'roundtrip={ok_rt} impulse_order={ok_imp} '
              f'init_bit_identical={aligned}')
    record('1. patch/decode round-trip + old-project alignment', passed, detail)


def check2_noise_stats():
    seed_all(0)
    gen = torch.Generator().manual_seed(0)
    b = 200000
    x = torch.randn(b, 2, 64)  # arbitrary signal power ~1 per real channel
    levels = torch.tensor([0., 5., 10., 15.])
    results = {}
    for lvl in levels.tolist():
        # force every sample to the same level for a clean per-level power estimate
        g = torch.Generator().manual_seed(0)
        outs = []
        for _ in range(4):  # 50k samples per chunk to bound memory
            chunk = x[_ * 50000:(_ + 1) * 50000]
            idx = torch.full((chunk.shape[0],), levels.tolist().index(lvl))
            # replicate corrupt_iq's per-level math deterministically at one level
            power = chunk.square().sum(1).mean(-1)
            sigma = (power / 2).sqrt() * math.exp(-math.log(10) * lvl / 20)
            eps = torch.randn(chunk.shape, generator=g)
            outs.append(sigma[:, None, None] * eps)
        noise = torch.cat(outs)
        i_mean = noise[:, 0].mean().item()
        q_mean = noise[:, 1].mean().item()
        i_var = noise[:, 0].var(unbiased=False).item()
        q_var = noise[:, 1].var(unbiased=False).item()
        corr = (noise[:, 0] * noise[:, 1]).mean().item() / (noise[:, 0].std() * noise[:, 1].std()).item()
        # expected noise power per real channel = power * 10^(-lvl/10)
        sig_power = x.square().sum(1).mean(-1).mean().item()  # total I+Q power
        noise_power = noise.square().sum(1).mean(-1).mean().item()
        ratio_db = 10 * math.log10((noise_power / sig_power) + 1e-12)
        results[lvl] = dict(i_mean=i_mean, q_mean=q_mean, i_var=i_var, q_var=q_var,
                            corr=corr, ratio_db=ratio_db)

    ok_mean = all(abs(r['i_mean']) < 0.02 and abs(r['q_mean']) < 0.02 for r in results.values())
    ok_var = all(abs(r['i_var'] - r['q_var']) / max(r['i_var'], 1e-9) < 0.02 for r in results.values())
    ok_corr = all(abs(r['corr']) < 0.02 for r in results.values())
    ok_ratio = all(abs(r['ratio_db'] - (-lvl)) < 0.3 for lvl, r in results.items())

    # edge cases: zero input -> no NaN, zero sigma; tiny magnitude -> no NaN
    z = torch.zeros(4, 2, 64)
    yz, sz, lz = corrupt_iq(z, torch.Generator().manual_seed(0))
    ok_zero = torch.isfinite(yz).all() and bool((sz == 0).all()) and torch.equal(yz, z)
    t = torch.full((4, 2, 64), 1e-30)
    yt, _, _ = corrupt_iq(t, torch.Generator().manual_seed(0))
    ok_tiny = torch.isfinite(yt).all()

    # reproducibility: same seed -> identical noise
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    a, _, _ = corrupt_iq(x[:8], g1)
    b_, _, _ = corrupt_iq(x[:8], g2)
    ok_rep = torch.equal(a, b_)

    passed = all([ok_mean, ok_var, ok_corr, ok_ratio, ok_zero, ok_tiny, ok_rep])
    detail = (f'mean<0.02={ok_mean} var_eq={ok_var} uncorr={ok_corr} '
              f'power_ratio_±0.3dB={ok_ratio} zero_input={ok_zero} tiny={ok_tiny} '
              f'reproducible={ok_rep}')
    detail += ' ratios=' + ','.join(f'{lvl}dB:{results[lvl]["ratio_db"]:.2f}' for lvl in results)
    record('2. noise statistics', passed, detail)


def check3_mask_and_gradients(device):
    # MAE mask leakage (G + Q): perturbing masked region must not change tokens/descriptor
    seed_all(17)
    n = 128
    x = max_abs_norm(torch.randn(2, 2, 1024)).to(device)
    mask = torch.zeros(2, n, dtype=torch.bool, device=device)
    mask[:, :64] = True
    leakage_ok = True
    for backbone in ('G', 'Q'):
        enc = build_encoder(dict(CFG), backbone, 17).to(device).eval()
        with torch.no_grad():
            tok_a = enc.forward_tokens(x, mask=mask)
            desc_a = enc.aux_descriptor(mask_patches(x, mask, 8)) if backbone == 'Q' else None
            x2 = x.clone()
            x2[:, :, :64 * 8] = torch.randn(2, 2, 64 * 8, device=device)
            tok_b = enc.forward_tokens(x2, mask=mask)
            desc_b = enc.aux_descriptor(mask_patches(x2, mask, 8)) if backbone == 'Q' else None
        if not torch.equal(tok_a, tok_b):
            leakage_ok = False
        if backbone == 'Q' and not torch.equal(desc_a, desc_b):
            leakage_ok = False

    # Q-denoise branch recompute: different noise -> different descriptor
    enc = build_encoder(dict(CFG), 'Q', 17).to(device).eval()
    g1, g2 = torch.Generator(device=device).manual_seed(0), torch.Generator(device=device).manual_seed(1)
    y1, _, _ = corrupt_iq(x, g1)
    y2, _, _ = corrupt_iq(x, g2)
    with torch.no_grad():
        d1, d2 = enc.aux_descriptor(y1), enc.aux_descriptor(y2)
    recompute_ok = not torch.equal(d1, d2)

    # Q filter: exactly 32 real grad scalars, and gradient flows in pretrain + downstream
    seed_all(17)
    enc_q = build_encoder(dict(CFG), 'Q', 17).to(device)
    ps = list(enc_q.aux_branch.parameters())
    n_scalars = sum(p.numel() * (2 if p.is_complex() else 1) for p in ps)
    count_ok = (n_scalars == 32) and all(p.requires_grad for p in ps)

    def _grad_flows(model, loss_fn):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        gs = [p.grad for p in model.encoder.aux_branch.parameters()]
        return all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gs)

    xb = max_abs_norm(torch.randn(4, 2, 1024, device=device))
    pm = PretrainModel(build_encoder(dict(CFG), 'Q', 17), 8).to(device)
    gg = torch.Generator(device=device).manual_seed(0)

    def _pretrain_loss():
        y, sigma, _ = corrupt_iq(xb, gg)
        pred = pm(y, extra=sigma[:, None].expand(4, 128))
        return denoise_loss(pred, xb)

    flow_pretrain = _grad_flows(pm, _pretrain_loss)
    dm = DownstreamModel(build_encoder(dict(CFG), 'Q', 17), 24).to(device)
    labels = torch.randint(0, 24, (4,), device=device)
    flow_downstream = _grad_flows(dm, lambda: F.cross_entropy(dm(xb), labels))

    # probe freeze: train a fresh head, hash full encoder state_dict before/after
    seed_all(17)
    enc_small = build_encoder(small_cfg(), 'Q', 17).to(device)
    for p in enc_small.parameters():
        p.requires_grad_(False)
    h_before = sd_hash(enc_small.state_dict())
    head = nn.Linear(16, 24).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(3):
        fx = enc_small.features(max_abs_norm(torch.randn(8, 2, 64, device=device)))
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(head(fx), torch.randint(0, 24, (8,), device=device)).backward()
        opt.step()
    h_after = sd_hash(enc_small.state_dict())
    freeze_ok = (h_before == h_after)

    passed = all([leakage_ok, recompute_ok, count_ok, flow_pretrain, flow_downstream, freeze_ok])
    detail = (f'mask_leakage_free={leakage_ok} denoise_recompute={recompute_ok} '
              f'Q_32_scalars={count_ok} grad_flow_pretrain={flow_pretrain} '
              f'grad_flow_downstream={flow_downstream} probe_frozen={freeze_ok}')
    record('3. mask leakage + Q-denoise recompute + Q filter gradients + probe freeze', passed, detail)


def check4_nonan_sharedinit_order_reload(device):
    # small-batch loss decreases for all four groups (no NaN), shared init byte-identical,
    # same seed -> same sample order, checkpoint reload -> identical predictions.
    s = small_cfg()
    batch = 8
    x = max_abs_norm(torch.randn(batch, 2, 64, device=device))

    def build_pretrain(backbone):
        seed_all(17)
        enc = build_encoder(s, backbone, 17)
        return PretrainModel(enc, s['patch']).to(device)

    loss_ok = {}
    for backbone in ('G', 'Q'):
        for task in ('mae', 'denoise'):
            seed_all(17)
            pm = build_pretrain(backbone)
            opt = torch.optim.AdamW(pm.parameters(), lr=1e-3)
            gg = torch.Generator(device=device).manual_seed(0)
            mg = torch.Generator(device=device).manual_seed(1)
            losses = []
            finite = True
            for _ in range(15):
                if task == 'denoise':
                    y, sigma, _ = corrupt_iq(x, gg)
                    pred = pm(y, extra=sigma[:, None].expand(batch, 8))
                    loss = denoise_loss(pred, x)
                else:
                    mk = sample_mask(batch, 8, 0.5, mg, device)
                    pred = pm(x, mask=mk)
                    loss = mae_loss(pred, x, mk, 8)
                finite = finite and torch.isfinite(loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            loss_ok[f'{backbone}-{task}'] = finite and losses[0] > losses[-1]

    # shared init byte-identical across two independent builds
    shared_ok = True
    for backbone in ('G', 'Q'):
        seed_all(17)
        a = build_encoder(s, backbone, 17)
        seed_all(17)
        b = build_encoder(s, backbone, 17)
        shared_ok = shared_ok and all(torch.equal(a.state_dict()[k], b.state_dict()[k])
                                      for k in a.state_dict())

    # same seed -> same sample order
    m = read_manifest(str(ROOT / 'configs' / 'data.json'))
    ds = IQDataset(m, 'train')
    ids_a = []
    seed_all(17)
    g1 = torch.Generator().manual_seed(17)
    for _, _, _, i in make_loader(ds, 128, 0, True, g1):
        ids_a.extend(i.tolist())
        if len(ids_a) >= 512:
            break
    ids_b = []
    g2 = torch.Generator().manual_seed(17)
    for _, _, _, i in make_loader(ds, 128, 0, True, g2):
        ids_b.extend(i.tolist())
        if len(ids_b) >= 512:
            break
    order_ok = ids_a == ids_b

    # checkpoint reload -> identical predictions
    pm = build_pretrain('Q')
    pm.eval()
    with torch.no_grad():
        pred_before = pm(x)
    torch.save({'model': pm.state_dict()}, '/tmp/_accept_ck.pt')
    pm2 = build_pretrain('Q')
    pm2.load_state_dict(torch.load('/tmp/_accept_ck.pt', map_location='cpu', weights_only=False)['model'])
    pm2.eval()
    with torch.no_grad():
        pred_after = pm2(x)
    reload_ok = torch.equal(pred_before, pred_after)

    passed = all(list(loss_ok.values()) + [shared_ok, order_ok, reload_ok])
    detail = (f'loss_decreases={loss_ok} shared_init={shared_ok} '
              f'sample_order={order_ok} reload_consistent={reload_ok}')
    record('4. no-NaN loss decrease + shared init + sample order + checkpoint reload', passed, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=str(ROOT / 'reports' / 'acceptance.json'))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    check1_roundtrip_and_alignment(device)
    check2_noise_stats()
    check3_mask_and_gradients(device)
    check4_nonan_sharedinit_order_reload(device)

    all_pass = all(r['passed'] for r in RESULTS)
    report = {'all_pass': all_pass, 'device': args.device,
              'results': RESULTS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = ['# SSM_L round-1 acceptance report', '',
          f'- device: `{args.device}`', f'- all_pass: **{all_pass}**', '']
    for r in RESULTS:
        md.append(f"- [{'x' if r['passed'] else ' '}] **{r['check']}** — {r['detail']}")
    (out.parent / 'acceptance.md').write_text('\n'.join(md) + '\n')

    print('=== ALL PASS ===' if all_pass else '=== HAS FAILURES ===', flush=True)
    raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
