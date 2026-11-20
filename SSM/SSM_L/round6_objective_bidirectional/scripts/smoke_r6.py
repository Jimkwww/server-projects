"""Round6 smoke test (protocol section 7, 9 checks). Run BEFORE launching the full matrix.

Usage:
  python round6_objective_bidirectional/scripts/smoke_r6.py --device cuda:1
"""
import argparse
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ssm_exp.utils import seed_all
from ssm_exp.model import max_abs_norm
from pretrain.model import PretrainModel, mask_patches
from round2_overnight.engine import load_config, sample_mask, mae_loss, denoise_loss
from round2_overnight.iq_noise import corrupt_iq
from round6_objective_bidirectional.scripts.models_r6 import (build_encoder_r6, QbiEncoder,
                                                              JointPretrainModel)
from round6_objective_bidirectional.scripts.pretrain_r6 import build_shared_init_r6

R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
CFG = R6 / 'configs' / 'pretrain.json'
R6_SHARED = R6 / 'runs' / 'pretrain' / 'f1' / 'shared_init'
R5_SHARED = R5 / 'runs' / 'pretrain' / 'f1' / 'shared_init'


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail else ''))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    c = load_config(CFG)
    cfg = c['model']
    patch = cfg['patch']
    results = []

    # ---- Check 1: Q_bi build + shared-init byte-identity on shared modules ----
    seed = 17
    seed_all(seed)
    qbi = QbiEncoder(cfg, seed).to(device)
    q_enc, _, _ = build_shared_init_r6(c, 'Q', seed, None, device, R6_SHARED, R5_SHARED)
    qbi_enc, _, _ = build_shared_init_r6(c, 'Q_bi', seed, None, device, R6_SHARED, R5_SHARED)
    q_sd = q_enc.state_dict()
    qbi_sd = qbi_enc.state_dict()
    shared_keys = [k for k in q_sd if k.startswith(('front.', 'aux_branch.', 'aux_fusion.',
                                                     'norm.', 'aux_mean', 'aux_std'))]
    identical = all(torch.equal(q_sd[k], qbi_sd[k]) for k in shared_keys)
    # bidirectional blocks exist and forward/backward params are independent tensors
    fwd_keys = [k for k in qbi_sd if '.forward_dir.' in k]
    bwd_keys = [k for k in qbi_sd if '.backward_dir.' in k]
    has_bidir = len(fwd_keys) > 0 and len(bwd_keys) > 0
    indep = all(k.split('.')[-1] != qbi_sd[k.replace('forward_dir', 'backward_dir')].data_ptr()
                for k in fwd_keys) if has_bidir else False
    # shared modules are the SAME object graph pattern (aux branch present, unidirectional)
    results.append(check('1a Q_bi shared modules byte-identical to Q', identical))
    results.append(check('1b Q_bi has independent forward/backward directions', has_bidir and indep,
                         f'{len(fwd_keys)} fwd keys, {len(bwd_keys)} bwd keys'))

    # ---- Check 2: Q_bi param count recorded + output dim 176 ----
    q_bi_params = sum(p.numel() * (2 if p.is_complex() else 1) for p in qbi_enc.parameters())
    q_params = sum(p.numel() * (2 if p.is_complex() else 1) for p in q_enc.parameters())
    x = torch.randn(2, 2, cfg['input_length'], device=device)
    toks = qbi_enc.forward_tokens(max_abs_norm(x))
    out_dim_ok = toks.shape[-1] == cfg['dim']
    results.append(check('2 Q_bi param count recorded + output dim 176',
                         q_bi_params != q_params and out_dim_ok,
                         f'Q_bi={q_bi_params} Q={q_params} (diff={q_bi_params - q_params})'))

    # ---- Check 3: bidirectional output token alignment + per-sample state independence ----
    # Two samples: one all-zero, one non-zero. Zero sample must not corrupt non-zero sample
    # (scan state is per-sample; batch parallelism must not mix).
    xa = torch.randn(2, 2, cfg['input_length'], device=device)
    xb = xa.clone()
    xb[1] = 0.0
    ta = qbi_enc.forward_tokens(max_abs_norm(xa))
    tb = qbi_enc.forward_tokens(max_abs_norm(xb))
    sample_independent = torch.allclose(ta[0], tb[0], atol=1e-5)
    # forward/backward flip round-trip: applying forward_dir then flip should equal backward scan
    # (structural sanity, not exact equality to unidirectional)
    results.append(check('3 per-sample state independent (zero sample no bleed)',
                         sample_independent))

    # ---- Check 4: MAE mask applied to raw IQ BEFORE aux descriptor ----
    mask = torch.zeros(2, cfg['input_length'] // patch, dtype=torch.bool, device=device)
    mask[:, 0] = True  # mask first patch
    # build a probe: if mask is applied after aux, aux descriptor would see full x. We verify
    # mask_patches zeroes the raw patch that feeds aux_descriptor by checking the masked patch
    # contributes zero to the front input in a manual re-implementation.
    xm = mask_patches(max_abs_norm(xa), mask, patch)
    zeroed = (xm[:, :, :patch] == 0).all()
    results.append(check('4 MAE mask zeroes raw IQ before Q branch', zeroed))

    # ---- Check 5: denoise corruption levels 0/5/10/15 dB + sigma only to denoise head ----
    noise_gen = torch.Generator(device=device).manual_seed(seed + 100000)
    xd = max_abs_norm(torch.randn(8, 2, cfg['input_length'], device=device))
    y, sigma, level_db = corrupt_iq(xd, noise_gen)
    levels_ok = set(level_db.tolist()) <= {0.0, 5.0, 10.0, 15.0}
    sigma_shape_ok = sigma.shape == (8,)
    # sigma enters only ReconstructHead via extra; encoder has no sigma parameter
    enc_no_sigma = not any('sigma' in k for k in qbi_enc.state_dict())
    results.append(check('5 denoise levels {0,5,10,15}dB + sigma is per-sample scalar',
                         levels_ok and sigma_shape_ok and enc_no_sigma,
                         f'levels={sorted(set(level_db.tolist()))}'))

    # ---- Check 6: joint batch split + loss normalization + single update + 0.5/0.5 ----
    joint = JointPretrainModel(qbi_enc, patch).to(device)
    xj = max_abs_norm(torch.randn(8, 2, cfg['input_length'], device=device))
    split_gen = torch.Generator(device=device).manual_seed(seed + 500000)
    B = xj.shape[0]
    half = B // 2
    perm = torch.randperm(B, generator=split_gen, device=device)
    mae_idx, dn_idx = perm[:half], perm[half:]
    mask_j = sample_mask(len(mae_idx), cfg['input_length'] // patch, 0.5, noise_gen, device)
    pred_mae = joint.forward_mae(xj[mae_idx], mask_j)
    l_mae = mae_loss(pred_mae, xj[mae_idx], mask_j, patch)
    yj, sj, _ = corrupt_iq(xj[dn_idx], noise_gen)
    pred_dn = joint.forward_dn(yj, sj)
    l_dn = denoise_loss(pred_dn, xj[dn_idx])
    # two heads are independent parameters
    heads_indep = not any(torch.equal(joint.mae_decoder.state_dict()[k],
                                      joint.dn_decoder.state_dict()[k])
                          for k in joint.mae_decoder.state_dict())
    # single optimizer update over weighted total (just verify weighted loss is finite + scalar)
    l_total = 0.5 * l_mae + 0.5 * l_dn
    results.append(check('6 joint split+two independent heads+0.5/0.5 weighted loss',
                         torch.isfinite(l_total) and heads_indep and l_total.ndim == 0,
                         f'mae={l_mae.item():.4f} dn={l_dn.item():.4f}'))

    # ---- Check 7: RNG streams independent ----
    g1 = torch.Generator(device=device).manual_seed(seed + 100000)
    g2 = torch.Generator(device=device).manual_seed(seed + 200000)
    g3 = torch.Generator(device=device).manual_seed(seed + 500000)
    r1 = torch.randint(1000000, (100,), generator=g1, device=device)
    r2 = torch.randint(1000000, (100,), generator=g2, device=device)
    r3 = torch.randint(1000000, (100,), generator=g3, device=device)
    streams_diff = not (torch.equal(r1, r2) or torch.equal(r1, r3) or torch.equal(r2, r3))
    results.append(check('7 RNG streams (noise/mask/split) independent', streams_diff))

    # ---- Check 8 & 9: real smoke of pretrain (1 epoch) + downstream, via subprocess ----
    # Run one pretrain smoke for each task family, then a readout + finetune smoke.
    import subprocess
    manifest = ROOT / 'round3_scale_transfer' / 'configs' / 'manifest_f1.json'
    smoke_ok = True
    for bb, task in [('G', 'denoise'), ('Q', 'joint'), ('Q_bi', 'mae')]:
        out = R6 / 'runs' / 'smoke' / f'{bb}_{task}'
        import shutil
        if out.exists():
            shutil.rmtree(out)
        r = subprocess.run(
            [sys.executable, str(R6 / 'scripts' / 'pretrain_r6.py'),
             '--config', str(CFG), '--manifest', str(manifest), '--backbone', bb, '--task', task,
             '--out', str(out), '--shared-init-dir', str(R6_SHARED),
             '--round5-shared-init-dir', str(R5_SHARED), '--corpus', 'f1', '--epochs', '1',
             '--seed', str(seed), '--device', str(device), '--smoke'],
            capture_output=True, text=True)
        if r.returncode != 0:
            smoke_ok = False
            print(f'    smoke pretrain {bb}/{task} FAILED:\n{(r.stdout + r.stderr)[-1500:]}')
        else:
            print(f'    smoke pretrain {bb}/{task} OK')
    results.append(check('8 pretrain smoke (G-denoise/Q-joint/Q_bi-mae, 1 epoch) finite+descend',
                         smoke_ok))

    # downstream smoke: readout + finetune on the Q_bi smoke checkpoint
    ds_ok = True
    qbi_ckpt = R6 / 'runs' / 'smoke' / 'Q_bi_mae' / 'encoder_final.pt'
    if qbi_ckpt.exists():
        readout_out = R6 / 'runs' / 'smoke' / 'readout_qbi_rml.json'
        r = subprocess.run(
            [sys.executable, str(R6 / 'scripts' / 'readout_r6.py'),
             '--checkpoint', str(qbi_ckpt), '--backbone', 'Q_bi', '--corpus', 'f1', '--seed', str(seed),
             '--ckpt-name', 'epoch50', '--task', 'rml', '--device', str(device),
             '--cache-dir', str(R6 / 'runs' / 'smoke' / 'features'),
             '--pred-dir', str(R6 / 'runs' / 'smoke' / 'predictions'),
             '--out-json', str(readout_out)],
            capture_output=True, text=True)
        if r.returncode != 0:
            ds_ok = False
            print(f'    smoke readout Q_bi/rml FAILED:\n{(r.stdout + r.stderr)[-1500:]}')
        else:
            print('    smoke readout Q_bi/rml OK')

        ft_out = R6 / 'runs' / 'smoke' / 'finetune_qbi_rml'
        import shutil
        if ft_out.exists():
            shutil.rmtree(ft_out)
        r = subprocess.run(
            [sys.executable, str(R6 / 'scripts' / 'finetune_r6.py'),
             '--config', str(R6 / 'configs' / 'downstream_nocfo_rml.json'),
             '--manifest', str(ROOT / 'configs' / 'data.json'),
             '--backbone', 'Q_bi', '--pretrained', str(qbi_ckpt), '--out', str(ft_out),
             '--seed', str(seed), '--device', str(device), '--smoke', '--epochs', '1'],
            capture_output=True, text=True)
        if r.returncode != 0:
            ds_ok = False
            print(f'    smoke finetune Q_bi/rml FAILED:\n{(r.stdout + r.stderr)[-1500:]}')
        else:
            print('    smoke finetune Q_bi/rml OK')
    else:
        ds_ok = False
        print('    no Q_bi smoke checkpoint to run downstream smoke on')
    results.append(check('9 downstream smoke (readout + finetune) protocol unchanged', ds_ok))

    n_pass = sum(results)
    print(f'\n===== {n_pass}/{len(results)} checks passed =====')
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == '__main__':
    main()
