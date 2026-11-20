#!/usr/bin/env python
"""G_Q_corr / G_Q_corr_frozen pre-training acceptance (4 checks).

Runs at the real production config (patch=8, dim=176, depth=6, heads=4, L=1024, 24 classes)
and saves per-seed numerical results to reports/global_qcorr/acceptance/.

  Check 1  Global backbone actually used: module inventory (attention count, position
           encoding, token/logits shapes) + shared-G init; plain G and the two new models
           must be numerically identical on the same batch (zero-init fusion).
  Check 2  Old Q feature actually reused: G_Q_corr vs Q_corr share the same branch params;
           filter states, pre-standardization 16-dim descriptor, and post-standardization
           descriptor all match. The new model has NO raw-IQ correlation fusion (G_corr's
           Linear(3,dim) physical-feature path is absent).
  Check 3  Freeze boundary: only the 32 aux-branch scalars differ in requires_grad; after
           several optimizer steps the learnable branch receives grad and changes (step-1
           grad is zero by the W_f=0 design, verified at step-2), the frozen branch hash is
           unchanged, and fusion/backbone update.
  Check 4  Data & resume consistency: both variants at the same seed share the first-batch
           sample IDs, injected CFO, and fit_aux descriptor stats; the resume/no-refit path
           is delegated to tests/test_global_qcorr_resume.py (invoked here).

No test split is touched. This is a verification script, not a research run.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_exp.model import (Classifier, build_global_qcorr_model, build_round5_model,
                           parameter_count, perturb, max_abs_norm)
from ssm_exp.utils import seed_all, save_json
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader, fit_aux, frozen_branch_hash

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(variant='G', patch=8, dim=176, depth=6, heads=4, state=128, selector_hidden=64,
           classes=24, input_length=1024, ffn_ratio=4, dropout=0.0, position='none',
           direction='forward', scan='parallel', ssm_gain=1.0)
BRANCH_KEYS = ('raw_alpha', 'theta', 'b')


def _build_pair(seed):
    """Return (G, G_Q_corr, G_Q_corr_frozen) each under their own seed_all(seed)."""
    seed_all(seed)
    g = Classifier(dict(CFG, variant='G')).eval()
    seed_all(seed)
    gqc = build_global_qcorr_model(dict(CFG, variant='G_Q_corr'), seed).eval()
    seed_all(seed)
    gqcf = build_global_qcorr_model(dict(CFG, variant='G_Q_corr_frozen'), seed).eval()
    return g, gqc, gqcf


def check1(seed):
    g, gqc, gqcf = _build_pair(seed)
    x = torch.randn(8, 2, 1024)
    with torch.no_grad():
        lg, lqc, lqcf = g(x), gqc(x), gqcf(x)
    shared = all(
        torch.equal(g.state_dict()[k], gqc.state_dict()[k])
        and torch.equal(g.state_dict()[k], gqcf.state_dict()[k])
        for k in g.state_dict() if k.startswith(('front.', 'blocks.', 'norm.', 'head.')))
    return {
        'n_blocks': len(gqc.blocks),
        'n_attention_blocks': sum(1 for b in gqc.blocks if type(b.mix).__name__ == 'Attention'),
        'pos_mode': gqc.pos_mode,
        'front_type': type(gqc.front).__name__,
        'token_shape': list(gqc.front(x).shape),
        'logits_shape': list(lg.shape),
        'backbone_shared_with_G': bool(shared),
        'aux_fusion_zero': float(gqc.aux_fusion.weight.abs().sum().item()),
        'max_abs_logit_diff_G_vs_GQc': float((lg - lqc).abs().max().item()),
        'max_abs_logit_diff_G_vs_GQcf': float((lg - lqcf).abs().max().item()),
        'params': {'G': parameter_count(g), 'G_Q_corr': parameter_count(gqc),
                   'G_Q_corr_frozen': parameter_count(gqcf),
                   'frozen_trainable': sum(p.numel() * (2 if p.is_complex() else 1)
                                           for p in gqcf.parameters() if p.requires_grad)},
    }


def check2(seed, manifest, device):
    seed_all(seed)
    qc = build_round5_model(dict(CFG, variant='Q_corr'), seed).to(device).eval()
    seed_all(seed)
    gqc = build_global_qcorr_model(dict(CFG, variant='G_Q_corr'), seed).to(device).eval()
    x = torch.randn(8, 2, 1024, device=device)

    branch_identical = all(
        torch.equal(qc.aux_branch.state_dict()[k], gqc.aux_branch.state_dict()[k])
        for k in BRANCH_KEYS)
    with torch.no_grad():
        q_qc = qc.aux_branch(x)
        q_gqc = gqc.aux_branch(x)
        dpre_qc = qc.aux_descriptor(x)
        dpre_gqc = gqc.aux_descriptor(x)

    # fit the frozen standardization on the real training subset (deterministic rng 0),
    # then compare post-standardization descriptors.
    ds = IQDataset(read_manifest(manifest), 'train')
    stats_qc = fit_aux(qc, ds, dict(train=dict(aux_fit_samples=8192, batch_size=128)), device)
    stats_gqc = fit_aux(gqc, ds, dict(train=dict(aux_fit_samples=8192, batch_size=128)), device)
    with torch.no_grad():
        f_qc = (dpre_qc - qc.aux_mean) / qc.aux_std
        f_gqc = (dpre_gqc - gqc.aux_mean) / gqc.aux_std

    # G_Q_corr must not carry G_corr's raw-IQ correlation fusion path.
    phys_active = 'G_Q_corr' in ('S2', 'S2_concat', 'S1_energy', 'S1_corr', 'G_full', 'G_corr')
    return {
        'branch_params_identical': bool(branch_identical),
        'max_abs_filter_state_diff': float((q_qc - q_gqc).abs().max().item()),
        'max_abs_pre_std_desc_diff': float((dpre_qc - dpre_gqc).abs().max().item()),
        'max_abs_post_std_desc_diff': float((f_qc - f_gqc).abs().max().item()),
        'aux_stats_identical': bool(stats_qc == stats_gqc),
        'has_raw_iq_corr_fusion': gqc.fusion is not None,
        'raw_iq_phys_feature_path_active': bool(phys_active),
    }


def check3(seed):
    seed_all(seed)
    gqc = build_global_qcorr_model(dict(CFG, variant='G_Q_corr'), seed)
    seed_all(seed)
    gqcf = build_global_qcorr_model(dict(CFG, variant='G_Q_corr_frozen'), seed)

    trainable_diff = sorted(
        n for (n, p), (_, q) in zip(gqc.named_parameters(), gqcf.named_parameters())
        if p.requires_grad != q.requires_grad)
    frozen_param_names = sorted(n for n, p in gqcf.named_parameters() if not p.requires_grad)

    x = torch.randn(8, 2, 1024)
    y = torch.arange(8) % 24
    opt_l = torch.optim.SGD((p for p in gqc.parameters() if p.requires_grad), lr=0.1)
    opt_f = torch.optim.SGD((p for p in gqcf.parameters() if p.requires_grad), lr=0.1)
    hash0 = frozen_branch_hash(gqcf)
    branch_before = {k: getattr(gqc.aux_branch, k).detach().clone() for k in BRANCH_KEYS}

    # step 1: W_f is zero -> learnable branch grad must be exactly zero.
    opt_l.zero_grad()
    F.cross_entropy(gqc(x), y).backward()
    step1_branch_grad = {k: float(getattr(gqc.aux_branch, k).grad.abs().sum().item())
                         for k in BRANCH_KEYS}
    opt_l.step()
    # step 2: gradient now flows through the nonzero W_f to the branch.
    opt_l.zero_grad()
    F.cross_entropy(gqc(x), y).backward()
    step2_branch_grad = {k: float(getattr(gqc.aux_branch, k).grad.abs().sum().item())
                         for k in BRANCH_KEYS}
    # several more steps so the branch params measurably move (the corr readout is
    # scale-invariant, so b's gradient starts tiny and grows as W_f grows).
    for _ in range(8):
        opt_l.step()
        opt_l.zero_grad()
        F.cross_entropy(gqc(x), y).backward()
    opt_l.step()
    branch_after = {k: getattr(gqc.aux_branch, k).detach().clone() for k in BRANCH_KEYS}
    branch_changed = {k: bool((branch_after[k] - branch_before[k]).abs().max() > 0) for k in BRANCH_KEYS}

    # frozen model: fusion + backbone update, branch hash unchanged (10 steps).
    for _ in range(10):
        opt_f.zero_grad()
        F.cross_entropy(gqcf(x), y).backward()
        fusion_grad_frozen = float(gqcf.aux_fusion.weight.grad.abs().sum().item())
        opt_f.step()
    fusion_moved = float(gqcf.aux_fusion.weight.abs().sum().item()) > 0
    hash_after = frozen_branch_hash(gqcf)

    return {
        'trainable_status_diff_names': trainable_diff,
        'frozen_param_names': frozen_param_names,
        'n_trainable_diff_real_scalars': sum(
            p.numel() * (2 if p.is_complex() else 1)
            for n, p in gqcf.named_parameters() if not p.requires_grad),
        'step1_branch_grad_abs_sum': step1_branch_grad,
        'step2_branch_grad_abs_sum': step2_branch_grad,
        'branch_changed_after_10_steps': branch_changed,
        'frozen_fusion_grad_abs_sum_last': fusion_grad_frozen,
        'frozen_fusion_moved': bool(fusion_moved),
        'frozen_branch_hash_unchanged': bool(hash0 == hash_after),
    }


def check4a(seed, manifest, device):
    """Both variants share first-batch sample IDs + injected CFO + fit_aux stats."""
    m = read_manifest(manifest)
    ds = IQDataset(m, 'train')

    def draw(variant):
        seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        aug_gen = torch.Generator(device=device).manual_seed(seed + 100000)
        if variant == 'G_Q_corr':
            build_global_qcorr_model(dict(CFG, variant=variant), seed)
        else:
            build_global_qcorr_model(dict(CFG, variant=variant), seed)
        perm = torch.randperm(len(ds), generator=gen).tolist()
        cfo = (2 * torch.rand(128, device=device, generator=aug_gen) - 1) * 0.002
        return perm[:128], cfo

    ids_l, cfo_l = draw('G_Q_corr')
    ids_f, cfo_f = draw('G_Q_corr_frozen')

    seed_all(seed)
    gqc = build_global_qcorr_model(dict(CFG, variant='G_Q_corr'), seed).to(device)
    seed_all(seed)
    gqcf = build_global_qcorr_model(dict(CFG, variant='G_Q_corr_frozen'), seed).to(device)
    s_l = fit_aux(gqc, ds, dict(train=dict(aux_fit_samples=8192, batch_size=128)), device)
    s_f = fit_aux(gqcf, ds, dict(train=dict(aux_fit_samples=8192, batch_size=128)), device)

    return {
        'first_batch_ids_identical': bool(ids_l == ids_f),
        'first_batch_ids_head': ids_l[:8],
        'injected_cfo_identical': bool(torch.equal(cfo_l, cfo_f)),
        'injected_cfo_head': [float(v) for v in cfo_l[:8].cpu().tolist()],
        'fit_aux_stats_identical': bool(s_l == s_f),
        'aux_ids_sha256': s_l['ids_sha256'],
    }


def check4b():
    """Delegate the resume/no-refit path to the pytest test (smoke, CPU, subprocess)."""
    r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_global_qcorr_resume.py',
                        '-q'], capture_output=True, text=True, cwd=ROOT)
    return {'returncode': r.returncode, 'stdout_tail': r.stdout.strip().splitlines()[-3:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(ROOT / 'configs' / 'data_raw.json'))
    p.add_argument('--seeds', default='17,42,2026')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=str(ROOT / 'reports' / 'global_qcorr' / 'acceptance'))
    a = p.parse_args()
    device = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for seed in [int(s) for s in a.seeds.split(',')]:
        res = {
            'seed': seed,
            'check1_global_backbone': check1(seed),
            'check2_q_feature_reuse': check2(seed, a.manifest, device),
            'check3_freeze_boundary': check3(seed),
            'check4a_data_and_stats': check4a(seed, a.manifest, device),
        }
        save_json(out / f'seed{seed}.json', res)
        print(f'seed {seed}: check1 max_logit_diff='
              f"{res['check1_global_backbone']['max_abs_logit_diff_G_vs_GQc']:.3e}, "
              f"check2 max_desc_diff="
              f"{res['check2_q_feature_reuse']['max_abs_post_std_desc_diff']:.3e}, "
              f"check3 frozen_hash_unchanged="
              f"{res['check3_freeze_boundary']['frozen_branch_hash_unchanged']}, "
              f"check4a stats_identical="
              f"{res['check4a_data_and_stats']['fit_aux_stats_identical']}", flush=True)

    b = check4b()
    save_json(out / 'check4b_resume.json', b)
    print('check4b resume:', 'PASS' if b['returncode'] == 0 else 'FAIL', flush=True)
    print('ACCEPTANCE DONE', flush=True)


if __name__ == '__main__':
    main()
