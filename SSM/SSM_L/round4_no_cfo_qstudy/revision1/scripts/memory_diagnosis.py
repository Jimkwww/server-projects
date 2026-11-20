"""Revision-1 Section 4: corrected SSM effective-memory diagnosis on S1plain.

Two groups, both using a *correct* state cutoff (zero the recurrence state at segment
boundaries; NOT detach; zero-init at the start of every segment; all layers share the same
boundary):

A group (no downstream supervision): pretrained S1plain encoders (random / MAE / denoise, 3
seeds). Two frozen readouts:
  (1) fixed  -- StandardScaler + L2 LR(C=1,lbfgs) fit once on FULL-state train features,
                evaluated on each K's features (same scaler/classifier).
  (2) refit  -- per-K: scaler + LR refit on that K's train features, evaluated on that K.
B group (supervised): finetuned S1plain downstream models (scratch / MAE / denoise, RML+radar,
3 seeds; RML scratch wired in from the old SSM project with weight mapping + consistency
check). Original head unchanged; per-K OA vs full paired diff.

Reset schedule: K in {1,4,8,16,32,64,full}; plus half-segment offset for K=16,32. K>=N with
zero offset must equal full (checked). K=1 cuts all history (checked structurally).

Usage:
  python memory_diagnosis.py --cache-only --group A --task rml --device cuda:0
  python memory_diagnosis.py --readout-only --group A
  python memory_diagnosis.py --group B --task rml --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_r1 import (load_s1plain_downstream, lr_readout, R1, R4, R2,
                       SEEDS, task_cfg, task_splits)
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm, parallel_scan, serial_scan
from ssm_exp.utils import seed_all
from round2_overnight.models import build_encoder_r2

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEAT = R1 / 'runs' / 'memory_features'
BATCH = 128
K_LIST = (1, 4, 8, 16, 32, 64)
OFFSETS = {16: 8, 32: 16}
A_INITS = ('random', 'mae', 'denoise')
B_INITS = ('scratch', 'mae', 'denoise')


def k_key(K, offset=0):
    if K is None:
        return 'full'
    if offset:
        return f'K{K}_off{offset}'
    return f'K{K}'


def reset_count(N, K, offset=0):
    if K is None:
        return 0
    return sum(1 for n in range(N) if (n + offset) % K == 0)


# ---- segment-reset scan -------------------------------------------------------------------

def segment_scan_aligned(a, b, K):
    """Zero-init recurrence reset at positions 0, K, 2K, ... (vectorized per-segment parallel
    scan). a, b: [B, N, S] complex."""
    B, N, S = a.shape
    n_seg = (N + K - 1) // K
    pad = n_seg * K - N
    if pad:
        a = F.pad(a, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, pad))
    a2 = a.reshape(B * n_seg, K, S)
    b2 = b.reshape(B * n_seg, K, S)
    h = parallel_scan(a2, b2).reshape(B, n_seg * K, S)[:, :N]
    return h


def segment_scan(a, b, K, offset=0):
    """Reset at positions n where (n + offset) % K == 0."""
    B, N, S = a.shape
    if K is None or (K >= N and offset == 0):
        return parallel_scan(a, b)
    r = (-offset) % K
    if r == 0:
        return segment_scan_aligned(a, b, K)
    h_pre = parallel_scan(a[:, :r], b[:, :r])
    h_rest = segment_scan_aligned(a[:, r:], b[:, r:], K)
    return torch.cat((h_pre, h_rest), 1)


def segment_scan_serial(a, b, K, offset=0):
    """Reference: O(N) Python loop with the same reset semantics (correctness check only)."""
    B, N, S = a.shape
    h = torch.zeros_like(b[:, 0])
    out = []
    for n in range(N):
        if (n + offset) % K == 0:
            h = torch.zeros_like(h)
        h = a[:, n] * h + b[:, n]
        out.append(h)
    return torch.stack(out, 1)


# ---- S1plain forward with segment reset ---------------------------------------------------

def _ssm_terms(ssm, u, p):
    v = F.linear(u.to(ssm.B.dtype), ssm.B)
    base_delta = ssm.raw_delta.view(1, 1, -1)
    aux = torch.zeros_like(p)   # S1plain: zero physical slots
    d_raw, g_raw = ssm.selector(torch.cat((u, aux), -1)).chunk(2, -1)
    delta = F.softplus(base_delta + d_raw)
    g = torch.sigmoid(g_raw)
    rho = torch.exp(-F.softplus(ssm.raw_alpha) * delta)
    a = torch.polar(rho.expand_as(v.real), ssm.theta.expand_as(v.real))
    return v, a, g, rho


def forward_s1(enc, x, K=None, offset=0):
    """S1plain forward with the recurrence state zeroed every K tokens (K=None -> full)."""
    u = enc.front(x)
    p = u.new_zeros(u.shape[0], u.shape[1], 3)
    for block in enc.blocks:
        ssm = block.mix
        xn = block.norm1(u)
        v, a, g, rho = _ssm_terms(ssm, xn, p)
        h = parallel_scan(a, g * v) if K is None else segment_scan(a, g * v, K, offset)
        y = ssm.gain * F.linear(h, ssm.C).real + ssm.direct * xn
        u = u + block.drop(y)
        u = u + block.drop(block.ffn(block.norm2(u)))
    return enc.norm(u)


def feats_s1(enc, ds, batch, device, K=None, offset=0):
    enc.eval()
    feats, ys = [], []
    loader = make_loader(ds, batch, 0)
    with torch.no_grad():
        for x, y, z, i in loader:
            x = max_abs_norm(x.to(device))
            feats.append(forward_s1(enc, x, K, offset).mean(1).cpu())
            ys.append(y)
    return torch.cat(feats), torch.cat(ys)


# ---- consistency checks -------------------------------------------------------------------

def consistency_checks(enc, device, N=128):
    """Verify the segment implementation against (a) the original forward, (b) the serial
    reference, (c) K>=N == full, (d) K=1 cuts history. Returns a report dict."""
    torch.manual_seed(0)
    x = torch.randn(8, 2, N, device=device)
    x = max_abs_norm(x)
    rep = {}
    with torch.no_grad():
        orig = enc.forward_tokens(x)
        mine_full = forward_s1(enc, x, K=None)
        rep['K_none_matches_original_maxdiff'] = float((orig - mine_full).abs().max().item())
    # segment impl vs serial reference on random recurrence terms (layer-0 terms)
    with torch.no_grad():
        u = enc.front(x)
        p = u.new_zeros(u.shape[0], u.shape[1], 3)
        v, a, g, _ = _ssm_terms(enc.blocks[0].mix, enc.blocks[0].norm1(u), p)
        b = g * v
        for K in (4, 16, 64):
            seg = segment_scan(a, b, K)
            ser = segment_scan_serial(a, b, K)
            rep[f'segment_vs_serial_K{K}_maxdiff'] = float((seg - ser).abs().max().item())
        for K, off in OFFSETS.items():
            seg = segment_scan(a, b, K, offset=off)
            ser = segment_scan_serial(a, b, K, offset=off)
            rep[f'segment_vs_serial_K{K}_off{off}_maxdiff'] = float((seg - ser).abs().max().item())
        # K>=N == full (zero offset)
        full = parallel_scan(a, b)
        rep['K_eq_N_equals_full_maxdiff'] = float((segment_scan(a, b, N) - full).abs().max().item())
        # K=1 cuts history: equals b (a never applied across steps)
        k1 = segment_scan(a, b, 1)
        rep['K1_equals_b_maxdiff'] = float((k1 - b).abs().max().item())
    return rep


# ---- A group ------------------------------------------------------------------------------

def a_ckpt(init, seed):
    if init == 'random':
        return R2 / 'runs' / 'random_ckpt' / f'S1plain_seed{seed}.pt'
    return R2 / 'runs' / 'pretrain' / f'S1plain_{init}_seed{seed}' / 'encoder_final.pt'


def load_a_encoder(init, seed, cfg, device):
    ck = torch.load(a_ckpt(init, seed), map_location='cpu', weights_only=False)
    s = (ck['config'].get('train', {}) or {}).get('seed') or ck['provenance'].get('seed', 0)
    seed_all(s)
    enc = build_encoder_r2(cfg, 'S1plain', s).to(device)
    enc.load_state_dict(ck['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def a_feat_path(init, task, seed):
    return FEAT / f'A_{init}__{task}_seed{seed}.pt'


def a_cache(init, task, seed, device):
    out = a_feat_path(init, task, seed)
    if out.exists():
        return out, 'skip'
    cfg = task_cfg(task)
    enc = load_a_encoder(init, seed, cfg, device)
    manifest, _ = task_splits(task)
    m = read_manifest(str(manifest))
    payload = {}
    # section 4 eval splits: RML val + radar dev (=radar 'val'); test not used here
    for split in ('train', 'val'):
        ds = IQDataset(m, split)
        payload[split] = {}
        for K, off in [(None, 0)] + [(k, 0) for k in K_LIST] + [(k, OFFSETS[k]) for k in OFFSETS]:
            f, y = feats_s1(enc, ds, BATCH, device, K, off)
            payload[split][k_key(K, off)] = {'feats': f, 'y': y}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out, 'ok'


def a_readout(init, task, seed):
    payload = torch.load(a_feat_path(init, task, seed), map_location='cpu', weights_only=False)
    Xtr_full = payload['train']['full']['feats'].numpy().astype(np.float32)
    ytr_full = payload['train']['full']['y'].numpy().astype(np.int64)
    out = {'fixed': {}, 'refit': {}}
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    # (1) fixed: fit once on full-state train
    scaler = StandardScaler().fit(Xtr_full)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs').fit(scaler.transform(Xtr_full), ytr_full)
    for split in ('val', 'test'):
        if split not in payload:
            continue
        for kk, entry in payload[split].items():
            Xva = entry['feats'].numpy().astype(np.float32)
            yva = entry['y'].numpy().astype(np.int64)
            pred = clf.predict(scaler.transform(Xva))
            oa = float((pred == yva).mean())
            out['fixed'][f'{split}:{kk}'] = oa
    # (2) refit: per-K scaler + LR on train
    for kk in payload['train']:
        Xtr = payload['train'][kk]['feats'].numpy().astype(np.float32)
        ytr = payload['train'][kk]['y'].numpy().astype(np.int64)
        s2 = StandardScaler().fit(Xtr)
        c2 = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs').fit(s2.transform(Xtr), ytr)
        for split in ('val', 'test'):
            if split not in payload:
                continue
            Xva = payload[split][kk]['feats'].numpy().astype(np.float32)
            yva = payload[split][kk]['y'].numpy().astype(np.int64)
            pred = c2.predict(s2.transform(Xva))
            oa = float((pred == yva).mean())
            out['refit'][f'{split}:{kk}'] = oa
    return out


# ---- B group ------------------------------------------------------------------------------

def b_ckpt(task, init, seed):
    if task == 'rml' and init == 'scratch':
        return Path('/home/user_wzk/wzk/projects/SSM/SSM/runs/clean') / f'S1_seed{seed}' / 'best.pt', True
    return R4 / 'runs' / task / init / f'S1plain_seed{seed}' / 'best.pt', False


def b_eval(model, loader, device, K, offset=0):
    model.eval()
    ys, ps = [], []
    for x, y, z, i in loader:
        x = max_abs_norm(x.to(device))
        with torch.no_grad():
            toks = forward_s1(model.encoder, x, K, offset)
            logits = model.head(toks.mean(1))
        ys.extend(y.tolist())
        ps.extend(logits.argmax(-1).cpu().tolist())
    ys = np.asarray(ys); ps = np.asarray(ps)
    return float((ps == ys).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['A', 'B'], required=True)
    ap.add_argument('--cache-only', action='store_true')
    ap.add_argument('--readout-only', action='store_true')
    ap.add_argument('--tasks', nargs='+', default=['rml', 'radar'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    FEAT.mkdir(parents=True, exist_ok=True)

    consistency = {}
    results = {}
    for task in a.tasks:
        cfg = task_cfg(task)
        manifest, splits = task_splits(task)
        m = read_manifest(str(manifest))
        val_split = 'val'   # radar 'val' == dev; section 4 evals on RML val + radar dev
        if a.group == 'A':
            inits = A_INITS
            for init in inits:
                for seed in a.seeds:
                    if not a.readout_only:
                        out, st = a_cache(init, task, seed, device)
                        print(f'[A cache] {init} {task} {seed}: {st}', flush=True)
                    # one consistency report per encoder (first seed only)
                    if seed == a.seeds[0]:
                        enc = load_a_encoder(init, seed, cfg, device)
                        N = cfg['input_length'] // cfg['patch']
                        consistency[f'A_{init}'] = consistency_checks(enc, device, N)
                    if not a.cache_only:
                        r = a_readout(init, task, seed)
                        results[f'A_{init}__{task}_seed{seed}'] = r
        else:
            inits = B_INITS
            for init in inits:
                for seed in a.seeds:
                    ckpt, old = b_ckpt(task, init, seed)
                    if not ckpt.exists():
                        results[f'B_{init}__{task}_seed{seed}'] = {'missing': str(ckpt)}
                        print(f'[B skip] {ckpt} missing', flush=True)
                        continue
                    model, cfg_l, ck, rep = load_s1plain_downstream(ckpt, device, old_classifier=old)
                    if seed == a.seeds[0]:
                        consistency[f'B_{init}__{task}'] = {'mapping': rep,
                            'checks': consistency_checks(model.encoder, device,
                                                         cfg_l['input_length'] // cfg_l['patch'])}
                    if a.readout_only:
                        continue
                    ds = IQDataset(m, val_split)
                    loader = make_loader(ds, BATCH, 0)
                    N = cfg_l['input_length'] // cfg_l['patch']
                    per_k = {}
                    base = None
                    for K, off in [(None, 0)] + [(k, 0) for k in K_LIST] + [(k, OFFSETS[k]) for k in OFFSETS]:
                        oa = b_eval(model, loader, device, K, off)
                        per_k[k_key(K, off)] = oa
                        if K is None:
                            base = oa
                    per_k = {kk: {'oa': v, 'delta_vs_full': v - base} for kk, v in per_k.items()}
                    per_k['n_tokens'] = N
                    per_k['reset_counts'] = {k_key(K, off): reset_count(N, K, off)
                                             for K, off in [(None, 0)] + [(k, 0) for k in K_LIST]
                                             + [(k, OFFSETS[k]) for k in OFFSETS]}
                    results[f'B_{init}__{task}_seed{seed}'] = per_k
                    print(json.dumps({f'B_{init}__{task}_seed{seed}': per_k}), flush=True)

    (R1 / 'reports').mkdir(parents=True, exist_ok=True)
    outjson = R1 / 'reports' / f'memory_{a.group}_raw.json'
    outjson.write_text(json.dumps({'results': results, 'consistency': consistency}, indent=2))
    _write_csv(a.group, results)
    print(f'[done] {outjson}')


def _write_csv(group, results):
    import csv
    rows = []
    for key, r in sorted(results.items()):
        if r.get('missing'):
            rows.append([key, '', '', 'missing'])
            continue
        if group == 'A':
            for mode in ('fixed', 'refit'):
                for kk, oa in sorted(r[mode].items()):
                    split, kname = kk.split(':', 1)
                    rows.append([key, mode, split, kname, f'{oa:.6f}'])
        else:
            for kk, d in sorted(r.items()):
                if kk in ('n_tokens', 'reset_counts'):
                    continue
                rows.append([key, kk, f"{d['oa']:.6f}", f"{d['delta_vs_full']:+.6f}"])
    name = {'A': 'memory_pretrain', 'B': 'memory_finetuned_original_head'}[group]
    with (R1 / 'reports' / f'{name}.csv').open('w', newline='') as f:
        w = csv.writer(f)
        if group == 'A':
            w.writerow(['key', 'readout', 'split', 'K', 'oa'])
        else:
            w.writerow(['key', 'K', 'oa', 'delta_vs_full'])
        w.writerows(rows)


if __name__ == '__main__':
    main()
