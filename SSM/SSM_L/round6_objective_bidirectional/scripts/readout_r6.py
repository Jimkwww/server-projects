"""Round6 frozen linear readout (MAIN endpoint) for one (checkpoint, task) pair.

Protocol identical to round5 readout (round-3 corrected): freeze encoder -> mean-pool ->
StandardScaler(fit train) + L2 LogisticRegression(C=1, lbfgs, max_iter=2000). No eval-based
hyperparameter selection. RML main = val (test sealed); radar main = eval (dev reported too).

Differs from round5 readout_r5.py only in the encoder builder: build_encoder_r6 (G/Q/Q_bi).
Q_bi keeps its bidirectional A64 trunk in the frozen readout (encoder is used as-is).

Usage:
  python readout_r6.py --checkpoint <encoder_final.pt> --backbone G --corpus f1 --seed 17 \
      --ckpt-name epoch50 --task rml --device cuda:1 \
      --cache-dir round6_objective_bidirectional/runs/downstream/features \
      --pred-dir  round6_objective_bidirectional/runs/downstream/predictions \
      --out-json round6_objective_bidirectional/runs/downstream/readout/f1/G_denoise_seed17__epoch50__rml.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'round4_no_cfo_qstudy' / 'revision1' / 'scripts'))

from common_r1 import (features_for_encoder, task_cfg, task_splits)
from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.utils import seed_all
from round6_objective_bidirectional.scripts.models_r6 import build_encoder_r6, R6_BACKBONES


def sd_hash(state_dict):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _ckpt_seed(ck):
    t = ck.get('config', {}).get('train', {}).get('seed')
    if t is not None:
        return t
    p = ck.get('provenance', {})
    if isinstance(p, dict) and p.get('seed') is not None:
        return p['seed']
    return 0


def load_encoder(backbone, ckpt_path, device, cfg=None):
    """Build + load an encoder from an encoder_final.pt checkpoint (round6 builder)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = cfg or ck['config']['model']
    seed = _ckpt_seed(ck)
    seed_all(seed)
    enc = build_encoder_r6(cfg, backbone, seed).to(device)
    enc.load_state_dict(ck['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg, ck


def _ckpt_stem(ckpt_path):
    p = Path(ckpt_path)
    return f'{p.parent.parent.name}__{p.parent.name}__{p.name}'


def extract_task_features(backbone, ckpt_path, task, device, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{_ckpt_stem(ckpt_path)}__{task}.pt'
    if cache_path.exists():
        return torch.load(cache_path, map_location='cpu', weights_only=False), 'cached'

    cfg = task_cfg(task)
    enc, _, ck = load_encoder(backbone, ckpt_path, device, cfg=cfg)
    manifest, splits = task_splits(task)
    m = read_manifest(str(manifest))
    payload = {'encoder_sha256': sd_hash(ck['encoder']), 'ckpt': str(Path(ckpt_path).resolve()),
               'step': ck.get('step'), 'pass': ck.get('pass')}
    for split in splits:
        ds = IQDataset(m, split)
        feats, ys, zs, ids = features_for_encoder(enc, ds, 128, device)
        payload[split] = {'feats': feats, 'y': ys, 'snr': zs, 'idx': ids}
    torch.save(payload, cache_path)
    return payload, 'ok'


def run_readout(backbone, ckpt_path, task, device, cache_dir, pred_dir):
    payload, status = extract_task_features(backbone, ckpt_path, task, device, cache_dir)
    Xtr = payload['train']['feats'].numpy().astype(np.float32)
    ytr = payload['train']['y'].numpy().astype(np.int64)
    out = {'status': status, 'encoder_sha256': payload['encoder_sha256'],
           'ckpt': payload['ckpt'], 'step': payload.get('step'), 'pass': payload.get('pass')}
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs').fit(scaler.transform(Xtr), ytr)
    n_iter = int(np.asarray(clf.n_iter_).max())
    main_split = 'val' if task == 'rml' else 'test'
    for split in ('val', 'test'):
        if split not in payload:
            continue
        p = payload[split]
        X = scaler.transform(p['feats'].numpy().astype(np.float32))
        pred = clf.predict(X)
        yva = p['y'].numpy().astype(np.int64)
        oa = float(accuracy_score(yva, pred))
        f1 = float(f1_score(yva, pred, average='macro'))
        out[split] = {'oa': oa, 'macro_f1': f1, 'n_iter': n_iter, 'converged': n_iter < 2000}
        if split == main_split and pred_dir is not None:
            pd_ = Path(pred_dir); pd_.mkdir(parents=True, exist_ok=True)
            stem = f'{_ckpt_stem(ckpt_path)}__{task}'
            np.savez_compressed(pd_ / f'{stem}__{split}_pred.npz',
                                pred=pred.astype(np.int64), y=p['y'].numpy().astype(np.int64),
                                snr=p['snr'].numpy().astype(np.float32),
                                idx=p['idx'].numpy().astype(np.int64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', choices=R6_BACKBONES, required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--ckpt-name', required=True)
    ap.add_argument('--task', choices=['rml', 'radar'], required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--out-json', required=True)
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    r = run_readout(a.backbone, a.checkpoint, a.task, device, a.cache_dir, a.pred_dir)
    r.update({'corpus': a.corpus, 'seed': a.seed, 'ckpt_name': a.ckpt_name,
              'backbone': a.backbone, 'task': a.task})
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(r, indent=2, default=float))
    main = 'val' if a.task == 'rml' else 'test'
    print(json.dumps({'corpus': a.corpus, 'backbone': a.backbone, 'seed': a.seed,
                      'ckpt': a.ckpt_name, 'task': a.task, 'main': main,
                      'oa': r.get(main, {}).get('oa'), 'status': r['status']}), flush=True)


if __name__ == '__main__':
    main()
