"""Round-4 revision-1 shared helpers: encoder loading (incl. G_corr), feature extraction,
unified LR readout, metrics, hashing, and S1plain downstream loading (incl. old-S1 mapping)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent   # SSM_L
R4 = ROOT / 'round4_no_cfo_qstudy'
R1 = R4 / 'revision1'
R2 = ROOT / 'round2_overnight'
sys.path.insert(0, str(ROOT))

from ssm_exp.data import IQDataset, read_manifest
from ssm_exp.engine import make_loader
from ssm_exp.model import max_abs_norm
from ssm_exp.utils import seed_all
from round4_no_cfo_qstudy.models_r4 import build_encoder_r4, aux_param_hash
from round2_overnight.models import build_encoder_r2, Encoder2
from pretrain.model import DownstreamModel

SEEDS = (17, 42, 2026)
RML_MANIFEST = ROOT / 'configs' / 'data.json'
RADAR_MANIFEST = ROOT / 'round2_overnight' / 'configs' / 'radchar.json'


def model_cfg():
    return json.loads((R4 / 'configs' / 'downstream_nocfo_rml.json').read_text())['model']


def _ckpt_seed(ck):
    t = ck.get('config', {}).get('train', {}).get('seed')
    if t is not None:
        return t
    p = ck.get('provenance', {})
    if isinstance(p, dict) and p.get('seed') is not None:
        return p['seed']
    return 0


def load_encoder(backbone, ckpt_path, device, cfg=None):
    """Build + load an encoder from an encoder_final.pt checkpoint. Handles all 5 backbones.

    cfg (optional) overrides the stored model config (e.g. radar input_length/classes); the
    encoder body is length-agnostic so the same state_dict loads either way.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = cfg or ck['config']['model']
    seed = _ckpt_seed(ck)
    seed_all(seed)
    enc = build_encoder_r4(cfg, backbone, seed).to(device)
    enc.load_state_dict(ck['encoder'])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg, ck


def features_for_encoder(enc, ds, batch, device):
    """Mean-pooled features [N,dim] + labels/snr/ids, max_abs_norm input."""
    enc.eval()
    feats, ys, zs, ids = [], [], [], []
    loader = make_loader(ds, batch, 0)
    with torch.no_grad():
        for x, y, z, i in loader:
            x = max_abs_norm(x.to(device))
            feats.append(enc.features(x).cpu())
            ys.append(y)
            zs.append(z)
            ids.append(i)
    return (torch.cat(feats), torch.cat(ys), torch.cat(zs), torch.cat(ids))


def sd_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def lr_readout(Xtr, ytr, Xva, yva, max_iter=2000):
    """Unified round-3 readout: StandardScaler(train) + L2 LR(C=1, lbfgs). Returns
    (va_oa, va_macro_f1, n_iter, converged). No eval-based hyperparameter selection."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr, dtype=np.int64)
    Xva = np.asarray(Xva, dtype=np.float32); yva = np.asarray(yva, dtype=np.int64)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=max_iter, solver='lbfgs')
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(scaler.transform(Xva))
    oa = float(accuracy_score(yva, pred))
    f1 = float(f1_score(yva, pred, average='macro'))
    n_iter = int(np.asarray(clf.n_iter_).max()) if hasattr(clf, 'n_iter_') else None
    converged = bool(int(np.asarray(clf.n_iter_).max()) < max_iter) if hasattr(clf, 'n_iter_') else None
    return oa, f1, n_iter, converged


def task_splits(task):
    """Return (manifest, [split names to extract]). Radar: train/dev/eval; RML: train/val (test sealed)."""
    if task == 'rml':
        return RML_MANIFEST, ('train', 'val')
    return RADAR_MANIFEST, ('train', 'val', 'test')   # radar 'val'=dev, 'test'=eval


def task_cfg(task):
    cfg = json.loads((R4 / 'configs' / f'downstream_nocfo_{task}.json').read_text())['model']
    return cfg


# ---- S1plain downstream loading for the B-group memory diagnosis -------------------------

def load_s1plain_downstream(ckpt_path, device, old_classifier=False):
    """Load a trained S1plain downstream model (encoder + original head) from either format.

    - round-4 downstream best.pt: DownstreamModel state_dict under 'model' (encoder.* + head.*).
    - old SSM project S1 Classifier best.pt (old_classifier=True): maps the Classifier
      state_dict (phys_* dropped, aux_mean/aux_std absent in the old checkpoint, head.* copied)
      into Encoder2(S1plain) + DownstreamModel head. Returns the model plus a mapping report
      so the caller can verify forward consistency against the original Classifier.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['config']['model']
    cfg = {k: v for k, v in cfg.items() if k != 'variant'}   # S1plain is variant-agnostic
    seed = ck['config']['train']['seed']
    seed_all(seed)
    classes = cfg['classes']
    enc = build_encoder_r2(cfg, 'S1plain', seed).to(device)
    report = {}
    if old_classifier:
        sd = ck['model']
        enc_sd = {k: v for k, v in sd.items()
                  if not k.startswith('head.') and not k.startswith('phys_')}
        missing = set(enc.state_dict().keys()) - set(enc_sd.keys())
        unexpected = set(enc_sd.keys()) - set(enc.state_dict().keys())
        enc.load_state_dict(enc_sd, strict=False)
        # the only expected missing keys are the unused aux buffers (S1plain never reads them)
        assert missing == {'aux_mean', 'aux_std'}, f'unexpected missing: {missing}'
        assert not unexpected, f'unexpected keys: {unexpected}'
        head = torch.nn.Linear(cfg['dim'], classes)
        head.load_state_dict({k.replace('head.', ''): v for k, v in sd.items()
                              if k.startswith('head.')})
        head = head.to(device)
        report = {'old_classifier': True, 'missing_keys': sorted(missing),
                  'dropped': ['phys_mean', 'phys_std', 'head.weight', 'head.bias']}
    else:
        model_sd = ck['model']
        enc_sd = {k.replace('encoder.', ''): v for k, v in model_sd.items()
                  if k.startswith('encoder.')}
        enc.load_state_dict(enc_sd, strict=True)
        head = torch.nn.Linear(cfg['dim'], classes)
        head.load_state_dict({k.replace('head.', ''): v for k, v in model_sd.items()
                              if k.startswith('head.')})
        head = head.to(device)
        report = {'old_classifier': False}
    model = DownstreamModel(enc, classes).to(device)
    model.head.load_state_dict(head.state_dict())
    model.eval()
    return model, cfg, ck, report
