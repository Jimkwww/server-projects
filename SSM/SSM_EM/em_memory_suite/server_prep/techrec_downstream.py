"""Cross-domain transfer: main-matrix pretrained checkpoints -> TECHRec technology task.

Uses this package's own trainer (emmem.train.run) with init=<pretrained best.pt>, so the
protocol matches the main matrix (frozen probe / full finetune, 50ep, linear head). TECHRec
is a held-out cross-domain task (comm IQ pretrain -> real LTE/WiFi/DVB-T captures).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run
from emmem.evaluate import evaluate_checkpoint

MANIFEST = ROOT / 'data/techrec/manifest.json'
TASK = 'technology'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)  # pretrain best.pt
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='17,42,2026')
    ap.add_argument('--modes', default='probe,finetune')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()

    ck = json.loads((Path(a.checkpoint).parent / 'config.json').read_text())
    model_cfg = ck['model']
    for seed in [int(s) for s in a.seeds.split(',')]:
        for mode in a.modes.split(','):
            cfg = dict(stage=mode, seed=seed, data=[str(MANIFEST)], model=model_cfg,
                       epochs=a.epochs, batch_size=128, lr=3e-4, device=a.device,
                       require_cuda=True, workers=4, mask_ratio=0.5, aux_weight=ck.get('aux_weight', 1.0),
                       equiv_weight=0.0, history=ck.get('history', 8), shuffle_targets=False,
                       paired_augmentation=False, fraction=1.0, cpu_threads=4, grad_accum=1,
                       task=TASK, init=str(Path(a.checkpoint).resolve()))
            out = Path(a.out) / f's{seed}_{mode}'
            run(cfg, out, resume=False)
            result = evaluate_checkpoint(out / 'best.pt', str(MANIFEST), TASK,
                                         out / 'test_eval', device=a.device)
            print(json.dumps({'seed': seed, 'mode': mode, 'oa': result['oa'],
                              'macro_f1': result['macro_f1']}))


if __name__ == '__main__':
    main()
