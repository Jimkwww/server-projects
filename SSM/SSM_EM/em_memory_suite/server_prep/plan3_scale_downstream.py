"""Phase-3 (H9): AMC downstream (probe/finetune) for the scale-study checkpoints.
27 checkpoints x (probe,finetune) = 54 runs, sharded across 3 GPUs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run
from emmem.evaluate import evaluate_checkpoint

SCALES = ['small', 'medium', 'large']
MODELS = ['structured', 'generic_ssm', 'generic_transformer']
SEEDS = [17, 42, 2026]
MANIFEST = str(ROOT / 'data/rml2018_clsdev/manifest.json')
TASK = 'modulation'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)   # results/plan3_scale_downstream
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()

    jobs = [(sc, m, s) for sc in SCALES for m in MODELS for s in SEEDS]
    my = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]

    for scale, m, s in my:
        ckpt = str(ROOT / f'results/plan3_scale/{scale}_{m}_s{s}/best.pt')
        if not Path(ckpt).exists():
            print(f'[skip] {scale}_{m}_s{s}: no checkpoint')
            continue
        pre_cfg = json.loads((Path(ckpt).parent / 'config.json').read_text())
        model_cfg = pre_cfg['model']
        for mode in ['probe', 'finetune']:
            out = Path(a.out) / f'{scale}_{m}_s{s}_{mode}'
            if (out / 'DONE.json').exists():
                print(f'[skip] {scale}_{m}_s{s}_{mode}')
                continue
            cfg = dict(stage=mode, seed=s, data=[MANIFEST], model=model_cfg,
                       epochs=a.epochs, batch_size=128, lr=3e-4, device=a.gpu,
                       require_cuda=True, workers=4, mask_ratio=0.5,
                       aux_weight=pre_cfg.get('aux_weight', 1.0), equiv_weight=0.0,
                       history=pre_cfg.get('history', 8), shuffle_targets=False,
                       paired_augmentation=False, fraction=1.0, cpu_threads=4, grad_accum=1,
                       task=TASK, init=ckpt)
            run(cfg, out, resume=False)
            r = evaluate_checkpoint(out / 'best.pt', MANIFEST, TASK, out / 'test_eval', device=a.gpu)
            print(f'[done] {scale}_{m}_s{s}_{mode}: oa={r.get("oa")} macro_f1={r.get("macro_f1")}', flush=True)
    print(f'[shard {a.shard}] ALL DONE')


if __name__ == '__main__':
    main()
