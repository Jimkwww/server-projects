"""Phase-3 (H9): scale study — pretrain structured / generic_ssm / generic_transformer at
3 scales (small dim32/depth2, medium dim128/depth6, large dim256/depth12) on F1 100k,
so we can compare with the phase-1 dim64/depth3 baseline and check whether the structured
advantage holds across model sizes. Each --gpu shard runs its assigned (scale, model, seed).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run

MODEL_SRC = {
    'structured': 'results/main3_cuda0/plan/runs/pre_structured_s17/config.json',
    'generic_ssm': 'results/main3_cuda0/plan/runs/pre_generic_ssm_s17/config.json',
    'generic_transformer': 'results/main3_cuda0/plan/runs/pre_generic_transformer_s17/config.json',
}
SCALES = {'small': (32, 2), 'medium': (128, 6), 'large': (256, 12)}
MODELS = ['structured', 'generic_ssm', 'generic_transformer']
SEEDS = [17, 42, 2026]
F1 = str(ROOT / 'data/rml2018_f1/manifest.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)   # results/plan3_scale
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()

    jobs = [(sc, m, s) for sc in SCALES for m in MODELS for s in SEEDS]
    my = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]

    for scale, m, s in my:
        dim, depth = SCALES[scale]
        src = json.loads((ROOT / MODEL_SRC[m]).read_text())
        model_cfg = dict(src['model'])
        model_cfg['dim'] = dim
        model_cfg['states'] = dim // 2
        model_cfg['depth'] = depth
        out = Path(a.out) / f'{scale}_{m}_s{s}'
        if (out / 'DONE.json').exists():
            print(f'[skip] {scale}_{m}_s{s}')
            continue
        cfg = dict(stage='pretrain', seed=s, data=[F1], model=model_cfg,
                   epochs=a.epochs, batch_size=128, lr=3e-4, device=a.gpu,
                   require_cuda=True, workers=4, mask_ratio=0.5,
                   aux_weight=src.get('aux_weight', 1.0), equiv_weight=0.0,
                   history=src.get('history', 8), shuffle_targets=False,
                   paired_augmentation=False, fraction=1.0, cpu_threads=4, grad_accum=1)
        run(cfg, out, resume=False)
        print(f'[done] {scale}_{m}_s{s} (dim{dim}/depth{depth})', flush=True)
    print(f'[shard {a.shard}] ALL DONE')


if __name__ == '__main__':
    main()
