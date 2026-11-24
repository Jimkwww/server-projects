"""Phase-2 experiment 2: pretrain key models on the EXPANDED 200k corpus (F1 + extra 100k),
then we run downstream on all tasks (AMC/radar/TECHRec/HisarMod) afterwards.
Each --gpu shard pretrains its assigned (model, seed) pairs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run

MODEL_CFG_SRC = {
    'structured': 'results/main3_cuda0/plan/runs/pre_structured_s17/config.json',
    'generic_ssm': 'results/main3_cuda0/plan/runs/pre_generic_ssm_s17/config.json',
    'generic_transformer': 'results/main3_cuda0/plan/runs/pre_generic_transformer_s17/config.json',
    'generic_ssm_recon': 'results/main3_cuda2/plan/runs/pre_generic_ssm_recon_s17/config.json',
}
F1 = str(ROOT / 'data/rml2018_f1/manifest.json')
EXTRA = str(ROOT / 'data/rml2018_f1_extra/manifest.json')
SEEDS = [17, 42, 2026]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)  # results/plan2_pretrain
    ap.add_argument('--models', default='structured,generic_ssm,generic_transformer,generic_ssm_recon')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    a = ap.parse_args()

    models = a.models.split(',')
    pairs = [(m, s) for m in models for s in SEEDS]
    my = [p for i, p in enumerate(pairs) if i % a.nshards == a.shard]

    for m, s in my:
        src = json.loads((ROOT / MODEL_CFG_SRC[m]).read_text())
        out = Path(a.out) / f'{m}_s{s}'
        if (out / 'DONE.json').exists():
            print(f'[skip] {m}_s{s}: done')
            continue
        cfg = dict(stage='pretrain', seed=s, data=[F1, EXTRA], model=src['model'],
                   epochs=a.epochs, batch_size=128, lr=3e-4, device=a.gpu,
                   require_cuda=True, workers=4, mask_ratio=0.5,
                   aux_weight=src.get('aux_weight', 1.0), equiv_weight=0.0,
                   history=src.get('history', 8), shuffle_targets=src.get('shuffle_targets', False),
                   paired_augmentation=src.get('paired_augmentation', False),
                   fraction=1.0, cpu_threads=4, grad_accum=1)
        run(cfg, out, resume=False)
        print(f'[done] {m}_s{s} pretrain (200k)', flush=True)
    print(f'[shard {a.shard}] ALL PRETRAIN DONE')


if __name__ == '__main__':
    main()
