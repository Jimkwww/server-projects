"""Phase-2 experiment 1: run a set of (model, seed) downstream (probe/finetune) on a new task
using phase-1 pretrained checkpoints. Each --gpu shard runs its assigned (model, seed) pairs.
Uses the package's own trainer/evaluator so protocol matches the main matrix.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run, dump
from emmem.evaluate import evaluate_checkpoint

CKPT = {
    'structured': 'results/main3_cuda0/plan/runs/pre_structured_s{seed}/best.pt',
    'generic_ssm': 'results/main3_cuda0/plan/runs/pre_generic_ssm_s{seed}/best.pt',
    'generic_transformer': 'results/main3_cuda0/plan/runs/pre_generic_transformer_s{seed}/best.pt',
    'generic_ssm_recon': 'results/main3_cuda2/plan/runs/pre_generic_ssm_recon_s{seed}/best.pt',
}
MODELS = list(CKPT)
SEEDS = [17, 42, 2026]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)          # cuda:0/1/2
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--out', required=True)          # e.g. results/plan2_techrec
    ap.add_argument('--models', default='structured,generic_ssm,generic_transformer,generic_ssm_recon')
    ap.add_argument('--modes', default='probe,finetune')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--shard', type=int, default=0)  # which slice of (model,seed)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()

    device = a.gpu
    models = a.models.split(',')
    pairs = [(m, s) for m in models for s in SEEDS]
    my = [p for i, p in enumerate(pairs) if i % a.nshards == a.shard]

    for m, s in my:
        ckpt = str(ROOT / CKPT[m].format(seed=s))
        if not Path(ckpt).exists():
            print(f'[skip] {m}_s{s}: no checkpoint')
            continue
        pre_cfg = json.loads((Path(ckpt).parent / 'config.json').read_text())
        model_cfg = pre_cfg['model']
        for mode in a.modes.split(','):
            out = Path(a.out) / f'{m}_s{s}_{mode}'
            if (out / 'DONE.json').exists():
                print(f'[skip] {m}_s{s}_{mode}: done')
                continue
            cfg = dict(stage=mode, seed=s, data=[a.manifest], model=model_cfg,
                       epochs=a.epochs, batch_size=128, lr=3e-4, device=device,
                       require_cuda=True, workers=4, mask_ratio=0.5,
                       aux_weight=pre_cfg.get('aux_weight', 1.0), equiv_weight=0.0,
                       history=pre_cfg.get('history', 8), shuffle_targets=False,
                       paired_augmentation=False, fraction=1.0, cpu_threads=4, grad_accum=1,
                       task=a.task, init=ckpt)
            run(cfg, out, resume=False)
            r = evaluate_checkpoint(out / 'best.pt', a.manifest, a.task, out / 'test_eval', device=device)
            print(f'[done] {m}_s{s}_{mode}: oa={r.get("oa")} macro_f1={r.get("macro_f1")}', flush=True)
    print(f'[shard {a.shard}] ALL DONE')


if __name__ == '__main__':
    main()
