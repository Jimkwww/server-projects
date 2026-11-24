"""Phase-2: full downstream on ALL 4 tasks using the 200k-pretrained checkpoints.
4 models x 3 seeds x 4 tasks x (probe,finetune) = 96 runs, sharded across 3 GPUs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emmem.train import run
from emmem.evaluate import evaluate_checkpoint

MODELS = ['structured', 'generic_ssm', 'generic_transformer', 'generic_ssm_recon']
SEEDS = [17, 42, 2026]
TASKS = [
    ('amc', 'data/rml2018_clsdev/manifest.json', 'modulation'),
    ('radar', 'data/radchar_tiny/manifest.json', 'radar_waveform'),
    ('techrec', 'data/techrec/manifest.json', 'technology'),
    ('hisarmod', 'data/hisarmod/manifest.json', 'modulation'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out', required=True)   # results/plan2_full
    ap.add_argument('--modes', default='probe,finetune')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=50)
    a = ap.parse_args()

    jobs = [(m, s, tag, man, task) for m in MODELS for s in SEEDS for tag, man, task in TASKS]
    my = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]

    for m, s, tag, man, task in my:
        ckpt = str(ROOT / f'results/plan2_pretrain/{m}_s{s}/best.pt')
        if not Path(ckpt).exists():
            print(f'[skip] {m}_s{s}: no 200k checkpoint')
            continue
        pre_cfg = json.loads((Path(ckpt).parent / 'config.json').read_text())
        model_cfg = pre_cfg['model']
        for mode in a.modes.split(','):
            out = Path(a.out) / f'{tag}/{m}_s{s}_{mode}'
            if (out / 'DONE.json').exists():
                print(f'[skip] {tag}/{m}_s{s}_{mode}')
                continue
            cfg = dict(stage=mode, seed=s, data=[str(ROOT / man)], model=model_cfg,
                       epochs=a.epochs, batch_size=128, lr=3e-4, device=a.gpu,
                       require_cuda=True, workers=4, mask_ratio=0.5,
                       aux_weight=pre_cfg.get('aux_weight', 1.0), equiv_weight=0.0,
                       history=pre_cfg.get('history', 8), shuffle_targets=False,
                       paired_augmentation=False, fraction=1.0, cpu_threads=4, grad_accum=1,
                       task=task, init=ckpt)
            run(cfg, out, resume=False)
            r = evaluate_checkpoint(out / 'best.pt', str(ROOT / man), task, out / 'test_eval', device=a.gpu)
            print(f'[done] {tag}/{m}_s{s}_{mode}: oa={r.get("oa")} macro_f1={r.get("macro_f1")}', flush=True)
    print(f'[shard {a.shard}] ALL DONE')


if __name__ == '__main__':
    main()
