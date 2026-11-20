"""Interim cross-plan summary: aggregate completed eval metrics from the 3 GPU plans into
one table + figure. Only clean test rows (identity perturbation, normal mode, best.pt init).
"""
import json
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GPUS = ['main3_cuda0', 'main3_cuda1', 'main3_cuda2']
TASKS = {'modulation': 'AMC', 'radar_waveform': 'radar'}


def collect():
    rows = []
    for gpu in GPUS:
        for f in glob.glob(f'results/{gpu}/plan/eval/*/metrics.json'):
            d = json.loads(open(f).read())
            if d.get('perturbation') != 'identity' or d.get('mode') != 'normal' or d.get('drop'):
                continue
            if d.get('split') != 'test':
                continue
            name = Path(f).parent.name  # e.g. pre_structured_s17_rml2018_modulation_probe_f1.0_test
            # parse variant / seed / task / stage / fraction
            parts = name.split('_')
            # variant = parts[1] (may contain underscores for some names)
            ck = d['checkpoint']
            cfg = json.loads(open(Path(ck).parent / 'config.json').read())
            rows.append({
                'variant': cfg['model'].get('__variant__', name),
                'seed': cfg['seed'],
                'task': d['task'],
                'stage': cfg['stage'],
                'fraction': cfg.get('fraction', 1.0),
                'init': Path(cfg.get('init', 'scratch')).name,
                'oa': d['metrics'].get('oa'),
                'macro_f1': d['metrics'].get('macro_f1'),
            })
    return rows


def variant_of(name):
    # pre_<variant>_s<seed>_<task>_<stage>_f<frac>_test
    import re
    m = re.match(r'pre_(.+?)_s\d+_', name)
    return m.group(1) if m else name


def main():
    rows = []
    for gpu in GPUS:
        for f in glob.glob(f'results/{gpu}/plan/eval/*/metrics.json'):
            d = json.loads(open(f).read())
            if d.get('perturbation') != 'identity' or d.get('mode') != 'normal' or d.get('drop') or d.get('split') != 'test':
                continue
            name = Path(f).parent.name
            ck = d['checkpoint']
            cfg = json.loads(open(Path(ck).parent / 'config.json').read())
            init = Path(cfg.get('init', 'scratch')).name
            if init not in ('best.pt', 'scratch'):
                continue
            rows.append({
                'variant': variant_of(name),
                'seed': cfg['seed'],
                'task': d['task'],
                'stage': cfg['stage'],
                'fraction': cfg.get('fraction', 1.0),
                'init': init,
                'oa': d['metrics'].get('oa'),
                'macro_f1': d['metrics'].get('macro_f1'),
            })

    # group by (variant, task, stage) for fraction=1.0 best.pt
    hdr = '{:<24s} {:<8s} {:<10s} {:>3s} {:>18s} {:>10s}'.format('variant', 'task', 'stage', 'n', 'OA mean±std', 'macroF1')
    print(hdr)
    for variant in sorted(set(r['variant'] for r in rows)):
        for task, tag in TASKS.items():
            for stage in ['probe', 'finetune']:
                sub = [r for r in rows if r['variant'] == variant and r['task'] == task
                       and r['stage'] == stage and r['fraction'] == 1.0 and r['init'] == 'best.pt']
                if not sub:
                    continue
                oas = [r['oa'] for r in sub if r['oa'] is not None]
                f1s = [r['macro_f1'] for r in sub if r['macro_f1'] is not None]
                if oas:
                    oa = np.array(oas)
                    f1 = np.array(f1s) if f1s else None
                    line = '{:<24s} {:<8s} {:<10s} {:>3d} {:>18s}'.format(
                        variant, tag, stage, len(oas), f'{oa.mean():.4f}±{oa.std(ddof=1):.4f}')
                    if f1 is not None:
                        line += f' {f1.mean():>9.4f}'
                    print(line)


if __name__ == '__main__':
    main()
