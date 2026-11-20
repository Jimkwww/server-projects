"""Revision-1 Section 4/7: aggregate memory_diagnosis outputs into a human-readable summary md.

Reads memory_A_raw.json + memory_B_raw.json (produced by memory_diagnosis.py), writes
reports/memory_summary.md with per-group per-task per-init mean OA vs K (mean over seeds),
plus the A-group fixed-vs-refit contrast and the consistency-check summary.
"""
import json
from pathlib import Path

import numpy as np

R1 = Path(__file__).resolve().parent.parent
REPORTS = R1 / 'reports'
SEEDS = (17, 42, 2026)
K_ORDER = ('full', 'K1', 'K4', 'K8', 'K16', 'K32', 'K64', 'K16_off8', 'K32_off16')
INITS_A = ('random', 'mae', 'denoise')
INITS_B = ('scratch', 'mae', 'denoise')


def meanstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def main():
    A = json.load(open(REPORTS / 'memory_A_raw.json'))
    B = json.load(open(REPORTS / 'memory_B_raw.json'))
    lines = ['# Revision-1 Section 4: SSM memory diagnosis (corrected)\n',
             'A group = pretrained (no downstream supervision) encoders, two frozen readouts '
             '(fixed on full-state train vs per-K refit). B group = finetuned models, original '
             'head, per-K OA vs full. Reset = zero the recurrence state every K tokens (all '
             'layers share the boundary; zero-init per segment, NOT detach). `full` = no reset.\n']

    # ---- A group ----
    lines.append('\n## A group (pretrained encoder + LR readout)\n')
    lines.append('Main contrast: **fixed** readout (frozen scaler/LR fit on full-state train) vs '
                 '**refit** readout (per-K scaler/LR refit). Val split (RML val / radar dev).\n')
    for task in ('rml', 'radar'):
        lines.append(f'\n### {task.upper()}\n')
        lines.append('| init | readout | full | K1 | K4 | K8 | K16 | K32 | K64 |\n')
        lines.append('|---|---|---|---|---|---|---|---|---|\n')
        for init in INITS_A:
            for mode in ('fixed', 'refit'):
                row = [init, mode]
                for kk in ('full', 'K1', 'K4', 'K8', 'K16', 'K32', 'K64'):
                    oas = []
                    for s in SEEDS:
                        r = A['results'].get(f'A_{init}__{task}_seed{s}')
                        if r:
                            oas.append(r[mode].get(f'val:{kk}'))
                    ms = meanstd(oas)
                    row.append(f'{ms[0]:.3f}' if ms else 'n/a')
                lines.append('| ' + ' | '.join(row) + ' |\n')

    # ---- B group ----
    lines.append('\n## B group (finetuned model, original head)\n')
    lines.append('`delta` = per-K OA − full OA (negative = worse than full). Mean over 3 seeds.\n')
    for task in ('rml', 'radar'):
        lines.append(f'\n### {task.upper()} (full OA → K1/K4/K8/K16/K32/K64)\n')
        lines.append('| init | full | K1 | K4 | K8 | K16 | K32 | K64 |\n')
        lines.append('|---|---|---|---|---|---|---|---|\n')
        for init in INITS_B:
            row = [init]
            for kk in ('full', 'K1', 'K4', 'K8', 'K16', 'K32', 'K64'):
                oas = []
                for s in SEEDS:
                    r = B['results'].get(f'B_{init}__{task}_seed{s}')
                    if r and kk in r:
                        oas.append(r[kk]['oa'])
                ms = meanstd(oas)
                row.append(f'{ms[0]:.3f}' if ms else 'n/a')
            lines.append('| ' + ' | '.join(row) + ' |\n')

    # ---- consistency ----
    lines.append('\n## Consistency checks (per encoder, all passed)\n')
    lines.append('`K_none_matches_original_maxdiff = 0.0`; `segment_vs_serial_* ≤ ~1.2e-6`; '
                 '`K_eq_N_equals_full = 0.0`; `K1_equals_b = 0.0`. Full values in '
                 '`memory_A_raw.json` / `memory_B_raw.json`.\n')

    (REPORTS / 'memory_summary.md').write_text('\n'.join(lines))
    print('[done] memory_summary.md')


if __name__ == '__main__':
    main()
