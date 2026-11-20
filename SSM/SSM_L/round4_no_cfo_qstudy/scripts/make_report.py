"""Round-4 report generator: aggregates P0/P1/P2 results into tables + conclusion.md.

Reads best_val.json / results.json / p2_ssm_memory_*.json, computes per-seed + mean + sample
std + paired deltas, writes the deliverable tables and the acceptance checklist. Run after the
orchestrator finishes:
  python round4_no_cfo_qstudy/scripts/make_report.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
R4 = ROOT / 'round4_no_cfo_qstudy'
RUNS = R4 / 'runs'
REPORTS = R4 / 'reports'
SEEDS = (17, 42, 2026)

# Reused RML scratch val OA (old SSM project, no-CFO). source -> (seed -> oa).
REUSED_RML_SCRATCH = {
    'G': {17: 0.39326923076923076, 42: 0.3836538461538462, 2026: 0.3836538461538462,
          'src': 'SSM/runs/clean/G_seed*'},
    'S1plain': {17: 0.39006410256410257, 42: 0.37596153846153846, 2026: 0.3778846153846154,
                'src': 'SSM/runs/clean/S1_seed*'},
    'G_corr': {17: 0.37884615384615383, 42: 0.37115384615384617, 2026: 0.36955128205128207,
               'src': 'SSM/runs/round3/G_corr_seed*'},
}


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, s


def fmt(m, s):
    return f'{m:.4f} ± {s:.4f}'


def load_oa(run_dir):
    b = run_dir / 'best_val.json'
    if not b.exists():
        return None
    d = json.loads(b.read_text())
    return d['metrics']['overall']['oa']


def load_eval_test_oa(run_dir):
    b = run_dir / 'best_val.json'
    if not b.exists():
        return None
    d = json.loads(b.read_text())
    if 'eval_test' in d:
        return d['eval_test']['overall']['oa']
    return None


def collect():
    res = {}
    for task in ('rml', 'radar'):
        res[task] = {}
        for init in ('scratch', 'mae', 'denoise'):
            res[task][init] = {}
            for bb in ('G', 'Q', 'S1plain', 'G_Qlearn'):
                per_seed = {}
                for s in SEEDS:
                    if task == 'rml' and init == 'scratch' and bb in REUSED_RML_SCRATCH:
                        per_seed[s] = REUSED_RML_SCRATCH[bb][s]
                        continue
                    oa = load_oa(RUNS / task / init / f'{bb}_seed{s}')
                    per_seed[s] = oa
                res[task][init][bb] = per_seed
    # P1
    res['p1'] = {}
    for task in ('rml', 'radar'):
        res['p1'][task] = {}
        for setting in ('A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq'):
            per_seed = {}
            for s in SEEDS:
                per_seed[s] = load_oa(RUNS / task / setting / f'G_Qlearn_seed{s}')
            res['p1'][task][setting] = per_seed
        # D = P0 G_Qlearn MAE finetune
        res['p1'][task]['D_learnq_learnq'] = {s: load_oa(RUNS / task / 'mae' / f'G_Qlearn_seed{s}') for s in SEEDS}
        # G_corr reference: mae finetune + probe
        res['p1'][task]['G_corr_mae'] = {s: load_oa(RUNS / task / 'mae' / f'G_corr_seed{s}') for s in SEEDS}
        res['p1'][task]['G_corr_probe'] = {s: load_oa(RUNS / task / 'mae_probe' / f'G_corr_seed{s}') for s in SEEDS}
        # G and G_corr scratch references (RML reused from old SSM; radar trained fresh)
        if task == 'rml':
            res['p1'][task]['G_scratch'] = REUSED_RML_SCRATCH['G']
            res['p1'][task]['G_corr_scratch'] = REUSED_RML_SCRATCH['G_corr']
        else:
            res['p1'][task]['G_scratch'] = {s: load_oa(RUNS / task / 'scratch' / f'G_seed{s}') for s in SEEDS}
            res['p1'][task]['G_corr_scratch'] = {s: load_oa(RUNS / task / 'scratch' / f'G_corr_seed{s}') for s in SEEDS}
    return res


def table_main(res, task, fh):
    fh.write(f'## {task.upper()} val OA (no-CFO)\n\n')
    inits = ['scratch', 'mae', 'denoise']
    bbs = ['G', 'Q', 'S1plain', 'G_Qlearn']
    hdr = '| backbone | init | ' + ' | '.join(f'seed{s}' for s in SEEDS) + ' | mean ± sample-std |\n'
    fh.write(hdr)
    fh.write('|' + '---|' * (len(SEEDS) + 3) + '\n')
    for bb in bbs:
        for init in inits:
            vals = [res[task][init][bb][s] for s in SEEDS]
            if any(v is None for v in vals):
                continue
            m, sd = mean_std(vals)
            fh.write(f'| {bb} | {init} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {fmt(m, sd)} |\n')
    # G_corr reference (P1, same task)
    fh.write('\n### G_corr reference (P1)\n\n')
    for init in ('scratch', 'mae'):
        key = 'G_corr_scratch' if init == 'scratch' else 'G_corr_mae'
        vals = [res['p1'][task][key][s] for s in SEEDS]
        if any(v is None for v in vals):
            continue
        m, sd = mean_std(vals)
        fh.write(f'| G_corr | {init} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {fmt(m, sd)} |\n')
    fh.write('\n')


def table_radar_test(res, fh):
    fh.write('## RadChar test (=eval) OA (no-CFO, selected on val)\n\n')
    inits = ['scratch', 'mae', 'denoise']
    bbs = ['G', 'Q', 'S1plain', 'G_Qlearn', 'G_corr']
    fh.write('| backbone | init | ' + ' | '.join(f'seed{s}' for s in SEEDS) + ' | mean ± sample-std |\n')
    fh.write('|' + '---|' * (len(SEEDS) + 3) + '\n')
    for bb in bbs:
        for init in inits:
            vals = []
            for s in SEEDS:
                if bb == 'G_corr' and init == 'scratch':
                    vals.append(load_eval_test_oa(RUNS / 'radar' / 'scratch' / f'G_corr_seed{s}'))
                else:
                    vals.append(load_eval_test_oa(RUNS / 'radar' / init / f'{bb}_seed{s}'))
            if any(v is None for v in vals):
                continue
            m, sd = mean_std(vals)
            fh.write(f'| {bb} | {init} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {fmt(m, sd)} |\n')
    fh.write('\n')


def table_p1(res, fh):
    fh.write('## P1: Q learning-phase 2×2 (G_Qlearn, F0-MAE) val OA\n\n')
    for task in ('rml', 'radar'):
        fh.write(f'### {task.upper()}\n\n')
        settings = ['A_fixedq_fixedq', 'B_learnq_fixedq', 'C_fixedq_learnq', 'D_learnq_learnq']
        fh.write('| setting | ' + ' | '.join(f'seed{s}' for s in SEEDS) + ' | mean ± sample-std |\n')
        fh.write('|' + '---|' * (len(SEEDS) + 2) + '\n')
        for st in settings:
            vals = [res['p1'][task][st][s] for s in SEEDS]
            if any(v is None for v in vals):
                continue
            m, sd = mean_std(vals)
            fh.write(f'| {st} | ' + ' | '.join(f'{v:.4f}' for v in vals) + f' | {fmt(m, sd)} |\n')
        # paired diffs vs D (learn-learn)
        base = [res['p1'][task]['D_learnq_learnq'][s] for s in SEEDS]
        fh.write('\nPaired deltas vs D (learn-Q pretrain + learn-Q finetune):\n\n')
        for st in settings:
            if st == 'D_learnq_learnq':
                continue
            vals = [res['p1'][task][st][s] for s in SEEDS]
            if any(v is None for v in vals) or any(b is None for b in base):
                continue
            diffs = [v - b for v, b in zip(vals, base)]
            m, sd = mean_std(diffs)
            fh.write(f'- {st}: per-seed {["%.4f" % d for d in diffs]}, mean {m:+.4f} ± {sd:.4f}\n')
        # references
        fh.write('\nReferences:\n\n')
        for key in ('G_scratch', 'G_corr_scratch', 'G_corr_mae', 'G_corr_probe'):
            if key not in res['p1'][task]:
                continue
            vals = [res['p1'][task][key][s] for s in SEEDS]
            if any(v is None for v in vals):
                continue
            m, sd = mean_std(vals)
            fh.write(f'- {key}: {fmt(m, sd)}\n')
        fh.write('\n')


def table_baselines(fh):
    fh.write('## P0 appendix: radar simple baselines\n\n')
    r = RUNS / 'radar_baselines' / 'results.json'
    if not r.exists():
        fh.write('(not yet available)\n\n')
        return
    d = json.loads(r.read_text())
    fh.write('| features | dim | val(=dev) OA | test(=eval) OA |\n|---|---|---|---|\n')
    for kind in ('flattened_iq', 'physical'):
        k = d[kind]
        fh.write(f"| {kind} | {k['feature_dim']} | {k['val']['oa']:.4f} | {k['test']['oa']:.4f} |\n")
    fh.write('\n')


def table_p2(fh):
    fh.write('## P2: SSM effective-memory diagnosis (S1plain)\n\n')
    for task in ('rml', 'radar'):
        p = REPORTS / f'p2_ssm_memory_{task}.json'
        if not p.exists():
            fh.write(f'({task}: not yet available)\n\n')
            continue
        d = json.loads(p.read_text())
        fh.write(f'### {task.upper()} segment-reset val OA vs K\n\n')
        fh.write('| init_seed | K=none | K=64 | K=16 | K=4 | K=1 |\n|---|---|---|---|---|---|\n')
        for key, v in d.items():
            if key in ('task', 'inits', 'seeds'):
                continue
            if 'segment_oa' not in v:
                continue
            seg = v['segment_oa']
            fh.write(f"| {key} | {seg['K_none']:.4f} | {seg['K_64']:.4f} | {seg['K_16']:.4f} | "
                     f"{seg['K_4']:.4f} | {seg['K_1']:.4f} |\n")
        fh.write('\n')
        # state stats summary (first available init_seed)
        for key, v in d.items():
            if key in ('task', 'inits', 'seeds') or 'state_stats' not in v:
                continue
            fh.write(f'### {task.upper()} state stats ({key})\n\n')
            fh.write('| layer | rho mean | rho p50 | rho>0.999 | tau mean (tok) | tau p50 | gate mean | state rms |\n')
            fh.write('|---|---|---|---|---|---|---|---|\n')
            for L in v['state_stats']:
                fh.write(f"| {L['layer']} | {L['rho_mean']:.4f} | {L['rho_p50']:.4f} | "
                         f"{L['rho_frac_gt_0_999']:.3f} | {L['tau_mean_tokens']:.1f} | "
                         f"{L['tau_p50']:.1f} | {L['gate_mean']:.4f} | {L['state_rms']:.4f} |\n")
            fh.write('\n')
            break


def acceptance(fh):
    fh.write('## Acceptance checks\n\n')
    checks = []
    # 1: no CFO in any round4 config
    bad = []
    for cfg in R4.glob('configs/*.json'):
        c = json.loads(cfg.read_text())
        if c.get('train', {}).get('augment_cfo_max', 0.0) != 0.0:
            bad.append(cfg.name)
    checks.append(('no-CFO in all round4 configs', not bad, f'bad={bad}'))
    # 2: every trained run has DONE.json next to its best.pt
    missing = []
    for jd in list((RUNS / 'radar').rglob('best.pt')) + list((RUNS / 'rml').rglob('best.pt')):
        if not (jd.parent / 'DONE.json').exists():
            missing.append(str(jd.parent))
    checks.append(('every best.pt has DONE.json', not missing, f'missing={len(missing)}'))
    # 3: manifest exists
    mf = REPORTS / 'run_manifest.csv'
    checks.append(('run_manifest.csv exists', mf.exists(), str(mf)))
    # 4: no failed runs in manifest
    n_fail = 0
    if mf.exists():
        for line in mf.read_text().splitlines()[1:]:
            if 'failed' in line:
                n_fail += 1
    checks.append(('no failed runs in manifest', n_fail == 0, f'failed={n_fail}'))
    # 5: fixed-Q aux hash verified for A/B (only the Q filter frozen)
    fz = []
    for d in list((RUNS / 'rml').rglob('best_val.json')) + list((RUNS / 'radar').rglob('best_val.json')):
        if ('A_fixedq' in str(d) or 'B_learnq' in str(d)):
            bv = json.loads(d.read_text())
            if not bv.get('aux_frozen_verified'):
                fz.append(str(d))
    checks.append(('fixed-Q aux hash verified for A/B', not fz, f'bad={fz}'))
    # 6: every downstream run saved val_predictions.npz
    no_pred = []
    for jd in list((RUNS / 'radar').rglob('best.pt')) + list((RUNS / 'rml').rglob('best.pt')):
        if not (jd.parent / 'val_predictions.npz').exists():
            no_pred.append(str(jd.parent))
    checks.append(('all runs have val_predictions.npz', not no_pred, f'missing={len(no_pred)}'))
    # 7: radar runs have eval_test (eval-on-test-once) + eval_predictions
    no_eval = []
    for jd in (RUNS / 'radar').rglob('best.pt'):
        bv = jd.parent / 'best_val.json'
        if bv.exists() and ('eval_test' not in json.loads(bv.read_text())):
            no_eval.append(str(jd.parent))
        if not (jd.parent / 'eval_predictions.npz').exists():
            no_eval.append(str(jd.parent) + ' (no eval_predictions)')
    checks.append(('radar eval-on-test once + eval_predictions', not no_eval, f'bad={len(no_eval)}'))
    # 8: finetune runs record the pretrained checkpoint SHA256
    no_sha = []
    for d in list((RUNS / 'rml').rglob('provenance.json')) + list((RUNS / 'radar').rglob('provenance.json')):
        pv = json.loads(d.read_text())
        if pv.get('mode') == 'finetune' and not pv.get('pretrained_sha256'):
            no_sha.append(str(d))
    checks.append(('finetune provenance records pretrained_sha256', not no_sha, f'missing={len(no_sha)}'))
    # 9: radar input length is 512 points (5 classes), RML 1024 (24 classes)
    rc = json.loads((R4 / 'configs/downstream_nocfo_radar.json').read_text())
    mc = json.loads((R4 / 'configs/downstream_nocfo_rml.json').read_text())
    ok_len = rc['model'].get('input_length') == 512 and rc['model']['classes'] == 5 \
        and mc['model'].get('input_length') == 1024 and mc['model']['classes'] == 24
    checks.append(('radar 512pt/5cls, RML 1024pt/24cls', ok_len,
                   f"radar={rc['model'].get('input_length')}/{rc['model']['classes']} "
                   f"rml={mc['model'].get('input_length')}/{mc['model']['classes']}"))
    # 10: MAE mask is applied to raw input BEFORE the aux descriptor (static ordering check
    #     inside the forward_tokens method body).
    pt = (R4 / 'pretrain_r4.py').read_text()
    enc2 = (ROOT / 'round2_overnight' / 'models.py').read_text()
    gcorr = (R4 / 'models_r4.py').read_text()

    def body(src, marker, before, after):
        k = src.find(marker)
        if k == -1:
            return None
        seg = src[k:]
        return seg.find(before) != -1 and seg.find(after) != -1 and seg.find(before) < seg.find(after)

    ok_e2 = body(enc2, 'def forward_tokens', 'mask_patches', 'aux_descriptor')
    ok_gc = body(gcorr, 'def forward_tokens', 'mask_patches', 'physical_features')
    ok_mask = ('sample_mask' in pt and 'mask_ratio' in pt and ok_e2 and ok_gc)
    checks.append(('MAE mask applied before aux descriptor', ok_mask,
                   f'encoder2={ok_e2}, gcorr={ok_gc}'))
    for i, (name, ok, detail) in enumerate(checks, 1):
        fh.write(f'- [{"x" if ok else " "}] {name} — {detail}\n')
    fh.write('\n')


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    res = collect()
    (REPORTS / 'results_summary.json').write_text(json.dumps(res, indent=2))

    with (REPORTS / 'tables_main.md').open('w') as fh:
        fh.write('# Round-4 main tables\n\n')
        table_main(res, 'rml', fh)
        table_main(res, 'radar', fh)
        table_radar_test(res, fh)

    with (REPORTS / 'tables_p1.md').open('w') as fh:
        fh.write('# Round-4 P1 tables\n\n')
        table_p1(res, fh)

    with (REPORTS / 'tables_baselines.md').open('w') as fh:
        fh.write('# Round-4 radar baselines\n\n')
        table_baselines(fh)

    with (REPORTS / 'tables_p2.md').open('w') as fh:
        fh.write('# Round-4 P2 SSM memory\n\n')
        table_p2(fh)

    with (REPORTS / 'acceptance.md').open('w') as fh:
        fh.write('# Round-4 acceptance\n\n')
        acceptance(fh)

    print('[make_report] wrote tables_main.md tables_p1.md tables_baselines.md tables_p2.md '
          'acceptance.md results_summary.json')


if __name__ == '__main__':
    main()
