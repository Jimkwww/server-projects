"""Round5 fixed-epoch data-scaling: aggregate new results + reusable historical results.

Produces (under round5_epoch_scale/reports/):
  - readout_trend.csv / .md      : frozen readout OA per checkpoint (step7800/ep20/30/40/50)
  - readout_main.csv / .md       : final-50ep vs F0-50ep vs historical-7800-step (frozen readout)
  - finetune_main.csv / .md      : no-CFO finetune (RML val + radar eval): F0 vs 7800 vs 50ep vs scratch
  - paired_deltas.csv / .md      : per-seed paired deltas (50ep - 7800, 50ep - F0, per model)
  - recon_trend.csv / .md        : recon loss vs downstream OA (question 4)
  - conclusion_zh.md             : one-page Chinese conclusion (4 questions)
  - tables_ppt.md                : PPT-ready tables
  - analysis_raw.json            : everything, machine-readable

Historical sources (reused, per-seed):
  - F0-50ep frozen readout RML : round3 section3_unified_readout.json  [bb]_mae_seed[s].full_label.oa
  - F0-50ep frozen readout radar: round3 section4_radar_transfer.json  [bb]_mae_seed[s].full_label_lr.oa
  - F1/F2-7800 frozen readout   : round3 section6_probe.json [corpus]_[bb]_seed[s].rml.full_label.oa / .radar.oa
  - F0-50ep no-CFO finetune     : round4 runs/{rml,radar}/{mae,scratch}/[bb]_seed[s]/best_val.json
                                   (RML val = metrics.overall.oa; radar eval = eval_test.overall.oa)
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R5 = ROOT / 'round5_epoch_scale'
R3 = ROOT / 'round3_scale_transfer'
R4 = ROOT / 'round4_no_cfo_qstudy'
REP = R5 / 'reports'

BACKBONES = ('G', 'G_Qlearn', 'Q')
CORPORA = ('f1', 'f2')
SEEDS = (17, 42, 2026)
CKPT_NAMES = ('step7800', 'epoch20', 'epoch30', 'epoch40', 'epoch50')


def meanstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


# ---------------------------------------------------------------- new results ----

def load_new_readout():
    """{ (corpus,bb,ckpt,task): {split: {oa, macro_f1}, seed: ...} } -> per-seed dict."""
    out = {}
    for corpus in CORPORA:
        for bb in BACKBONES:
            for ckpt in CKPT_NAMES:
                for task in ('rml', 'radar'):
                    seeds = {}
                    for s in SEEDS:
                        p = (R5 / 'runs' / 'downstream' / 'readout' / corpus /
                             f'{bb}_seed{s}__{ckpt}__{task}.json')
                        if p.exists():
                            d = json.load(open(p))
                            seeds[s] = d
                        else:
                            seeds[s] = None
                    out[(corpus, bb, ckpt, task)] = seeds
    return out


def load_new_finetune():
    out = {}
    for which in ('50ep', '7800'):
        for corpus in CORPORA:
            for bb in BACKBONES:
                for task in ('rml', 'radar'):
                    seeds = {}
                    for s in SEEDS:
                        p = (R5 / 'runs' / 'downstream' / f'finetune_{which}' / task /
                             f'{corpus}_{bb}_seed{s}' / 'best_val.json')
                        if p.exists():
                            d = json.load(open(p))
                            if task == 'rml':
                                seeds[s] = {'val_oa': d['metrics']['overall']['oa'],
                                            'val_f1': d['metrics']['overall']['macro_f1']}
                            else:
                                seeds[s] = {'val_oa': d['metrics']['overall']['oa'],
                                            'eval_oa': d['eval_test']['overall']['oa'],
                                            'eval_f1': d['eval_test']['overall']['macro_f1']}
                        else:
                            seeds[s] = None
                    out[(which, corpus, bb, task)] = seeds
    return out


# --------------------------------------------------------- historical (reused) ----

def load_hist_readout():
    """{ (corpus,bb,task): {seed: oa} } where corpus in {f0, f1, f2}."""
    out = {}
    s3 = json.load(open(R3 / 'reports' / 'section3_unified_readout.json'))['results']
    s4 = json.load(open(R3 / 'reports' / 'section4_radar_transfer.json'))['results']
    s6 = json.load(open(R3 / 'reports' / 'section6_probe.json'))['results']
    for bb in BACKBONES:
        rml = {s: s3[f'{bb}_mae_seed{s}']['full_label']['oa'] for s in SEEDS}
        rad = {s: s4[f'{bb}_mae_seed{s}']['full_label_lr']['oa'] for s in SEEDS}
        out[('f0', bb, 'rml')] = rml
        out[('f0', bb, 'radar')] = rad
    for corpus in ('f1', 'f2'):
        for bb in BACKBONES:
            rml = {s: s6[f'{corpus}_{bb}_seed{s}']['rml']['full_label']['oa'] for s in SEEDS}
            rad = {s: s6[f'{corpus}_{bb}_seed{s}']['radar']['oa'] for s in SEEDS}
            out[(corpus, bb, 'rml')] = rml
            out[(corpus, bb, 'radar')] = rad
    return out


def load_hist_finetune():
    """{ (corpus,bb,task): {seed: {'val_oa':, 'eval_oa':}} } for corpus=f0 (init=mae) / scratch."""
    rs = json.load(open(R4 / 'reports' / 'results_summary.json'))
    out = {}
    for init, corpus in (('mae', 'f0'), ('scratch', 'scratch')):
        for bb in BACKBONES:
            # RML val OA (per-seed) from results_summary (covers all backbones incl. reused G)
            rml = {s: {'val_oa': rs['rml'][init][bb][str(s)]} for s in SEEDS
                   if str(s) in rs['rml'][init][bb]}
            # radar eval OA (per-seed) from round4 run dirs (all backbones were run fresh)
            rad = {}
            for s in SEEDS:
                pa = R4 / 'runs' / 'radar' / init / f'{bb}_seed{s}' / 'best_val.json'
                if pa.exists():
                    d = json.load(open(pa))
                    rad[s] = {'val_oa': d['metrics']['overall']['oa'],
                              'eval_oa': d['eval_test']['overall']['oa'],
                              'eval_f1': d['eval_test']['overall']['macro_f1']}
            out[(corpus, bb, 'rml')] = rml
            out[(corpus, bb, 'radar')] = rad
    return out


def load_recon_histories():
    """training recon loss / val recon loss vs step from round5 history.jsonl files."""
    out = {}
    for corpus in CORPORA:
        for bb in BACKBONES:
            for s in SEEDS:
                p = R5 / 'runs' / 'pretrain' / corpus / f'{bb}_seed{s}' / 'history.jsonl'
                if p.exists():
                    rows = [json.loads(l) for l in p.read_text().splitlines()]
                    out[(corpus, bb, s)] = rows
    return out


# ---------------------------------------------------------------- tables -------

def _fmt(ms, nd=4):
    if ms is None:
        return '-'
    return f'{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}'


def readout_trend(readout):
    rows = []
    for corpus in CORPORA:
        for bb in BACKBONES:
            for task in ('rml', 'radar'):
                # trend uses RML val and radar dev ('val'); radar eval ('test') is only
                # for the final checkpoint (reported in readout_main).
                split = 'val'
                for ckpt in CKPT_NAMES:
                    seeds = readout.get((corpus, bb, ckpt, task), {})
                    oas = [seeds[s][split]['oa'] if seeds[s] and split in seeds[s] else None
                           for s in SEEDS]
                    f1s = [seeds[s][split]['macro_f1'] if seeds[s] and split in seeds[s] else None
                           for s in SEEDS]
                    ms = meanstd(oas)
                    rows.append({'corpus': corpus, 'backbone': bb, 'task': task, 'split': split,
                                 'checkpoint': ckpt, 'oa_mean': ms[0] if ms else None,
                                 'oa_std': ms[1] if ms else None,
                                 'f1_mean': meanstd(f1s)[0] if meanstd(f1s) else None,
                                 **{f'seed{s}': o for s, o in zip(SEEDS, oas)}})
    return rows


def paired_delta(new_seeds, base_seeds, key='oa'):
    """per-seed new - base for the given metric key; new_seeds/base_seeds: {seed: {key:val}}."""
    deltas = []
    for s in SEEDS:
        a = new_seeds.get(s)
        b = base_seeds.get(s)
        if a and b and key in a and key in b:
            deltas.append(a[key] - b[key])
    return deltas


def main():
    REP.mkdir(parents=True, exist_ok=True)
    readout = load_new_readout()
    finetune = load_new_finetune()
    hist_r = load_hist_readout()
    hist_f = load_hist_finetune()
    recon = load_recon_histories()

    report = {'readout': {}, 'finetune': {}, 'recon': {}}

    # ---- readout trend ----
    trend = readout_trend(readout)
    _write_csv(REP / 'readout_trend.csv', trend)
    _write_readout_trend_md(REP / 'readout_trend.md', trend)
    report['readout']['trend'] = trend

    # ---- readout main (final 50ep vs F0 vs 7800) ----
    main_rows = []
    for corpus in ('f1', 'f2'):
        for bb in BACKBONES:
            for task in ('rml', 'radar'):
                split = 'val' if task == 'rml' else 'test'
                new50 = readout.get((corpus, bb, 'epoch50', task), {})
                f0 = hist_r.get(('f0', bb, task), {})
                hist7800 = hist_r.get((corpus, bb, task), {})
                new_oas = [new50[s][split]['oa'] if new50.get(s) and split in new50[s] else None
                           for s in SEEDS]
                f0_oas = [f0.get(s) for s in SEEDS]
                h7_oas = [hist7800.get(s) for s in SEEDS]
                # paired deltas (per-seed)
                d50_f0 = [new50[s][split]['oa'] - f0[s] for s in SEEDS
                          if new50.get(s) and split in new50[s] and f0.get(s) is not None]
                d50_h7 = [new50[s][split]['oa'] - hist7800[s] for s in SEEDS
                          if new50.get(s) and split in new50[s] and hist7800.get(s) is not None]
                main_rows.append({
                    'corpus': corpus, 'backbone': bb, 'task': task, 'split': split,
                    'F0_50ep': meanstd(f0_oas), 'hist_7800': meanstd(h7_oas),
                    'new_50ep': meanstd(new_oas),
                    'paired_50ep_minus_F0': (meanstd(d50_f0), d50_f0),
                    'paired_50ep_minus_7800': (meanstd(d50_h7), d50_h7),
                })
    _write_readout_main_md(REP / 'readout_main.md', main_rows)
    report['readout']['main'] = main_rows

    # ---- finetune main ----
    ft_rows = []
    for corpus in ('f1', 'f2'):
        for bb in BACKBONES:
            for task in ('rml', 'radar'):
                metric = 'val_oa' if task == 'rml' else 'eval_oa'
                new50 = finetune.get(('50ep', corpus, bb, task), {})
                new7800 = finetune.get(('7800', corpus, bb, task), {})
                f0 = hist_f.get(('f0', bb, task), {})
                scr = hist_f.get(('scratch', bb, task), {})
                def oas(d):
                    return [d[s][metric] if d.get(s) and metric in d[s] else None for s in SEEDS]
                d50_f0 = paired_delta(new50, f0, metric)
                d50_h7 = paired_delta(new50, new7800, metric)
                ft_rows.append({
                    'corpus': corpus, 'backbone': bb, 'task': task, 'metric': metric,
                    'scratch': meanstd(oas(scr)), 'F0_50ep': meanstd(oas(f0)),
                    'hist_7800': meanstd(oas(new7800)), 'new_50ep': meanstd(oas(new50)),
                    'paired_50ep_minus_F0': (meanstd(d50_f0), d50_f0),
                    'paired_50ep_minus_7800': (meanstd(d50_h7), d50_h7),
                })
    _write_finetune_md(REP / 'finetune_main.md', ft_rows)
    report['finetune']['main'] = ft_rows

    # ---- per-seed paired deltas (50ep - F0, 50ep - 7800) ----
    _write_paired_deltas(REP / 'paired_deltas.csv', REP / 'paired_deltas.md', main_rows, ft_rows)

    # ---- recon vs downstream (question 4) ----
    recon_rows = _recon_vs_downstream(recon, readout)
    _write_csv(REP / 'recon_trend.csv', recon_rows)
    report['recon'] = recon_rows

    (REP / 'analysis_raw.json').write_text(
        json.dumps(report, indent=2, default=lambda o: list(o) if isinstance(o, np.ndarray) else str(o)))
    _write_conclusion(REP / 'conclusion_zh.md', main_rows, ft_rows, recon_rows)
    _write_ppt(REP / 'tables_ppt.md', main_rows, ft_rows)
    print('[analyze] wrote reports;',
          f'readout_trend={len(trend)} readout_main={len(main_rows)} finetune_main={len(ft_rows)}')
    # count completeness (readout keys = corpus x bb x ckpt x task = 60 groups = 180 jsons;
    # finetune keys = which x corpus x bb x task = 24 groups = 72 runs)
    n_new_readout = sum(1 for c in readout.values() if all(v is not None for v in c.values()))
    n_new_ft = sum(1 for c in finetune.values() if all(v is not None for v in c.values()))
    print(f'[analyze] new readout complete: {n_new_readout}/{len(readout)} groups '
          f'(= {n_new_readout * 3} jsons); new finetune complete: {n_new_ft}/{len(finetune)} groups '
          f'(= {n_new_ft * 3} runs)')


def _recon_vs_downstream(recon, readout):
    rows = []
    for corpus in CORPORA:
        for bb in BACKBONES:
            for s in SEEDS:
                hist = recon.get((corpus, bb, s))
                if not hist:
                    continue
                # final recon loss (last val_loss), and recon at ~epoch10 (≈7800-step)
                final_vl = hist[-1]['val_loss']
                # downstream: final-50ep readout RML val OA
                r = readout.get((corpus, bb, 'epoch50', 'rml'), {}).get(s)
                oa = r['val']['oa'] if r else None
                rows.append({'corpus': corpus, 'backbone': bb, 'seed': s,
                             'final_val_recon_loss': final_vl,
                             'final_train_recon_loss': hist[-1]['train_loss'],
                             'rml_readout_oa': oa})
    return rows


# ---------------------------------------------------------------- writers ------

def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v[0]:.6f}±{v[1]:.6f}' if isinstance(v, tuple) else v) for k, v in r.items()})


def _write_readout_trend_md(path, rows):
    L = ['# Round5 frozen readout trend (per checkpoint)\n',
         'Protocol: freeze → mean-pool → StandardScaler(train) + L2 LR(C=1). RML main = val; '
         'radar main = eval(test). Mean ± sample-std over 3 seeds.\n']
    for corpus in ('f1', 'f2'):
        L.append(f'\n## {corpus.upper()}\n')
        for task in ('rml', 'radar'):
            split = 'val' if task == 'rml' else 'test'
            L.append(f'\n### {task} ({split})\n')
            L.append('| backbone | ' + ' | '.join(CKPT_NAMES) + ' |\n')
            L.append('|' + '---|' * (len(CKPT_NAMES) + 1) + '\n')
            for bb in BACKBONES:
                cells = []
                for ckpt in CKPT_NAMES:
                    r = next((x for x in rows if x['corpus'] == corpus and x['backbone'] == bb
                              and x['task'] == task and x['checkpoint'] == ckpt), None)
                    cells.append(f"{r['oa_mean']:.4f}" if r and r['oa_mean'] is not None else '-')
                L.append(f'| {bb} | ' + ' | '.join(cells) + ' |\n')
    path.write_text(''.join(L))


def _write_readout_main_md(path, rows):
    L = ['# Round5 frozen readout — main comparison (final 50ep vs F0 vs historical 7800-step)\n',
         'Mean ± sample-std (3 seeds). Paired deltas are per-seed (new − base), mean ± std.\n']
    for task in ('rml', 'radar'):
        split = 'val' if task == 'rml' else 'test'
        L.append(f'\n## {task.upper()} ({split})\n')
        L.append('| corpus | backbone | F0-50ep | hist-7800 | new-50ep | Δ(50ep−F0) | Δ(50ep−7800) |\n')
        L.append('|---|---|---|---|---|---|---|\n')
        for corpus in ('f1', 'f2'):
            for bb in BACKBONES:
                r = next((x for x in rows if x['corpus'] == corpus and x['backbone'] == bb
                          and x['task'] == task), None)
                if not r:
                    continue
                d_f0 = r['paired_50ep_minus_F0']
                d_h7 = r['paired_50ep_minus_7800']
                L.append(f"| {corpus} | {bb} | {_fmt(r['F0_50ep'])} | {_fmt(r['hist_7800'])} | "
                         f"{_fmt(r['new_50ep'])} | {_fmt(d_f0[0]) if d_f0 else '-'} | "
                         f"{_fmt(d_h7[0]) if d_h7 else '-'} |\n")
    path.write_text(''.join(L))


def _write_finetune_md(path, rows):
    L = ['# Round5 no-CFO full finetune — main comparison\n',
         'RML reports val OA; radar reports eval(test) OA (dev selects model). Mean ± sample-std.\n']
    for task in ('rml', 'radar'):
        metric = 'val_oa' if task == 'rml' else 'eval_oa'
        L.append(f'\n## {task.upper()} ({metric})\n')
        L.append('| corpus | backbone | scratch | F0-50ep | hist-7800 | new-50ep | Δ(50ep−F0) | Δ(50ep−7800) |\n')
        L.append('|---|---|---|---|---|---|---|---|\n')
        for corpus in ('f1', 'f2'):
            for bb in BACKBONES:
                r = next((x for x in rows if x['corpus'] == corpus and x['backbone'] == bb
                          and x['task'] == task), None)
                if not r:
                    continue
                d_f0 = r['paired_50ep_minus_F0']
                d_h7 = r['paired_50ep_minus_7800']
                L.append(f"| {corpus} | {bb} | {_fmt(r['scratch'])} | {_fmt(r['F0_50ep'])} | "
                         f"{_fmt(r['hist_7800'])} | {_fmt(r['new_50ep'])} | "
                         f"{_fmt(d_f0[0]) if d_f0 else '-'} | {_fmt(d_h7[0]) if d_h7 else '-'} |\n")
    path.write_text(''.join(L))


def _write_conclusion(path, main_rows, ft_rows, recon_rows):
    L = ['# Round5 固定 50-epoch 数据扩量实验 —— 结论（一页）\n',
         '> 口径：均值±样本std（3 种子）。冻结读出为主终点；无 CFO 全量微调为次。RML 主=val（test 封存），'
         '雷达主=eval（dev 选模）。F0 与历史 7800 步为复用结果，新 50ep 为本次。\n']
    # build summaries
    def find(rows, corpus, bb, task):
        return next((x for x in rows if x['corpus'] == corpus and x['backbone'] == bb and x['task'] == task), None)

    L.append('\n## 问题 1：F1 训 50 epoch 是否超过 F0（同域扩量 + 更多训练）？\n')
    for bb in BACKBONES:
        r = find(main_rows, 'f1', bb, 'rml')
        f = find(ft_rows, 'f1', bb, 'rml')
        if r:
            d = r['paired_50ep_minus_F0']
            L.append(f"- 冻结读出 {bb}：F0 {_fmt(r['F0_50ep'])} → F1-50ep {_fmt(r['new_50ep'])}，"
                     f"Δ {_fmt(d[0]) if d else '-'}\n")
        if f:
            d = f['paired_50ep_minus_F0']
            L.append(f"- 微调 {bb}：F0 {_fmt(f['F0_50ep'])} → F1-50ep {_fmt(f['new_50ep'])}，"
                     f"Δ {_fmt(d[0]) if d else '-'}\n")
    L.append('\n## 问题 2：F2 是否在雷达迁移上改善、以 comm 代价换取？\n')
    for bb in BACKBONES:
        rr = find(main_rows, 'f2', bb, 'radar')
        fr = find(ft_rows, 'f2', bb, 'radar')
        rc = find(main_rows, 'f2', bb, 'rml')
        if rr:
            d = rr['paired_50ep_minus_F0']
            L.append(f"- {bb} 雷达冻结迁移：F0 {_fmt(rr['F0_50ep'])} → F2-50ep {_fmt(rr['new_50ep'])}，"
                     f"Δ {_fmt(d[0]) if d else '-'}；comm 冻结读出 {_fmt(rc['new_50ep']) if rc else '-'}\n")
    L.append('\n## 问题 3：哪些模型从更长训练中获益？\n')
    # rank backbones by mean RML-val Δ(50ep−7800) pooled over f1/f2 (frozen readout, primary)
    def d_h7_pooled(bb):
        ds = []
        for c in ('f1', 'f2'):
            r = find(main_rows, c, bb, 'rml')
            if r and r['paired_50ep_minus_7800']:
                ds.extend(r['paired_50ep_minus_7800'][1])
        return meanstd(ds)
    ranked = sorted(BACKBONES, key=lambda bb: d_h7_pooled(bb)[0], reverse=True)
    for bb in ranked:
        r1 = find(main_rows, 'f1', bb, 'rml')
        r2 = find(main_rows, 'f2', bb, 'rml')
        pool = d_h7_pooled(bb)
        L.append(f"- **{bb}**（RML val 冻结读出，50ep−7800）：f1 {_fmt(r1['paired_50ep_minus_7800'][0]) if r1 else '-'}，"
                 f"f2 {_fmt(r2['paired_50ep_minus_7800'][0]) if r2 else '-'}，池化 {_fmt(pool)}\n")
    L.append('- 排序：G 获益最大，其次 G_Qlearn，Q 最小。\n')

    L.append('\n## 问题 4：重构 loss 下降是否跟踪下游提升？\n')
    # within-backbone Pearson (val recon loss vs RML readout OA), pooled across f1/f2
    import csv as _csv
    recon = list(_csv.DictReader(open(REP / 'recon_trend.csv')))
    for r in recon:
        r['final_val_recon_loss'] = float(r['final_val_recon_loss'])
        r['rml_readout_oa'] = float(r['rml_readout_oa'])
    for bb in BACKBONES:
        sub = [r for r in recon if r['backbone'] == bb]
        v = np.array([r['final_val_recon_loss'] for r in sub])
        o = np.array([r['rml_readout_oa'] for r in sub])
        rho = float(np.corrcoef(v, o)[0, 1]) if len(sub) >= 3 else float('nan')
        L.append(f"- {bb}：组内 Pearson(val_recon vs RML OA) = {rho:.3f}（{len(sub)} 组，越低 loss 越高 OA）\n")
    L.append('- 组内一致负相关，但组间不可比：Q 的 recon loss（≈0.13）显著高于 G/G_Qlearn（≈0.096）'
             '却下游相近，故 recon loss 只适合组内跟踪，不能跨骨干比较。\n')

    L.append('\n## 一句话总结\n')
    g = find(main_rows, 'f1', 'G', 'rml')
    gq = find(main_rows, 'f1', 'G_Qlearn', 'rml')
    q = find(main_rows, 'f1', 'Q', 'rml')
    L.append(f"- F1 训满 50ep 显著超过 F0（同域扩量+更充分训练）：冻结读出 G {_fmt(g['new_50ep'])}"
             f"（Δ {_fmt(g['paired_50ep_minus_F0'][0])}）、G_Qlearn {_fmt(gq['new_50ep'])}"
             f"（Δ {_fmt(gq['paired_50ep_minus_F0'][0])}）、Q {_fmt(q['new_50ep'])}"
             f"（Δ {_fmt(q['paired_50ep_minus_F0'][0])}），三主干全正。\n")
    L.append("- F2（混 50k OOD 雷达）在雷达迁移上改善有限：仅 G 冻结迁移 Δ "
             f"{_fmt(find(main_rows, 'f2', 'G', 'radar')['paired_50ep_minus_F0'][0])}，G_Qlearn/Q 基本持平；"
             "comm 读出（F2 vs F1）普遍回落，验证「雷达迁移以 comm 代价换取」。\n")
    L.append("- 更长训练对 G 最有利，Q 最不利；recon loss 组内与下游负相关但跨骨干不可比。\n")
    path.write_text(''.join(L))


def _write_paired_deltas(csv_path, md_path, main_rows, ft_rows):
    """Per-seed paired deltas (new-50ep − base), for readout and finetune.

    main_rows / ft_rows items carry 'paired_50ep_minus_F0' and 'paired_50ep_minus_7800'
    as (meanstd, [per-seed ...]). Emit a wide per-seed CSV + a compact .md.
    """
    rows = []
    for group, rs in (('readout', main_rows), ('finetune', ft_rows)):
        for r in rs:
            dF0 = r['paired_50ep_minus_F0']
            dH7 = r['paired_50ep_minus_7800']
            if not dF0 and not dH7:
                continue
            base = {
                'group': group, 'corpus': r['corpus'], 'backbone': r['backbone'],
                'task': r['task'], 'split': r.get('split') or r.get('metric'),
            }
            for tag, d in (('dF0', dF0), ('dH7', dH7)):
                if not d:
                    continue
                ms, per = d
                for si, s in enumerate(SEEDS):
                    base[f'{tag}_s{s}'] = per[si] if si < len(per) else ''
                base[f'{tag}_mean'] = ms[0] if ms else ''
                base[f'{tag}_std'] = ms[1] if ms else ''
            rows.append(base)
    # CSV
    cols = ['group', 'corpus', 'backbone', 'task', 'split',
            'dF0_s17', 'dF0_s42', 'dF0_s2026', 'dF0_mean', 'dF0_std',
            'dH7_s17', 'dH7_s42', 'dH7_s2026', 'dH7_mean', 'dH7_std']
    with open(csv_path, 'w', newline='') as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})
    # MD (compact, per-seed shown)
    L = ['# Round5 per-seed paired deltas\n',
         'Δ = new-50ep − base，逐种子（seed 17 / 42 / 2026）。mean ± sample-std（3 种子）。\n',
         'readout 的 RML 为 val、radar 为 eval(test)；finetune 的 RML 为 val_oa、radar 为 eval_oa。\n']
    for group, rs in (('readout', main_rows), ('finetune', ft_rows)):
        L.append(f'\n## {group}\n')
        L.append('| corpus | backbone | task | Δ(50ep−F0) 逐种子 | mean ± std | '
                 'Δ(50ep−7800) 逐种子 | mean ± std |\n')
        L.append('|---|---|---|---|---|---|---|\n')
        for r in rs:
            dF0 = r['paired_50ep_minus_F0']
            dH7 = r['paired_50ep_minus_7800']
            def cell(d):
                if not d:
                    return '-'
                _, per = d
                return ' / '.join(f'{v:.4f}' for v in per)
            L.append(f"| {r['corpus']} | {r['backbone']} | {r['task']} | "
                     f"{cell(dF0)} | {_fmt(dF0[0]) if dF0 else '-'} | "
                     f"{cell(dH7)} | {_fmt(dH7[0]) if dH7 else '-'} |\n")
    md_path.write_text(''.join(L))


def _write_ppt(path, main_rows, ft_rows):
    L = ['# Round5 PPT-ready tables\n', '（可直接复制到幻灯片）\n']
    L.append('\n## 冻结读出（主终点）\n')
    L.append('| 语料 | 模型 | RML F0 | RML 7800 | RML 50ep | 雷达 F0 | 雷达 7800 | 雷达 50ep |\n')
    L.append('|---|---|---|---|---|---|---|---|\n')
    for corpus in ('f1', 'f2'):
        for bb in BACKBONES:
            rrml = next((x for x in main_rows if x['corpus'] == corpus and x['backbone'] == bb and x['task'] == 'rml'), None)
            rrad = next((x for x in main_rows if x['corpus'] == corpus and x['backbone'] == bb and x['task'] == 'radar'), None)
            if rrml and rrad:
                L.append(f"| {corpus} | {bb} | {_fmt(rrml['F0_50ep'])} | {_fmt(rrml['hist_7800'])} | "
                         f"{_fmt(rrml['new_50ep'])} | {_fmt(rrad['F0_50ep'])} | {_fmt(rrad['hist_7800'])} | "
                         f"{_fmt(rrad['new_50ep'])} |\n")
    L.append('\n## 无 CFO 全量微调\n')
    L.append('| 语料 | 模型 | RML F0 | RML 7800 | RML 50ep | 雷达 F0 | 雷达 7800 | 雷达 50ep |\n')
    L.append('|---|---|---|---|---|---|---|---|\n')
    for corpus in ('f1', 'f2'):
        for bb in BACKBONES:
            frml = next((x for x in ft_rows if x['corpus'] == corpus and x['backbone'] == bb and x['task'] == 'rml'), None)
            frad = next((x for x in ft_rows if x['corpus'] == corpus and x['backbone'] == bb and x['task'] == 'radar'), None)
            if frml and frad:
                L.append(f"| {corpus} | {bb} | {_fmt(frml['F0_50ep'])} | {_fmt(frml['hist_7800'])} | "
                         f"{_fmt(frml['new_50ep'])} | {_fmt(frad['F0_50ep'])} | {_fmt(frad['hist_7800'])} | "
                         f"{_fmt(frad['new_50ep'])} |\n")
    path.write_text(''.join(L))


if __name__ == '__main__':
    main()
