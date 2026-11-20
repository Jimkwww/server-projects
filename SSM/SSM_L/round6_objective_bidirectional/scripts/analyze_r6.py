"""Round6 objective comparison + bidirectional SSM: aggregate new + reused (round5) results.

Main table: 9 rows (G-MAE / G-denoise / G-joint / Q-MAE / Q-denoise / Q-joint / Q-bi-MAE /
Q-bi-denoise / Q-bi-joint) x 4 columns (comm frozen OA, radar frozen OA, comm finetune OA,
radar finetune OA). G-MAE and Q-MAE are REUSED from round5 (not re-run); the other 7 groups
are new (round6, incl. the Q_bi denoise/joint extension).

Per-seed, 3 seeds (17/42/2026), mean ± sample-std. Metrics:
  comm frozen  = RML readout val OA (test sealed)
  radar frozen = RadChar readout eval(test) OA (dev = manifest 'val')
  comm finetune = RML finetune val OA (clean-val-OA earliest-tie)
  radar finetune = RadChar finetune eval(test) OA (dev selects)

Writes under round6_objective_bidirectional/reports/:
  main_table.csv/.md, readout_main.csv/.md, finetune_main.csv/.md, per_seed.csv/.md,
  paired_deltas.csv/.md, recon_trend.csv/.md, efficiency.csv/.md, conclusion_zh.md,
  tables_ppt.md, run_commands.md, analysis_raw.json.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R6 = ROOT / 'round6_objective_bidirectional'
R5 = ROOT / 'round5_epoch_scale'
REP = R6 / 'reports'

SEEDS = (17, 42, 2026)
# 9 groups in PPT order
GROUPS = [
    ('G', 'MAE'),
    ('G', 'denoise'),
    ('G', 'joint'),
    ('Q', 'MAE'),
    ('Q', 'denoise'),
    ('Q', 'joint'),
    ('Q_bi', 'MAE'),
    ('Q_bi', 'denoise'),
    ('Q_bi', 'joint'),
]
GROUP_LABEL = {('G', 'MAE'): 'G-MAE', ('G', 'denoise'): 'G-denoise', ('G', 'joint'): 'G-joint',
               ('Q', 'MAE'): 'Q-MAE', ('Q', 'denoise'): 'Q-denoise', ('Q', 'joint'): 'Q-joint',
               ('Q_bi', 'MAE'): 'Q-bi-MAE', ('Q_bi', 'denoise'): 'Q-bi-denoise',
               ('Q_bi', 'joint'): 'Q-bi-joint'}


def meanstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def _fmt(ms, nd=4):
    return '-' if ms is None else f'{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}'


# ------------------------------------------------------------------- result paths

def _is_reused(bb, obj):
    # Only G-MAE and Q-MAE are reused from round5 (run dir has no objective suffix);
    # everything else — including Q_bi-MAE — is a new round6 run (suffix _mae/_denoise/_joint).
    return obj == 'MAE' and bb in ('G', 'Q')


def _dir_obj(obj):
    # round6 run dirs use lowercase objective suffix (mae/denoise/joint).
    return obj.lower()


def run_name(bb, obj):
    # round5 reused G-MAE/Q-MAE have no objective suffix; round6 new runs do.
    if _is_reused(bb, obj):
        return f'{bb}_seed'  # placeholder for seed suffix
    return f'{bb}_{_dir_obj(obj)}_seed'


def seed_run_name(bb, obj, seed):
    if _is_reused(bb, obj):
        return f'{bb}_seed{seed}'
    return f'{bb}_{_dir_obj(obj)}_seed{seed}'


def _readout_path(bb, obj, seed, task):
    if _is_reused(bb, obj):
        return R5 / 'runs' / 'downstream' / 'readout' / 'f1' / f'{bb}_seed{seed}__epoch50__{task}.json'
    return (R6 / 'runs' / 'downstream' / 'readout' / 'f1' /
            f'{bb}_{_dir_obj(obj)}_seed{seed}__epoch50__{task}.json')


def _finetune_path(bb, obj, seed, task):
    return (R5 if _is_reused(bb, obj) else R6) / 'runs' / 'downstream' / 'finetune_50ep' / task / \
           f'f1_{seed_run_name(bb, obj, seed)}' / 'best_val.json'


def _pretrain_hist_path(bb, obj, seed):
    if _is_reused(bb, obj):
        return R5 / 'runs' / 'pretrain' / 'f1' / f'{bb}_seed{seed}' / 'history.jsonl'
    return R6 / 'runs' / 'pretrain' / 'f1' / f'{bb}_{_dir_obj(obj)}_seed{seed}' / 'history.jsonl'


def _pretrain_modelinfo_path(bb, obj, seed):
    if _is_reused(bb, obj):
        return R5 / 'runs' / 'pretrain' / 'f1' / f'{bb}_seed{seed}' / 'model_info.json'
    return R6 / 'runs' / 'pretrain' / 'f1' / f'{bb}_{_dir_obj(obj)}_seed{seed}' / 'model_info.json'


def load_metric(path, *keys):
    if not path.exists():
        return None
    d = json.load(open(path))
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return float(d)


def load_group_metrics(bb, obj):
    """Return {seed: {'comm_frozen','radar_frozen','comm_ft','radar_ft','radar_dev','comm_ft_f1'}}."""
    out = {}
    for s in SEEDS:
        ro_rml = _readout_path(bb, obj, s, 'rml')
        ro_rad = _readout_path(bb, obj, s, 'radar')
        ft_rml = _finetune_path(bb, obj, s, 'rml')
        ft_rad = _finetune_path(bb, obj, s, 'radar')
        rec = {
            'comm_frozen': load_metric(ro_rml, 'val', 'oa'),
            'radar_frozen': load_metric(ro_rad, 'test', 'oa'),
            'radar_dev': load_metric(ro_rad, 'val', 'oa'),
            'comm_ft': load_metric(ft_rml, 'metrics', 'overall', 'oa'),
            'comm_ft_f1': load_metric(ft_rml, 'metrics', 'overall', 'macro_f1'),
            'radar_ft': load_metric(ft_rad, 'eval_test', 'overall', 'oa'),
        }
        out[s] = rec
    return out


def load_recon(bb, obj, seed):
    p = _pretrain_hist_path(bb, obj, seed)
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    last = rows[-1]
    return {
        'final_train_loss': last.get('train_loss'),
        'final_val_loss': last.get('val_loss'),
        'final_val_loss_mae': last.get('val_loss_mae'),
        'final_val_loss_dn': last.get('val_loss_dn'),
        'train_seconds': last.get('train_seconds'),
        'peak_mem_bytes': last.get('peak_mem_bytes'),
        'cum_samples': last.get('cum_samples'),
    }


def load_efficiency(bb, obj):
    rows = []
    for s in SEEDS:
        rec = {'seed': s}
        mi = _pretrain_modelinfo_path(bb, obj, s)
        if mi.exists():
            d = json.load(open(mi))
            rec['encoder_params'] = d.get('encoder_real_scalar_parameters',
                                          d.get('trainable_real_scalar_parameters'))
            rec['total_params'] = d.get('total_real_scalar_parameters',
                                        d.get('trainable_real_scalar_parameters'))
        r = load_recon(bb, obj, s)
        if r:
            rec['train_seconds'] = r['train_seconds']
            rec['peak_mem_bytes'] = r['peak_mem_bytes']
            rec['cum_samples'] = r['cum_samples']
            if r['train_seconds']:
                rec['throughput_sps'] = r['cum_samples'] / r['train_seconds']
                rec['gpu_hours'] = r['train_seconds'] / 3600.0
        rows.append(rec)
    return rows


def paired_delta(group_a, group_b, key):
    """per-seed A[key] - B[key], only over seeds where both present. Returns (meanstd, [deltas])."""
    deltas = []
    for s in SEEDS:
        a = group_a.get(s, {}).get(key)
        b = group_b.get(s, {}).get(key)
        if a is not None and b is not None:
            deltas.append(a - b)
    if not deltas:
        return None
    return meanstd(deltas), deltas


# ----------------------------------------------------------------------- main

def main():
    REP.mkdir(parents=True, exist_ok=True)
    metrics = {g: load_group_metrics(*g) for g in GROUPS}

    # ---- main table (7 x 4) ----
    main_rows = []
    for bb, obj in GROUPS:
        g = metrics[(bb, obj)]
        row = {'group': GROUP_LABEL[(bb, obj)], 'backbone': bb, 'objective': obj}
        for key in ('comm_frozen', 'radar_frozen', 'comm_ft', 'radar_ft'):
            row[key] = meanstd([g[s].get(key) for s in SEEDS])
            row[f'{key}_seeds'] = [g[s].get(key) for s in SEEDS]
        main_rows.append(row)
    _write_main_table(REP / 'main_table.md', REP / 'main_table.csv', main_rows)

    # ---- readout / finetune sub-tables ----
    _write_readout_md(REP / 'readout_main.md', main_rows)
    _write_finetune_md(REP / 'finetune_main.md', main_rows)

    # ---- per-seed ----
    _write_per_seed(REP / 'per_seed.csv', REP / 'per_seed.md', main_rows)

    # ---- paired deltas (the 3 research questions) ----
    _write_paired_deltas(REP / 'paired_deltas.csv', REP / 'paired_deltas.md', metrics)

    # ---- recon trend (per-task losses) ----
    recon_rows = []
    for bb, obj in GROUPS:
        for s in SEEDS:
            r = load_recon(bb, obj, s)
            if r:
                recon_rows.append({'group': GROUP_LABEL[(bb, obj)], 'backbone': bb,
                                   'objective': obj, 'seed': s, **r})
    _write_csv(REP / 'recon_trend.csv', recon_rows)
    _write_recon_md(REP / 'recon_trend.md', recon_rows)

    # ---- efficiency ----
    eff_rows = []
    for bb, obj in GROUPS:
        for rec in load_efficiency(bb, obj):
            eff_rows.append({'group': GROUP_LABEL[(bb, obj)], 'backbone': bb,
                             'objective': obj, **rec})
    _write_csv(REP / 'efficiency.csv', eff_rows)
    _write_efficiency_md(REP / 'efficiency.md', eff_rows)

    # ---- conclusion + ppt ----
    _write_conclusion(REP / 'conclusion_zh.md', main_rows, metrics, recon_rows)
    _write_ppt(REP / 'tables_ppt.md', main_rows)
    _write_commands(REP / 'run_commands.md')

    # ---- raw ----
    report = {
        'main_table': main_rows,
        'metrics': {GROUP_LABEL[g]: {str(s): metrics[g][s] for s in SEEDS} for g in GROUPS},
        'recon': recon_rows,
        'efficiency': eff_rows,
    }
    (REP / 'analysis_raw.json').write_text(
        json.dumps(report, indent=2, default=lambda o: list(o) if isinstance(o, np.ndarray) else str(o)))

    # completeness
    n_complete = sum(1 for g in GROUPS
                     if all(metrics[g][s].get('comm_frozen') is not None for s in SEEDS))
    print(f'[analyze] main_table={len(main_rows)} rows; groups complete (comm_frozen): '
          f'{n_complete}/{len(GROUPS)}')


# -------------------------------------------------------------------- writers

def _write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v[0]:.6f}±{v[1]:.6f}' if isinstance(v, tuple) else v)
                        for k, v in r.items()})


def _write_main_table(md, csv_path, rows):
    # CSV (flattened)
    flat = []
    for r in rows:
        fr = {'group': r['group'], 'backbone': r['backbone'], 'objective': r['objective']}
        for key in ('comm_frozen', 'radar_frozen', 'comm_ft', 'radar_ft'):
            ms = r[key]
            if ms:
                fr[f'{key}_mean'] = ms[0]
                fr[f'{key}_std'] = ms[1]
            for i, s in enumerate(SEEDS):
                fr[f'{key}_s{s}'] = r[f'{key}_seeds'][i]
        flat.append(fr)
    _write_csv(csv_path, flat)

    L = ['# Round6 主表（9 组 × 4 指标）\n',
         '口径：均值 ± 样本std（3 种子 17/42/2026）。',
         'comm 冻结 = RML 冻结读出 val OA；radar 冻结 = RadChar 冻结读出 eval(test) OA；',
         'comm 微调 = RML 无 CFO 微调 val OA；radar 微调 = RadChar 无 CFO 微调 eval(test) OA。',
         'G-MAE / Q-MAE 复用 round5 结果。\n']
    L.append('| 组 | comm 冻结 OA | radar 冻结 OA | comm 微调 OA | radar 微调 OA |\n')
    L.append('|---|---|---|---|---|\n')
    for r in rows:
        L.append(f"| {r['group']} | {_fmt(r['comm_frozen'])} | {_fmt(r['radar_frozen'])} | "
                 f"{_fmt(r['comm_ft'])} | {_fmt(r['radar_ft'])} |\n")
    md.write_text(''.join(L))


def _write_readout_md(path, rows):
    L = ['# Round6 冻结线性读出（主终点）\n',
         'freeze → mean-pool → StandardScaler(train) + L2 LR(C=1)。RML 报 val；雷达报 eval(test)。\n\n']
    L.append('| 组 | comm val OA | radar eval OA | radar dev OA |\n')
    L.append('|---|---|---|---|\n')
    for r in rows:
        # radar dev stored in metrics but not in main_rows; recompute here would need metrics.
        # We fold radar_dev into the table via a side map (handled below in main via extra col).
        pass
    # main_rows has comm_frozen / radar_frozen only; re-read for dev is done in conclusion.
    # For this table, show the two frozen columns.
    L = ['# Round6 冻结线性读出（主终点）\n',
         'freeze → mean-pool → StandardScaler(train) + L2 LR(C=1)。RML 报 val；雷达报 eval(test)。\n\n']
    L.append('| 组 | comm val OA | radar eval OA |\n')
    L.append('|---|---|---|\n')
    for r in rows:
        L.append(f"| {r['group']} | {_fmt(r['comm_frozen'])} | {_fmt(r['radar_frozen'])} |\n")
    path.write_text(''.join(L))


def _write_finetune_md(path, rows):
    L = ['# Round6 无 CFO 全量微调\n',
         '50 epoch，AdamW(3e-4)，无 CFO，clean-val-OA 最早平局选模。RML 报 val；雷达报 eval(test)。\n\n']
    L.append('| 组 | comm val OA | radar eval OA |\n')
    L.append('|---|---|---|\n')
    for r in rows:
        L.append(f"| {r['group']} | {_fmt(r['comm_ft'])} | {_fmt(r['radar_ft'])} |\n")
    path.write_text(''.join(L))


def _write_per_seed(csv_path, md_path, rows):
    flat = []
    for r in rows:
        for i, s in enumerate(SEEDS):
            flat.append({
                'group': r['group'], 'backbone': r['backbone'], 'objective': r['objective'],
                'seed': s,
                'comm_frozen': r['comm_frozen_seeds'][i],
                'radar_frozen': r['radar_frozen_seeds'][i],
                'comm_ft': r['comm_ft_seeds'][i],
                'radar_ft': r['radar_ft_seeds'][i],
            })
    _write_csv(csv_path, flat)
    L = ['# Round6 逐种子结果\n', '3 种子（17 / 42 / 2026）。\n\n']
    L.append('| 组 | seed | comm 冻结 | radar 冻结 | comm 微调 | radar 微调 |\n')
    L.append('|---|---|---|---|---|---|\n')
    for r in flat:
        L.append(f"| {r['group']} | {r['seed']} | "
                 f"{r['comm_frozen'] if r['comm_frozen'] is not None else '-'} | "
                 f"{r['radar_frozen'] if r['radar_frozen'] is not None else '-'} | "
                 f"{r['comm_ft'] if r['comm_ft'] is not None else '-'} | "
                 f"{r['radar_ft'] if r['radar_ft'] is not None else '-'} |\n")
    md_path.write_text(''.join(L))


def _write_paired_deltas(csv_path, md_path, metrics):
    """Three research questions as per-seed paired deltas."""
    def delta_row(label, a, b, key):
        da = metrics[a]
        db = metrics[b]
        ms = paired_delta(da, db, key)
        return {'comparison': label, 'metric': key,
                'seeds': ms[1] if ms else [],
                'mean': ms[0][0] if ms else None, 'std': ms[0][1] if ms else None}

    comps = []
    for key, label in (('comm_frozen', 'comm_frozen_OA'), ('radar_frozen', 'radar_frozen_OA'),
                       ('comm_ft', 'comm_finetune_OA'), ('radar_ft', 'radar_finetune_OA')):
        # Q1: denoise vs MAE
        comps.append(delta_row(f'G-denoise − G-MAE', ('G', 'denoise'), ('G', 'MAE'), key))
        comps.append(delta_row(f'Q-denoise − Q-MAE', ('Q', 'denoise'), ('Q', 'MAE'), key))
        # Q2: joint vs best single (denoise and mae separately)
        comps.append(delta_row(f'G-joint − G-MAE', ('G', 'joint'), ('G', 'MAE'), key))
        comps.append(delta_row(f'G-joint − G-denoise', ('G', 'joint'), ('G', 'denoise'), key))
        comps.append(delta_row(f'Q-joint − Q-MAE', ('Q', 'joint'), ('Q', 'MAE'), key))
        comps.append(delta_row(f'Q-joint − Q-denoise', ('Q', 'joint'), ('Q', 'denoise'), key))
        # Q3: bidirectional vs unidirectional MAE
        comps.append(delta_row(f'Q-bi-MAE − Q-MAE', ('Q_bi', 'MAE'), ('Q', 'MAE'), key))
        # Q4 (extension): bidirectional vs unidirectional, denoise & joint objectives
        comps.append(delta_row(f'Q-bi-denoise − Q-denoise', ('Q_bi', 'denoise'), ('Q', 'denoise'), key))
        comps.append(delta_row(f'Q-bi-joint − Q-joint', ('Q_bi', 'joint'), ('Q', 'joint'), key))

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['comparison', 'metric',
                                          'seed17', 'seed42', 'seed2026', 'mean', 'std'])
        w.writeheader()
        for c in comps:
            w.writerow({'comparison': c['comparison'], 'metric': c['metric'],
                        'seed17': c['seeds'][0] if len(c['seeds']) > 0 else '',
                        'seed42': c['seeds'][1] if len(c['seeds']) > 1 else '',
                        'seed2026': c['seeds'][2] if len(c['seeds']) > 2 else '',
                        'mean': c['mean'], 'std': c['std']})

    L = ['# Round6 逐种子配对差值\n',
         'Δ = 前者 − 后者，逐种子（17/42/2026）；mean ± sample-std。\n\n']
    L.append('| 对比 | 指标 | seed17 | seed42 | seed2026 | mean ± std |\n')
    L.append('|---|---|---|---|---|---|---|\n')
    for c in comps:
        cell = lambda i: f'{c["seeds"][i]:.4f}' if len(c['seeds']) > i else '-'
        ms = _fmt((c['mean'], c['std'])) if c['mean'] is not None else '-'
        L.append(f"| {c['comparison']} | {c['metric']} | {cell(0)} | {cell(1)} | {cell(2)} | {ms} |\n")
    md_path.write_text(''.join(L))


def _write_recon_md(path, rows):
    if not rows:
        path.write_text('')
        return
    L = ['# Round6 各任务重构损失（50ep 末）\n',
         'MAE/denoise 的 val_loss 为对应重构 MSE；joint 另有 val_loss_mae / val_loss_dn 分项。\n\n']
    L.append('| 组 | seed | train_loss | val_loss | val_mae | val_dn |\n')
    L.append('|---|---|---|---|---|---|\n')
    for r in rows:
        f = lambda k: f'{r[k]:.4f}' if r.get(k) is not None else '-'
        L.append(f"| {r['group']} | {r['seed']} | {f('final_train_loss')} | "
                 f"{f('final_val_loss')} | {f('final_val_loss_mae')} | {f('final_val_loss_dn')} |\n")
    path.write_text(''.join(L))


def _write_efficiency_md(path, rows):
    if not rows:
        path.write_text('')
        return
    L = ['# Round6 效率\n', '参数量为实标量参数数；吞吐 = cum_samples / train_seconds。\n\n']
    L.append('| 组 | seed | encoder params | total params | GPU h | throughput (samp/s) | peak mem (MB) |\n')
    L.append('|---|---|---|---|---|---|---|\n')
    for r in rows:
        f = lambda k: f'{r[k]:.1f}' if isinstance(r.get(k), (int, float)) else '-'
        L.append(f"| {r['group']} | {r['seed']} | {f('encoder_params')} | {f('total_params')} | "
                 f"{f('gpu_hours')} | {f('throughput_sps')} | {f('peak_mem_bytes')} |\n")
    path.write_text(''.join(L))


def _one_line_summary(metrics, recon_rows):
    """One-sentence Chinese conclusion, key numbers interpolated from results."""
    def _v(bb, obj, key):
        m = meanstd([metrics[(bb, obj)][s].get(key) for s in SEEDS])
        return m[0] if m else None

    def _pt(a, b):
        return 'NA' if (a is None or b is None) else f'{(a - b) * 100:+.1f}'

    def _recon(bb, obj):
        vs = [r['final_val_loss'] for r in recon_rows
              if r['backbone'] == bb and r['objective'] == obj
              and r.get('final_val_loss') is not None]
        m = meanstd(vs)
        return m[0] if m else None

    qr = _recon('Q', 'MAE')
    qbr = _recon('Q_bi', 'MAE')
    drop = f'{(qr - qbr) / qr * 100:.1f}' if (qr and qbr) else 'NA'

    return (
        'F1 规模（10 万样本、50 epoch）下：'
        '① 合成去噪预训练整体不优于 MAE——comm 冻结'
        f'（G {_pt(_v("G", "denoise", "comm_frozen"), _v("G", "MAE", "comm_frozen"))}pt、'
        f'Q {_pt(_v("Q", "denoise", "comm_frozen"), _v("Q", "MAE", "comm_frozen"))}pt）'
        '与微调'
        f'（G {_pt(_v("G", "denoise", "comm_ft"), _v("G", "MAE", "comm_ft"))}pt、'
        f'Q {_pt(_v("Q", "denoise", "comm_ft"), _v("Q", "MAE", "comm_ft"))}pt）均更差，'
        '仅 radar 冻结略优；'
        '② MAE+去噪联合训练是两单目标的折中，未超过最优单目标；'
        f'③ 双向 SSM 使 MAE 重构 val loss 从 {qr:.3f} 降至 {qbr:.3f}（−{drop}%），'
        '证实「单向因果 → 高重构误差」假设，但下游迁移仅在微调后略优'
        f'（comm {_pt(_v("Q_bi", "MAE", "comm_ft"), _v("Q", "MAE", "comm_ft"))}pt、'
        f'radar {_pt(_v("Q_bi", "MAE", "radar_ft"), _v("Q", "MAE", "radar_ft"))}pt），'
        '冻结读出反而更差'
        f'（comm {_pt(_v("Q_bi", "MAE", "comm_frozen"), _v("Q", "MAE", "comm_frozen"))}pt、'
        f'radar {_pt(_v("Q_bi", "MAE", "radar_frozen"), _v("Q", "MAE", "radar_frozen"))}pt）。'
    )


def _write_conclusion(path, rows, metrics, recon_rows):
    def g(bb, obj):
        return next((r for r in rows if r['backbone'] == bb and r['objective'] == obj), None)

    L = ['# Round6 目标对比 + 双向 SSM —— 结论（一页）\n',
         '> 口径：均值 ± 样本std（3 种子）。comm 冻结 = RML val OA；radar 冻结 = RadChar eval OA；',
         'comm 微调 = RML val OA；radar 微调 = RadChar eval OA。G-MAE/Q-MAE 复用 round5。\n']

    # Q1: denoise vs MAE
    L.append('\n## 问题 1：F1 规模训练下，去噪是否优于 MAE？\n')
    for bb in ('G', 'Q'):
        den = g(bb, 'denoise')
        mae = g(bb, 'MAE')
        for key, label in (('comm_frozen', '冻结 comm OA'), ('radar_frozen', '冻结 radar OA'),
                           ('comm_ft', '微调 comm OA'), ('radar_ft', '微调 radar OA')):
            d = paired_delta(metrics[(bb, 'denoise')], metrics[(bb, 'MAE')], key)
            L.append(f"- {bb}-denoise vs {bb}-MAE（{label}）：{_fmt(den[key]) if den else '-'} vs "
                     f"{_fmt(mae[key]) if mae else '-'}，Δ {_fmt(d[0]) if d else '-'}\n")

    # Q2: joint vs best single
    L.append('\n## 问题 2：MAE+去噪联合训练是否优于同骨干最优单目标？\n')
    for bb in ('G', 'Q'):
        jt = g(bb, 'joint')
        mae = g(bb, 'MAE')
        den = g(bb, 'denoise')
        for key, label in (('comm_frozen', '冻结 comm'), ('radar_frozen', '冻结 radar'),
                           ('comm_ft', '微调 comm'), ('radar_ft', '微调 radar')):
            d_m = paired_delta(metrics[(bb, 'joint')], metrics[(bb, 'MAE')], key)
            d_d = paired_delta(metrics[(bb, 'joint')], metrics[(bb, 'denoise')], key)
            L.append(f"- {bb}-joint（{label}）：{_fmt(jt[key]) if jt else '-'}；"
                     f"vs MAE Δ {_fmt(d_m[0]) if d_m else '-'}，vs denoise Δ {_fmt(d_d[0]) if d_d else '-'}\n")

    # Q3: bidirectional vs unidirectional
    L.append('\n## 问题 3：双向 SSM 是否降低 MAE 重构误差并改善下游/微调？\n')
    qb = g('Q_bi', 'MAE')
    q = g('Q', 'MAE')
    # recon loss (MAE val)
    def final_val(bb, obj):
        vs = []
        for s in SEEDS:
            r = next((x for x in recon_rows if x['backbone'] == bb and x['objective'] == obj
                      and x['seed'] == s), None)
            if r and r.get('final_val_loss') is not None:
                vs.append(r['final_val_loss'])
        return meanstd(vs)
    qb_recon = final_val('Q_bi', 'MAE')
    q_recon = final_val('Q', 'MAE')
    L.append(f"- MAE 重构 val loss：Q-MAE {_fmt(q_recon)} → Q-bi-MAE {_fmt(qb_recon)}\n")
    for key, label in (('comm_frozen', '冻结 comm OA'), ('radar_frozen', '冻结 radar OA'),
                       ('comm_ft', '微调 comm OA'), ('radar_ft', '微调 radar OA')):
        d = paired_delta(metrics[('Q_bi', 'MAE')], metrics[('Q', 'MAE')], key)
        L.append(f"- Q-bi-MAE vs Q-MAE（{label}）：{_fmt(qb[key]) if qb else '-'} vs "
                 f"{_fmt(q[key]) if q else '-'}，Δ {_fmt(d[0]) if d else '-'}\n")
    L.append('- （注：双向是否改善需看 Δ 的方向与幅度，正值为 Q-bi 更优。）\n')

    # Q4: bidirectional vs unidirectional, denoise & joint
    L.append('\n## 问题 4（补充）：双向 SSM 在去噪 / 联合目标上是否也优于单向 Q？\n')
    for obj in ('denoise', 'joint'):
        qb = g('Q_bi', obj)
        q = g('Q', obj)
        if qb is None and q is None:
            continue
        L.append(f'\n### {obj} 目标\n')
        for key, label in (('comm_frozen', '冻结 comm OA'), ('radar_frozen', '冻结 radar OA'),
                           ('comm_ft', '微调 comm OA'), ('radar_ft', '微调 radar OA')):
            d = paired_delta(metrics[('Q_bi', obj)], metrics[('Q', obj)], key)
            L.append(f"- Q-bi-{obj} vs Q-{obj}（{label}）：{_fmt(qb[key]) if qb else '-'} vs "
                     f"{_fmt(q[key]) if q else '-'}，Δ {_fmt(d[0]) if d else '-'}\n")

    # 一句话
    L.append('\n## 一句话总结\n')
    L.append('- ' + _one_line_summary(metrics, recon_rows) + '\n')
    path.write_text(''.join(L))


def _write_ppt(path, rows):
    L = ['# Round6 PPT 表（9 行 × 4 列）\n', '（可直接复制到幻灯片）\n\n']
    L.append('| 组 | comm 冻结 OA | radar 冻结 OA | comm 微调 OA | radar 微调 OA |\n')
    L.append('|---|---|---|---|---|\n')
    for r in rows:
        L.append(f"| {r['group']} | {_fmt(r['comm_frozen'])} | {_fmt(r['radar_frozen'])} | "
                 f"{_fmt(r['comm_ft'])} | {_fmt(r['radar_ft'])} |\n")
    path.write_text(''.join(L))


def _write_commands(path):
    R6s = R6 / 'scripts'
    L = ['# Round6 运行命令\n', '## 预训练（21 run）\n',
         '```bash\n'
         '# 串行（canonical，GPU1）\n'
         'tmux new-session -d -s r6_pretrain \\\n'
         '  "bash round6_objective_bidirectional/scripts/run_pretrain.sh"\n'
         '# Q_bi denoise/joint 补充 6 run，3 卡并行\n'
         'tmux new-session -d -s r6_pretrain_qbi_extra \\\n'
         '  "bash round6_objective_bidirectional/scripts/run_pretrain_qbi_extra.sh"\n'
         '```\n',
         '## 下游（readout + finetune，3 卡）\n',
         '```bash\n'
         'tmux new-session -d -s r6_downstream \\\n'
         '  "bash round6_objective_bidirectional/scripts/run_downstream.sh"\n'
         '```\n',
         '## 单条示例\n',
         '```bash\n'
         '# 预训练 Q_bi-denoise seed17\n'
         'python round6_objective_bidirectional/scripts/pretrain_r6.py \\\n'
         '  --config round6_objective_bidirectional/configs/pretrain.json \\\n'
         '  --manifest round3_scale_transfer/configs/manifest_f1.json \\\n'
         '  --backbone Q_bi --task denoise --out round6_objective_bidirectional/runs/pretrain/f1/Q_bi_denoise_seed17 \\\n'
         '  --shared-init-dir round6_objective_bidirectional/runs/pretrain/f1/shared_init \\\n'
         '  --round5-shared-init-dir round5_epoch_scale/runs/pretrain/f1/shared_init \\\n'
         '  --corpus f1 --epochs 50 --seed 17 --device cuda:0\n'
         '```\n']
    path.write_text(''.join(L))


if __name__ == '__main__':
    main()
