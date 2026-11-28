import json, csv, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'public_baselines_rml_radar_v1' / 'runs'
V2 = ROOT / 'runs'
MODELS = ['q', 'structured', 'emind', 'wavesfm']
TASKS = ['rml', 'radar']
SEEDS = [17, 42, 2026]

def load_v1(task, model):
    return json.loads((V1/task/model/'result.json').read_text())

def load_v2(task, model, seed, lr, ep):
    p = V2/task/model/f's{seed}_lr{lr:g}_ep{ep}'/'result.json'
    return json.loads(p.read_text()) if p.exists() else None

def rval(r):
    return {'oa': r['test_oa'], 'f1': r['test_macro_f1'], 'val_f1': r['best_val_f1']}

# ---------- 1. three-seed main table (30ep lr3e-4) ----------
main = {}
for task in TASKS:
    for m in MODELS:
        vals = []
        for s in SEEDS:
            r = load_v1(task, m) if s == 17 else load_v2(task, m, s, 3e-4, 30)
            vals.append(rval(r))
        oas = [v['oa'] for v in vals]; f1s = [v['f1'] for v in vals]
        main[(task, m)] = {'oa_mean': float(np.mean(oas)), 'oa_std': float(np.std(oas, ddof=1)),
                           'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s, ddof=1)),
                           'per_seed': [{'seed': s, 'oa': v['oa'], 'f1': v['f1']} for s, v in zip(SEEDS, vals)]}

# ---------- 2. pairwise differences per seed ----------
pairs = [('structured','q'), ('q','emind'), ('q','wavesfm'), ('structured','emind'), ('structured','wavesfm')]
pairdiff = {}
for task in TASKS:
    for (a, b) in pairs:
        diffs = []
        for s in SEEDS:
            ra = load_v1(task, a) if s==17 else load_v2(task, a, s, 3e-4, 30)
            rb = load_v1(task, b) if s==17 else load_v2(task, b, s, 3e-4, 30)
            diffs.append(ra['test_oa'] - rb['test_oa'])
        pairdiff[(task, a, b)] = {'per_seed': [round(d,4) for d in diffs],
                                  'mean': float(np.mean(diffs)), 'sign_consistent': all(d>0 for d in diffs) or all(d<0 for d in diffs)}

# ---------- 3. lr/budget 2x2 (seed17 only), val-F1 selection ----------
budget = {}
for task in TASKS:
    for m in MODELS:
        cells = {}
        for lr in (3e-4, 1e-4):
            for ep in (30, 60):
                r = load_v1(task, m) if (lr==3e-4 and ep==30) else load_v2(task, m, 17, lr, ep)
                cells[(lr, ep)] = rval(r)
        # select by val_f1 within 30-budget and within 60-budget
        sel30 = max([(3e-4,30),(1e-4,30)], key=lambda k: cells[k]['val_f1'])
        sel60 = max([(3e-4,60),(1e-4,60)], key=lambda k: cells[k]['val_f1'])
        budget[(task, m)] = {'cells': {f'lr{lr:g}_ep{ep}': cells[(lr,ep)] for lr,ep in cells},
                             'sel_30budget': {'cfg': f'lr{sel30[0]:g}_ep{sel30[1]}', 'oa': cells[sel30]['oa'], 'f1': cells[sel30]['f1'], 'val_f1': cells[sel30]['val_f1']},
                             'sel_60budget': {'cfg': f'lr{sel60[0]:g}_ep{sel60[1]}', 'oa': cells[sel60]['oa'], 'f1': cells[sel60]['f1'], 'val_f1': cells[sel60]['val_f1']}}

# ---------- 4. efficiency ----------
eff = json.loads((V2/'efficiency.json').read_text())['rows']
eff_by = {(r['task'], r['model']): r for r in eff}

out = {'main': {}, 'pairdiff': {}, 'budget': {}, 'efficiency': {}}
for k, v in main.items():
    out['main'][f'{k[0]}/{k[1]}'] = v
for k, v in pairdiff.items():
    out['pairdiff'][f'{k[0]}/{k[1]}-{k[2]}'] = v
for k, v in budget.items():
    out['budget'][f'{k[0]}/{k[1]}'] = v
for k, v in eff_by.items():
    out['efficiency'][f'{k[0]}/{k[1]}'] = v

(ROOT/'analysis.json').write_text(json.dumps(out, indent=2))

# ============ print tables ============
print('\n========== 1. 三 seed 主表 (30ep lr3e-4) ==========')
for task in TASKS:
    print(f'\n[{task}]  {"model":11s} {"OA mean±std":>14s} {"F1 mean±std":>14s}   per-seed OA')
    for m in MODELS:
        d = main[(task, m)]
        ps = ' '.join(f's{v["seed"]}={v["oa"]:.4f}' for v in d['per_seed'])
        print(f'  {m:11s} {d["oa_mean"]:.4f}±{d["oa_std"]:.4f}  {d["f1_mean"]:.4f}±{d["f1_std"]:.4f}   {ps}')

print('\n========== 2. 配对差值 (每seed test OA 差) ==========')
for task in TASKS:
    print(f'\n[{task}]')
    for (a,b) in pairs:
        d = pairdiff[(task,a,b)]
        print(f'  {a}-{b}: mean {d["mean"]:+.4f}  per_seed {d["per_seed"]}  {"一致" if d["sign_consistent"] else "方向不一致"}')

print('\n========== 3. lr/预算 2x2 (seed17, val-F1 选) ==========')
for task in TASKS:
    print(f'\n[{task}]  30预算(选lr) vs 60预算(选lr)  test OA/F1')
    for m in MODELS:
        d = budget[(task,m)]
        print(f'  {m:11s} 30: {d["sel_30budget"]["cfg"]:14s} OA {d["sel_30budget"]["oa"]:.4f}  60: {d["sel_60budget"]["cfg"]:14s} OA {d["sel_60budget"]["oa"]:.4f}')

print('\n========== 4. 效率 (batch1延迟 / 吞吐 / 显存) ==========')
for task in TASKS:
    print(f'\n[{task}]  {"model":11s} {"tok":>4s} {"b1 ms":>8s} {"thr/s":>7s} {"mem_b64 MB":>11s}')
    for m in MODELS:
        e = eff_by[(task,m)]
        print(f'  {m:11s} {e["tokens"]:4d} {e["batch1_ms_median"]:8.2f} {e["throughput_batch64_samples_s"]:7.0f} {e["peak_allocated_MB_batch64"]:11.0f}')

print('\n[saved] analysis.json')
