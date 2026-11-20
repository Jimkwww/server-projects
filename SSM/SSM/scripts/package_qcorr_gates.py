#!/usr/bin/env python
"""Build deliverables/SSM_results_qcorr_physical_gates.zip for the Q_corr physical-gate round.

Reproducibility contract (weights stay on server): the zip carries run metadata
(config/history/best_val/provenance/model_info/DONE), the 198 per-sample prediction files
of the 18 NEW runs, all tables, acceptance evidence, the source diff vs the global_qcorr
baseline, the exact commands, and weight_manifest.json (server paths + SHA256 of
best.pt/last.pt — weights themselves are NOT archived).
"""
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/user_wzk/wzk/projects/SSM/SSM')
REPORT = ROOT / 'reports' / 'qcorr_gates'
PRED = ROOT / 'predictions_qcorr_gates'
OUT_ZIP = ROOT / 'deliverables' / 'SSM_results_qcorr_physical_gates.zip'
STAGE = Path('/tmp/SSM_results_qcorr_physical_gates')

SEEDS = [17, 42, 2026]
NEW_VARIANTS = ['Q_corr_gate_c', 'Q_corr_gate_d', 'Q_corr_gate_cd',
                'Q_corr_token_c', 'Q_corr_token_d', 'Q_corr_token_cd']
NEW_RUNS = [f'{v}_seed{s}' for v in NEW_VARIANTS for s in SEEDS]
RUNS_BY_VARIANT = {
    'Q_corr': 'round5_iq', 'G': 'round3_aug', 'G_corr': 'round3_aug',
    'G_Q_corr_frozen': 'global_qcorr',
}
for v in NEW_VARIANTS:
    RUNS_BY_VARIANT[v] = 'qcorr_gates'
ALL_VARIANTS = ['Q_corr'] + NEW_VARIANTS + ['G', 'G_corr', 'G_Q_corr_frozen']
CONDS = ['clean', 'cfo0p001', 'cfom0p001', 'cfo0p002', 'cfom0p002',
         'cfo0p005', 'cfom0p005', 'cfo0p01', 'cfom0p01', 'cfo0p02', 'cfom0p02']
BASELINE_ZIP = ROOT / 'deliverables' / 'SSM_results_global_qcorr.zip'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    elif src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    (STAGE / 'README.md').write_text(README.replace('__BUILT__', datetime.now(timezone.utc).isoformat()),
                                     encoding='utf-8')

    for f in ['report.md', 'cfo_curves.png', 'aggregate.json', 'conclusion.md', 'acceptance_summary.md']:
        if (REPORT / f).exists():
            cp(REPORT / f, STAGE / f)
    tbl = STAGE / 'tables'
    for f in ['main_table.csv', 'paired_diffs.csv', 'cfo_curves.csv', 'per_snr.csv',
              'per_class_f1.csv', 'class_delta_gate_cd_vs_Q_corr.csv', 'efficiency.csv']:
        if (REPORT / f).exists():
            cp(REPORT / f, tbl / f)

    # ---- configs ----------------------------------------------------------------
    cfg_dir = STAGE / 'configs'
    cp(ROOT / 'configs' / 'resolved_qcorr_gates.json', cfg_dir / 'resolved_qcorr_gates.json')
    cp(ROOT / 'configs' / 'data_raw.json', cfg_dir / 'data_raw.json')
    for name in NEW_RUNS:
        cp(ROOT / 'runs' / 'qcorr_gates' / name / 'config.json', cfg_dir / f'{name}_config.json')

    # ---- source diff + new files ------------------------------------------------
    sd = STAGE / 'source_diff'
    import zipfile as _zf
    with _zf.ZipFile(BASELINE_ZIP) as z:
        base_map = {}
        for n in z.namelist():
            if n.endswith('source_diff/full_model.py'):
                base_map['model.py'] = z.read(n).decode()
            if n.endswith('source_diff/full_engine.py'):
                base_map['engine.py'] = z.read(n).decode()
    import difflib
    for name in ['model.py', 'engine.py']:
        (sd / 'baseline' / name).parent.mkdir(parents=True, exist_ok=True)
        (sd / 'baseline' / name).write_text(base_map[name], encoding='utf-8')
        old = base_map[name].splitlines()
        new = (ROOT / 'ssm_exp' / name).read_text().splitlines()
        (sd / f'{name}.diff').write_text(
            '\n'.join(difflib.unified_diff(old, new, fromfile=f'a/ssm_exp/{name}',
                                           tofile=f'b/ssm_exp/{name}', lineterm='')) + '\n',
            encoding='utf-8')
        cp(ROOT / 'ssm_exp' / name, sd / f'full_{name}')
    new_files = [ROOT / 'scripts' / f for f in
                 ['accept_qcorr_gates.py', 'eval_qcorr_gates.py', 'bench_qcorr_gates.py',
                  'diag_qcorr_gates.py', 'aggregate_qcorr_gates.py', 'make_report_qcorr_gates.py',
                  'package_qcorr_gates.py', 'run_qcorr_gates.sh']] + \
                [ROOT / 'tests' / f for f in ['test_qcorr_gates.py', 'test_qcorr_gates_resume.py']]
    for f in new_files:
        cp(f, sd / 'new_files' / f.relative_to(ROOT))

    # ---- env + commands ----------------------------------------------------------
    prov = json.loads((ROOT / 'runs' / 'qcorr_gates' / 'Q_corr_gate_cd_seed17' / 'provenance.json').read_text())
    env = prov['environment']
    (STAGE / 'env.md').write_text(
        '```\n' + json.dumps(env, indent=2) + '\n```\n\ncode_sha256: ' + prov['code_sha256'] + '\n',
        encoding='utf-8')
    (STAGE / 'commands.md').write_text(commands_text(), encoding='utf-8')

    # ---- acceptance --------------------------------------------------------------
    acc = STAGE / 'acceptance'
    for s in SEEDS:
        cp(REPORT / 'acceptance' / f'seed{s}.json', acc / f'seed{s}.json')
    cp(REPORT / 'acceptance' / 'check5b_resume.json', acc / 'check5b_resume.json')

    # ---- runs (metadata only) ----------------------------------------------------
    run_root = STAGE / 'runs'
    for name in NEW_RUNS:
        src = ROOT / 'runs' / 'qcorr_gates' / name
        dst = run_root / name
        for f in ['config.json', 'history.jsonl', 'best_val.json', 'provenance.json',
                  'model_info.json', 'DONE.json']:
            cp(src / f, dst / f)
        cp(ROOT / 'runs' / 'qcorr_gates' / f'{name}.log', dst / 'train.log')

    # ---- mechanism diag ----------------------------------------------------------
    if (REPORT / 'mechanism_diag.jsonl').exists():
        cp(REPORT / 'mechanism_diag.jsonl', STAGE / 'mechanism_diag.jsonl')

    # ---- predictions (198 new) ----------------------------------------------------
    for name in NEW_RUNS:
        tag = f'qcorr_gates_{name}'
        for cond in CONDS:
            src = PRED / f'{tag}_{cond}.npz'
            if src.exists():
                cp(src, STAGE / 'predictions' / f'{tag}_{cond}.npz')

    # ---- weight manifest (SHA256 only; weights stay on server) -------------------
    manifest = []
    for name in NEW_RUNS:
        for w in ['best.pt', 'last.pt']:
            p = ROOT / 'runs' / 'qcorr_gates' / name / w
            manifest.append({'run': name, 'weight': w, 'server_path': str(p),
                             'sha256': sha256(p), 'bytes': p.stat().st_size})
    (STAGE / 'weight_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # ---- zip ---------------------------------------------------------------------
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(STAGE.parent))
    n_files = sum(1 for p in STAGE.rglob('*') if p.is_file())
    n_pred = sum(1 for p in (STAGE / 'predictions').rglob('*') if p.is_file()) if (STAGE / 'predictions').exists() else 0
    print(json.dumps({'zip': str(OUT_ZIP), 'files': n_files, 'bytes': OUT_ZIP.stat().st_size,
                      'predictions': n_pred, 'weight_manifest_entries': len(manifest)}, indent=2))


def commands_text():
    new_dirs = ','.join(f'runs/qcorr_gates/{v}_seed{s}' for v in NEW_VARIANTS for s in SEEDS)
    bench_dirs = ','.join(f'runs/qcorr_gates/{v}_seed17' for v in NEW_VARIANTS)
    diag_dirs = ','.join(f'runs/qcorr_gates/{v}_seed{s}' for v in NEW_VARIANTS for s in SEEDS)
    return f'''# Q_corr physical-gate round — reproduction commands

## training (18 runs; 3 GPUs in parallel, each seed runs its 6 variants sequentially)
tmux new-session -d -s qcg 'bash scripts/run_qcorr_gates.sh'
# equivalently, per run (example):
python -m ssm_exp.engine --config configs/resolved_qcorr_gates.json \\
  --manifest configs/data_raw.json --out runs/qcorr_gates/Q_corr_gate_cd_seed17 \\
  --variant Q_corr_gate_cd --seed 17 --device cuda:0

## acceptance (5 checks x 3 seeds + resume pytest)
python scripts/accept_qcorr_gates.py
python -m pytest tests/test_qcorr_gates.py tests/test_qcorr_gates_resume.py -q

## evaluation (18 new runs x 11 CFO conditions, order B; saves per-sample predictions)
python scripts/eval_qcorr_gates.py --runs {new_dirs} --manifest configs/data_raw.json
# 4 historical baselines reused from reports/global_qcorr/eval_cases.jsonl (alignment verified)

## mechanism diagnosis (zeroed-modulation ablation + per-layer gate weights)
python scripts/diag_qcorr_gates.py --runs {diag_dirs} --manifest configs/data_raw.json

## efficiency (7 models at L=1024)
python scripts/bench_qcorr_gates.py --runs {bench_dirs}

## aggregation + report (recomputes OA/Macro-F1 from predictions)
python scripts/aggregate_qcorr_gates.py
python scripts/make_report_qcorr_gates.py

## packaging
python scripts/package_qcorr_gates.py
'''


README = '''# SSM S1 + IQ 标量物理门控（Q_corr physical-gate round）

Protocol: RML2018A_cls_dev_qcorr_gates. Built __BUILT__.

## What this round asks
在 Q_corr（S1 复对角 MIMO SSM + 8 通道复递归滤波复相关辅助分支）基础上，把由 IQ 计算的
两个标量描述子接入 S1 的写入门与历史衰减门：
- `c_n = |r_n|`（块内滞后 1 归一化复相关强度）→ 调制写入门 logit：`g = σ(z^g + w·c̃_n)`；
- `d_n = |r_n − r_{n−1}|`（相关变化，首块置 0）→ 调制衰减参数：`δ = softplus(z^δ + v·d̃_n)`；
- `w` / `v` 逐层逐状态零初始化（128/layer × 6 层 = 768/门），selector 网络保持不变。

`_token_c/_d/_cd` 是把同样标量经零初始化无偏置 Linear 一次性加入输入 token 的对照
（同信息、仅间接到达门）。基线：Q_corr、G、G_corr、G_Q_corr_frozen（历史，复用已验证对齐）。

## Layout
- `report.md` — 主表；`tables/*.csv` + `aggregate.json` — 机器可读。
- `conclusion.md` — 三项结论（clean 最优+seed 一致性 / ±0.005 最优+逐类权衡 / 门 vs token）。
- `acceptance/` — 5 项验收证据（3 seed）+ resume pytest 结果。
- `runs/` — 18 个新 run（config/history/best_val/provenance/model_info/DONE + train.log）。
- `predictions/` — 198 份逐样本预测（18 run × 11 条件）。
- `mechanism_diag.jsonl` — 逐层 w/v + 零化消融。
- `configs/`, `commands.md`, `env.md`, `source_diff/` — 复现材料。
- `weight_manifest.json` — 服务器路径 + SHA256（权重留在服务器）。

## Key numbers (params, fp32/complex64)
Q_corr=2,219,784; gate_c/d=2,220,552 (+768); gate_cd=2,221,320 (+1536);
token_c/d=2,219,960 (+176); token_cd=2,220,136 (+352)。见 `report.md` 的 OA 表。
'''

if __name__ == '__main__':
    main()
