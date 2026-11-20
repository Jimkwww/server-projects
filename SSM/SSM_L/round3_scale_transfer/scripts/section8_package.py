"""Section 8: package round-3 deliverables.

Weights stay on the server (only their SHA256 is recorded in the zip).
The zip bundles: reports, scripts, configs, data index lists, per-sample
predictions, acceptance JSON, a weights SHA256 manifest, and a README.
"""
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R3 = ROOT / 'round3_scale_transfer'
DELIV = R3 / 'deliverables'
ZIP = DELIV / 'SSM_L_round3_results.zip'

WEIGHT_ROOTS = [R3 / 'runs' / 'pretrain_scale', R3 / 'runs' / 'finetune_s6']

INCLUDE_DIRS = [
    R3 / 'reports',
    R3 / 'scripts',
    R3 / 'configs',
    R3 / 'data',
]
# strip these heavy dirs from reports
EXCLUDE_SUFFIXES = ('.pyc',)
EXCLUDE_NAMES = {'orchestrate_downstream.sh'}  # orchestrator is a server-side runner, not a result


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def weight_manifest() -> dict:
    entries = {}
    for root in WEIGHT_ROOTS:
        for p in sorted(root.rglob('*.pt')):
            rel = str(p.relative_to(R3))
            entries[rel] = {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
        for p in sorted(root.rglob('DONE.json')):
            rel = str(p.relative_to(R3))
            entries[rel] = {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
    return entries


def build_zip():
    DELIV.mkdir(parents=True, exist_ok=True)
    manifest = weight_manifest()
    manifest_path = DELIV / 'weights_sha256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    readme = DELIV / 'README.md'
    readme.write_text(readme_text(manifest))

    files = []
    for d in INCLUDE_DIRS:
        for p in sorted(d.rglob('*')):
            if p.is_dir():
                continue
            if p.suffix in EXCLUDE_SUFFIXES or p.name in EXCLUDE_NAMES:
                continue
            files.append(p)
    files += [manifest_path, readme]

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arc = str(p.relative_to(R3))
            z.write(p, arc)
    n_weights = sum(1 for k in manifest if k.endswith('.pt'))
    print(f'packaged {len(files)} files + {n_weights} weight SHA256 into {ZIP}')
    print(f'zip sha256: {sha256_file(ZIP)}')
    print(f'zip size: {ZIP.stat().st_size} bytes')


def readme_text(manifest: dict) -> str:
    n_pt = sum(1 for k in manifest if k.endswith('.pt'))
    total = sum(v['bytes'] for k, v in manifest.items() if k.endswith('.pt'))
    return f"""# SSM_L 第三轮实验交付包（round3_scale_transfer）

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 五个问题的答案（摘要，详见 reports/round3_report.md）

1. 53.47% 的来源：Global 主干微调能力（~46pp）+ MAE 预训练（+7.5pp）+ Q 前端（+1.8pp）。
2. 统一冻结读出排名（MAE）：Q 0.3370 > S1plain 0.3281 > G 0.3169 > G_Qlearn 0.2843；Q 前端在 Global 内为负、在 S1 内为正。
3. Global+Q 表征能迁移到雷达（G_Qlearn−G = +4.01pp），与 RML 冻结读出（为负）相反。
4. 同域扩量（F1 100k vs F0 19968，固定 7800 步）：三项下游基本持平 → 数据量不是瓶颈。
5. OOD（F2 = 50k comm + 50k OOD）非两域通吃：Global 主干微调 +2.30pp、雷达 +1.71pp 两域都涨；
   带复数分支的 Q_corr / G_Qlearn 在 RML 微调上 −2.83 / −2.70pp（复数分支向 OOD 统计过拟合）。

## 目录说明

- reports/           全部报告与 JSON 结果（round3_report.md、correction.md、section2/3/4/6_*.json、section7_acceptance.json、eligibility_v2.csv）
- scripts/           全部脚本（数据准备、预训练、探测、评估、验收、报告、打包）
- configs/           F1/F2 语料 manifest
- data/              F1/F2 索引列表、OOD 源 ID、meta（npy/json）
- reports/section6_predictions/  18 个微调模型在 clean/CFO±0.005 的逐样本预测
- weights_sha256.json  所有预训练/微调权重文件的 SHA256（权重本体留在服务器，不入包）

## 权重位置（服务器，未打包）

- 预训练：round3_scale_transfer/runs/pretrain_scale/{{f1,f2}}/{{G,G_Qlearn,Q}}_seed{{17,42,2026}}/encoder_final.pt（=last.pt 的 encoder，7800 步）
- 微调：round3_scale_transfer/runs/finetune_s6/{{f1,f2}}_{{G,G_Qlearn,Q}}_seed{{17,42,2026}}/best.pt（clean-val OA 最早 tie 选择）

共 {n_pt} 个 .pt 权重文件，总大小约 {total/1e6:.0f} MB，SHA256 见 weights_sha256.json。

## RML 测试集封存

RML cls_dev_test（8112 条）全程未用于训练/选择，仅保留在服务器；所有指标均为 val 集。

## 复现命令

预训练（F1/F2，7800 步）：scripts/pretrain_scale.py（见 scripts/run_pretrain_scale.sh）
微调：scripts/run_finetune_s6.sh <seed> <gpu>
探测：scripts/section6_probe.py（先缓存再 --readout-only）
CFO 评估：scripts/eval_finetune_s6.py
验收：scripts/section7_acceptance.py
"""


if __name__ == '__main__':
    build_zip()
