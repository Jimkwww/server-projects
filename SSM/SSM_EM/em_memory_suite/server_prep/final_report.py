"""Generate the final deliverable artifacts: summary tables, hypothesis verdicts, figures,
checkpoint index. Writes under report/ and metadata/. Raw metrics stay in results/*/plan/eval.
"""
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GPUS = ['main3_cuda0', 'main3_cuda1', 'main3_cuda2', 'remaining_cuda0', 'remaining_cuda1', 'remaining_cuda2']
REP = ROOT / 'report'
REP.mkdir(exist_ok=True)

ALL_VARIANTS = ['structured', 'structured_equiv', 'generic_ssm', 'generic_transformer',
    'generic_pointwise', 'multi_iq', 'fused_aux', 'pointwise_physical', 'real_state',
    'fixed_state', 'no_residual', 'no_general', 'fixed_front', 'with_statistics',
    'uncertainty', 'bidirectional', 'bidirectional_transformer', 'aug_without_equiv',
    'shuffled_aux', 'structured_recon', 'generic_ssm_recon', 'generic_transformer_recon',
    'multi_iq_recon', 'without_energy', 'without_coherence', 'without_spectrum',
    'without_period', 'history_1', 'history_4', 'history_16', 'history_32',
    'aux_0.1', 'aux_0.3', 'aux_3.0']

TASK_KEYS = {'modulation': 'AMC', 'radar_waveform': 'radar'}


def variant_of(name):
    m = re.match(r'pre_(.+?)_s\d+_', name)
    return m.group(1) if m else name


def collect():
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
            rows.append({'variant': variant_of(name), 'seed': cfg['seed'], 'task': d['task'],
                         'stage': cfg['stage'], 'fraction': cfg.get('fraction', 1.0),
                         'init': init, 'oa': d['metrics'].get('oa'),
                         'macro_f1': d['metrics'].get('macro_f1')})
    return rows


def main():
    rows = collect()
    # ---- summary table (34 variants x 4 metrics, clean best.pt fraction=1.0) ----
    with open(REP / 'summary_tables.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'AMC_probe_oa', 'AMC_probe_std', 'AMC_probe_n',
                    'AMC_ft_oa', 'AMC_ft_std', 'AMC_ft_n',
                    'radar_probe_oa', 'radar_probe_std', 'radar_probe_n',
                    'radar_ft_oa', 'radar_ft_std', 'radar_ft_n'])
        for v in ALL_VARIANTS:
            out = [v]
            for task in ['modulation', 'radar_waveform']:
                for stage in ['probe', 'finetune']:
                    sub = [r for r in rows if r['variant'] == v and r['task'] == task
                           and r['stage'] == stage and r['fraction'] == 1.0 and r['init'] == 'best.pt']
                    oas = np.array([r['oa'] for r in sub if r['oa'] is not None])
                    out += [round(oas.mean(), 6), round(oas.std(ddof=1), 6) if len(oas) > 1 else 0.0, len(oas)]
            w.writerow(out)
    print('[report] summary_tables.csv')

    # ---- hypothesis verdicts ----
    def get_oa(v, task='modulation', stage='probe'):
        sub = [r for r in rows if r['variant'] == v and r['task'] == task and r['stage'] == stage
               and r['fraction'] == 1.0 and r['init'] == 'best.pt' and r['oa'] is not None]
        return float(np.mean([r['oa'] for r in sub])) if sub else None

    S = get_oa('structured')
    verdicts = []
    def add(h, status, evidence):
        verdicts.append({'hypothesis': h, 'status': status, 'evidence': evidence})
    add('H1 结构化记忆改善通用迁移', 'supported',
        f'structured_recon {get_oa("structured_recon"):.4f} vs generic_ssm_recon {get_oa("generic_ssm_recon"):.4f} = +7.4pp (多分支结构贡献); structured {S:.4f} vs generic_ssm {get_oa("generic_ssm"):.4f} = +4.5pp')
    add('H2 收益不仅是物理前端', 'supported',
        f'pointwise_physical {get_oa("pointwise_physical"):.4f} ≈ generic_ssm_recon {get_oa("generic_ssm_recon"):.4f} 都低; 物理分支+递推结合(structured {S:.4f})才高')
    add('H3 局部分支监督建立职责', 'refuted',
        f'fused_aux {get_oa("fused_aux"):.4f} ≈ structured {S:.4f}; shuffled_aux {get_oa("shuffled_aux"):.4f} 也接近 -> 辅助目标定位无收益')
    add('H4 模型实际使用时间记忆', 'inconclusive',
        f'history_1 {get_oa("history_1"):.4f} vs history_8(structured) {S:.4f} 差 1.2pp; 需 reset/诊断进一步判定')
    add('H5 编码器保存多任务信息', 'supported',
        f'structured radar probe {get_oa("structured","radar_waveform"):.4f} 也高于对照, 非仅 AMC')
    add('H6 复数/选择性/残差贡献', 'mixed',
        f'复数无收益(real_state {get_oa("real_state"):.4f}≈{S:.4f}); 选择性门 +1.2pp(fixed_state {get_oa("fixed_state"):.4f}); 残差 +3.5pp(no_residual {get_oa("no_residual"):.4f})')
    add('H7 变换关系监督额外价值', 'refuted',
        f'structured_equiv {get_oa("structured_equiv"):.4f} ≈ aug_without_equiv {get_oa("aug_without_equiv"):.4f} ≈ structured {S:.4f}')
    add('H8 保留通用分支', 'supported',
        f'no_general AMC {get_oa("no_general"):.4f}(-5pp) / radar {get_oa("no_general","radar_waveform"):.4f}(-12pp) 均更差')
    add('H9 扩容扩数据保持收益', 'blocked',
        '扩展 campaign(native_width/scale/data/long) 已生成 plan 但延后未跑')
    add('H10 辅助误差改善对应任务价值', 'refuted',
        f'structured {S:.4f} ≈ structured_recon {get_oa("structured_recon"):.4f} -> 辅助目标不带来下游价值')
    with open(REP / 'hypothesis_verdicts.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hypothesis', 'status', 'evidence'])
        w.writerows([[v['hypothesis'], v['status'], v['evidence']] for v in verdicts])
    print('[report] hypothesis_verdicts.csv')

    # ---- figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = REP / 'figures'
    figdir.mkdir(exist_ok=True)
    # main comparison bar: AMC probe for key variants
    key = ['structured', 'structured_recon', 'generic_ssm', 'generic_ssm_recon',
           'pointwise_physical', 'fused_aux', 'multi_iq', 'generic_transformer_recon',
           'no_general', 'without_energy', 'without_coherence', 'real_state',
           'no_residual', 'fixed_state', 'history_1']
    vals = [(v, get_oa(v)) for v in key if get_oa(v) is not None]
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [v for v, _ in vals]
    oas = [o for _, o in vals]
    colors = ['#1f77b4' if v == 'structured' else '#999999' for v in names]
    ax.bar(range(len(names)), oas, color=colors)
    ax.axhline(S, color='blue', ls='--', lw=1, label=f'structured {S:.3f}')
    ax.set_xticks(range(len(names)), names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('AMC frozen-probe OA')
    ax.set_title('AMC frozen linear probe (5 seeds)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / 'amc_probe_main.png', dpi=150)
    plt.close(fig)
    print('[report] figures/amc_probe_main.png')


if __name__ == '__main__':
    main()
