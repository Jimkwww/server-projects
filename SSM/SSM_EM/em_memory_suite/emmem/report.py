"""Aggregate all outcomes, preserving failures; seed-paired comparisons in metrics units."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from .metrics import paired_bootstrap
from .train import dump

def run(root):
    root=Path(root);rows=[]
    for p in sorted((root/'eval').glob('*/metrics.json')):
        e=json.loads(p.read_text());ck=Path(e['checkpoint']);cfg=json.loads((ck.parent/'config.json').read_text())
        name=p.parent.name;pre=ck.parent.name.split('_s'+str(cfg['seed']))[0].removeprefix('pre_')
        row=dict(run=name,variant=pre,seed=cfg['seed'],task=e['task'],manifest=e['manifest'],stage=cfg['stage'],
                 fraction=cfg.get('fraction',1),init_checkpoint=Path(cfg.get('init','scratch')).name,retain_aux=cfg.get('retain_aux',0.),split=e['split'],perturbation=e['perturbation'],value=e['value'],mode=e['mode'],drop=e['drop'] or '')
        for k in ['oa','macro_f1','miou','mse','nll','ece']:
            if k in e['metrics']:row[k]=e['metrics'][k]
        rows.append(row)
    keys=sorted(set().union(*(r.keys() for r in rows))) if rows else ['run']
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    pairs=[]
    groupfields=['task','manifest','stage','fraction','split','perturbation','value','mode','drop','init_checkpoint','retain_aux']
    groups={}
    for r in rows:
        key=tuple(r[k] for k in groupfields);groups.setdefault(key,{}).setdefault(r['variant'],{})[r['seed']]=r
    for group,variants in groups.items():
        if 'structured' not in variants:continue
        for baseline in ['generic_ssm_recon','generic_ssm','multi_iq','fused_aux','pointwise_physical','structured_recon']:
            if baseline not in variants:continue
            seeds=sorted(set(variants['structured'])&set(variants[baseline]))
            for metric in ['macro_f1','oa','mse']:
                if not seeds or metric not in variants['structured'][seeds[0]]:continue
                stats=paired_bootstrap([variants['structured'][s][metric] for s in seeds],[variants[baseline][s][metric] for s in seeds])
                pairs.append(dict(group=dict(zip(groupfields,group)),baseline=baseline,metric=metric,seeds=seeds,**stats))
    dump(root/'paired_comparisons.json',pairs)
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    failures={k:v for k,v in status.items() if v.get('status')!='done'}
    text=['# 实验汇总','',f'评估记录：{len(rows)}；未完成/失败作业：{len(failures)}。',
          '', '查看 results.csv、paired_comparisons.json、diagnostics/、bench/、status.json。',
          '分类差异乘以 100 才是百分点 pp；MSE 差异越小越好。',
          '固定预注册主比较，多个辅助比较不作未经校正的显著性宣称。',
          '合成测试不能替代真实多任务证据。缺失数据/基线必须列为未验证。']
    (root/'SUMMARY.md').write_text('\n'.join(text));dump(root/'failures.json',failures)
    from .plots import plot_all
    plot_all(root,rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
