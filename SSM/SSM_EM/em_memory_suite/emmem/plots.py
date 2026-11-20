"""Publication data are in CSV/JSON; these deterministic plots are a report starting point."""
import json,re
from pathlib import Path
import numpy as np

def plot_all(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    def safe(s):return re.sub(r'[^a-zA-Z0-9_.-]+','_',str(s))
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/(safe(name)+'.png'),dpi=180);fig.savefig(out/(safe(name)+'.pdf'));plt.close(fig)
    # Primary clean test rows; exclude supplemental checkpoint / retain-aux changes.
    primary=[r for r in rows if r['split']=='test' and r['perturbation']=='identity' and r['mode']=='normal' and not r['drop']
             and r.get('init_checkpoint') in ['best.pt','scratch'] and r.get('retain_aux',0)==0]
    for task in sorted({r['task'] for r in primary}):
        for stage in ['probe','finetune','scratch']:
            rs=[r for r in primary if r['task']==task and r['stage']==stage and r['fraction']==1.]
            if not rs:continue
            metric='macro_f1' if 'macro_f1' in rs[0] else 'mse'
            names=sorted({r['variant'] for r in rs});v=[[r[metric] for r in rs if r['variant']==k] for k in names]
            fig,ax=plt.subplots(figsize=(max(8,len(names)*.35),4));ax.bar(range(len(names)),[np.mean(z) for z in v],yerr=[np.std(z,ddof=1) if len(z)>1 else 0 for z in v])
            ax.set_xticks(range(len(names)),names,rotation=80,ha='right');ax.set_ylabel(metric);ax.set_title(task+' / '+stage)
            finish(fig,task+'_'+stage+'_clean')
        for perturb in sorted({r['perturbation'] for r in rows if r['task']==task and r['perturbation']!='identity'}):
            rs=[r for r in rows if r['task']==task and r['perturbation']==perturb and r['stage']=='finetune' and r['fraction']==1.]
            if not rs:continue
            fig,ax=plt.subplots(figsize=(7,4))
            for variant in ['structured','generic_ssm','generic_ssm_recon','generic_transformer_recon']:
                vr=[r for r in rs if r['variant']==variant];xs=sorted({r['value'] for r in vr})
                if xs:ax.plot(xs,[np.mean([r['macro_f1'] for r in vr if r['value']==x]) for x in xs],marker='o',label=variant)
            ax.set_xlabel(perturb);ax.set_ylabel('macro F1');ax.legend();finish(fig,task+'_'+perturb)
    for p in (root/'diagnostics').glob('*.json'):
        d=json.loads(p.read_text());m=d.get('probe_matrix',{})
        if not m:continue
        branches=list(m);roles=list(m[branches[0]]);z=np.array([[m[b][r]['standardized_mse'] for r in roles] for b in branches])
        fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(z,aspect='auto');fig.colorbar(im,ax=ax,label='standardized MSE')
        ax.set_xticks(range(len(roles)),roles,rotation=30);ax.set_yticks(range(len(branches)),branches);finish(fig,p.stem+'_probes')
    benches=[]
    for p in (root/'bench').glob('*.json'):
        b=json.loads(p.read_text())
        for r in b['rows']:
            if r.get('length')==1024 and r.get('batch')==1 and 'median_ms' in r:benches.append((p.stem,b['params'],r['median_ms']))
    if benches:
        fig,ax=plt.subplots(figsize=(8,5))
        for name,params,ms in benches:ax.scatter(params,ms);ax.annotate(name,(params,ms),fontsize=6)
        ax.set_xlabel('total parameters');ax.set_ylabel('B1 L1024 latency (ms)');finish(fig,'parameters_latency')
