"""Generate all requested campaigns in one invocation; no manual staged confirmation.
Requires completed real-data registry. Main matrix + scaling + transfer + length campaigns.
"""
import argparse,copy,json,subprocess,sys
from pathlib import Path
from emmem.train import dump
from emmem.plan import build

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--out',required=True);p.add_argument('--run',action='store_true')
    a=p.parse_args();base=json.loads(Path(a.registry).read_text());root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    campaigns=[('main',base)]
    selected=['structured','structured_equiv','generic_ssm','generic_ssm_recon','generic_transformer_recon','multi_iq','pointwise_physical']
    r=copy.deepcopy(base);r.update(variants=selected,parameter_mode='native',fractions=[1.]);campaigns.append(('native_width',r))
    for size,dim,depth in [('small',32,2),('medium',128,6),('large',256,12)]:
        r=copy.deepcopy(base);r.update(variants=selected,model_overrides=dict(dim=dim,states=dim//2,depth=depth),fractions=[1.])
        campaigns.append(('scale_'+size,r))
    for frac in [.1,.3]:
        r=copy.deepcopy(base);r.update(variants=selected,pretrain_fraction=frac,fractions=[1.]);campaigns.append(('data_'+str(frac),r))
        r=copy.deepcopy(r);r['pretrain_epochs']=round(base.get('pretrain_epochs',100)/frac);campaigns.append(('data_equal_updates_'+str(frac),r))
    r=copy.deepcopy(base);r.update(variants=selected,pretrain_epochs=300,fractions=[1.]);campaigns.append(('long_pretrain',r))
    # Explicit domain labels in registry are required; excluded-domain train data remains available for downstream FT only.
    for domain,paths in base.get('pretrain_domains',{}).items():
        r=copy.deepcopy(base);r.update(variants=selected,pretrain=paths,fractions=[1.]);campaigns.append(('pretrain_domain_'+domain,r))
    for name,override in base.get('extra_campaigns',{}).items():
        r=copy.deepcopy(base);r.update(override);campaigns.append((name,r))
    index=[]
    for name,r in campaigns:
        path=root/name;registry=path/'registry.json';dump(registry,r)
        count=build(registry,path/'plan');index.append(dict(name=name,jobs=count,plan=str(path/'plan'/'plan.json')))
    dump(root/'campaign_index.json',index)
    if a.run:
        failures=[]
        for item in index:
            cmd=[sys.executable,'-m','emmem.runner','--plan',item['plan']]
            status=subprocess.run(cmd).returncode
            subprocess.run([sys.executable,'-m','emmem.report','--root',str(Path(item['plan']).parent)],check=True)
            if status:failures.append(item['name'])
        dump(root/'campaign_completion.json',dict(failed=failures,total=len(index)))
        if failures:raise RuntimeError(f'Failed campaigns: {failures}')

if __name__=='__main__':main()
