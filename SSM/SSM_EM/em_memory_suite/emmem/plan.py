"""Build an explicit dependency DAG; full matrix is finite, reproducible, no test selection."""
import argparse,copy,json
import torch
from pathlib import Path
from .features import ROLES
from .models import Encoder
from .train import dump

BASE=dict(patch=8,dim=64,states=32,depth=3,branched=True,physical_input=True,roles=list(ROLES),
          kind='ssm',aux_route='local',residual=True,selective=True,real=False,scan='parallel')

def variants():
    v={}
    def add(name,model=None,**train):v[name]=dict(model={**BASE,**(model or {})},aux_weight=1.,equiv_weight=0.,history=8,**train)
    add('structured')
    v['structured_equiv']=copy.deepcopy(v['structured']);v['structured_equiv']['equiv_weight']=.2;v['structured_equiv']['paired_augmentation']=True
    add('generic_ssm',dict(branched=False,physical_input=False,aux_route='fused'))
    add('generic_transformer',dict(branched=False,physical_input=False,aux_route='fused',kind='transformer'))
    add('generic_pointwise',dict(branched=False,physical_input=False,aux_route='fused',kind='pointwise'))
    add('multi_iq',dict(physical_input=False))
    add('fused_aux',dict(aux_route='fused'))
    add('pointwise_physical',dict(kind='pointwise'))
    add('real_state',dict(real=True))
    add('fixed_state',dict(selective=False))
    add('no_residual',dict(residual=False))
    add('no_general',dict(general=False))
    add('fixed_front',dict(freeze_front=True))
    add('with_statistics',dict(roles=list(ROLES)+['statistics']))
    add('uncertainty',dict(uncertainty=True))
    add('bidirectional',dict(bidirectional=True))
    add('bidirectional_transformer',dict(branched=False,physical_input=False,aux_route='fused',kind='transformer',bidirectional=True))
    add('aug_without_equiv',paired_augmentation=True)
    add('shuffled_aux',shuffle_targets=True)
    for base in ['structured','generic_ssm','generic_transformer','multi_iq']:
        v[base+'_recon']=copy.deepcopy(v[base]);v[base+'_recon']['aux_weight']=0.
    for role in ROLES:
        add('without_'+role,dict(roles=[r for r in ROLES if r!=role]))
    for h in [1,4,16,32]:
        v['history_'+str(h)]=copy.deepcopy(v['structured']);v['history_'+str(h)]['history']=h
    for w in [.1,.3,3.]:
        name='aux_'+str(w);v[name]=copy.deepcopy(v['structured']);v[name]['aux_weight']=w
    return v

def parameter_count(cfg):
    with torch.device('meta'):
        return sum(p.numel() for p in Encoder(cfg).parameters())

def match_parameters(cfg,target):
    # Monotone parameter count: binary search width grid, then inspect neighbours.
    grid=list(range(16,1025,4 if cfg.get('kind','ssm')=='transformer' else 1));lo,hi=0,len(grid)-1;cache={}
    def candidate(i):
        if i not in cache:
            d=grid[i];c={**cfg,'dim':d,'states':max(8,d//2)}
            count=parameter_count(c);cache[i]=(abs(count-target)/target,c,count)
        return cache[i]
    while lo<hi:
        mid=(lo+hi)//2
        if candidate(mid)[2]<target:lo=mid+1
        else:hi=mid
    return min((candidate(i) for i in range(max(0,lo-2),min(len(grid),lo+3))),key=lambda t:t[0])

def build(registry,out,profile='full'):
    reg=json.loads(Path(registry).read_text());out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    pre=[str(Path(p).resolve()) for p in reg['pretrain']]
    if not pre:raise ValueError('No pretraining sources configured')
    seeds=reg.get('seeds',[17,42,2026]);vs=variants()
    if profile=='smoke':
        vs={k:vs[k] for k in ['structured','generic_ssm_recon','pointwise_physical']};seeds=[17]
    if reg.get('variants'):
        vs={name:vs[name] for name in reg['variants']}
    overrides=reg.get('model_overrides',{})
    for v in vs.values():v['model'].update(overrides)
    target=parameter_count({**BASE,**overrides});jobs=[];counts={}
    def job(jid,module,args,deps=()):jobs.append(dict(id=jid,module=module,args=args,deps=list(deps)))
    for name,v in vs.items():
        # Main comparison approximately parameter matched, feature dimensions allowed to differ.
        if reg.get('parameter_mode','matched')=='matched':
            err,mc,count=match_parameters(v['model'],target)
        else:
            mc=v['model'];count=parameter_count(mc);err=abs(count-target)/target
        v['model']=mc
        counts[name]=dict(params=count,target=target,relative_gap=err)
        if err>.05 and reg.get('parameter_mode','matched')=='matched':raise ValueError(f'Parameter match >5%: {name}: {err}')
        core=['structured','generic_ssm','generic_ssm_recon','generic_transformer_recon','pointwise_physical','fused_aux','structured_equiv','aug_without_equiv']
        runseeds=list(dict.fromkeys(seeds+(reg.get('core_extra_seeds',[7,123]) if profile!='smoke' and name in core else [])))
        for seed in runseeds:
            pid=f'pre_{name}_s{seed}';runout=out/'runs'/pid
            c=dict(stage='pretrain',seed=seed,data=pre,model=v['model'],epochs=1 if profile=='smoke' else reg.get('pretrain_epochs',100),
                   batch_size=8 if profile=='smoke' else reg.get('batch_size',32),lr=3e-4,device=reg.get('device','cuda'),
                   require_cuda=profile!='smoke',workers=reg.get('workers',0),mask_ratio=.5,
                   aux_weight=v['aux_weight'],equiv_weight=v['equiv_weight'],history=v['history'],shuffle_targets=v.get('shuffle_targets',False),paired_augmentation=v.get('paired_augmentation',False),
                   fraction=reg.get('pretrain_fraction',1.),cpu_threads=reg.get('cpu_threads',4),grad_accum=reg.get('grad_accum',1))
            cp=out/'configs'/f'{pid}.json';dump(cp,c);job(pid,'emmem.train',['--config',str(cp),'--out',str(runout),'--resume'])
            init=str(runout/'best.pt')
            for taskentry in reg['downstream']:
                manifest=str(Path(taskentry['manifest']).resolve());task=taskentry['task'];tag=taskentry['id']
                spec=json.loads(Path(manifest).read_text())['tasks'][task]
                modes=['probe','finetune'] if profile=='smoke' else ['probe','finetune','scratch']
                fractions=[1.] if profile=='smoke' else reg.get('fractions',[.01,.1,1.])
                for stage in modes:
                    for fraction in fractions:
                        # Scratch depends on architecture, not pretrain objective: dedup by config hash in report, explicit runs here.
                        jid=f'{pid}_{tag}_{stage}_f{fraction}';jo=out/'runs'/jid
                        dc={**c,'stage':stage,'task':task,'data':[manifest],'fraction':fraction,
                            'epochs':1 if profile=='smoke' else reg.get('downstream_epochs',50)}
                        if stage!='scratch':dc['init']=init
                        dc['aux_weight']=v['aux_weight'];dc['equiv_weight']=0.
                        dp=out/'configs'/f'{jid}.json';dump(dp,dc)
                        job(jid,'emmem.train',['--config',str(dp),'--out',str(jo),'--resume'],[pid] if stage!='scratch' else [])
                        ev=f'{jid}_test';baseargs=['--checkpoint',str(jo/'best.pt'),'--manifest',manifest,'--task',task,'--device',c['device']]
                        job(ev,'emmem.evaluate',baseargs+['--out',str(out/'eval'/ev)],[jid])
                        if profile!='smoke' and name in core and fraction==1. and stage in ['probe','finetune']:
                            extras=[('lastinit',{'init':str(runout/'last.pt')})]
                            if stage=='finetune':extras += [('retain_aux_'+str(w),{'retain_aux':w}) for w in [.1,1.]]
                            for suffix,override in extras:
                                xid=jid+'_'+suffix;xc={**dc,**override};xp=out/'configs'/f'{xid}.json';xo=out/'runs'/xid
                                dump(xp,xc);job(xid,'emmem.train',['--config',str(xp),'--out',str(xo),'--resume'],[pid])
                                job(xid+'_test','emmem.evaluate',['--checkpoint',str(xo/'best.pt'),'--manifest',manifest,'--task',task,
                                    '--device',c['device'],'--out',str(out/'eval'/(xid+'_test'))],[xid])
                        if 'ood' in json.loads(Path(manifest).read_text())['splits']:
                            job(jid+'_ood','emmem.evaluate',baseargs+['--split','ood','--out',str(out/'eval'/(jid+'_ood'))],[jid])
                        # Interventions on full-data finetuning; same job set for all architecture variants.
                        if stage=='finetune' and fraction==1.:
                            for drop in [None]+(list(mc['roles'])+(['general'] if mc.get('general',True) else []) if mc.get('branched',True) else ['general']):
                                suffix='reset' if drop is None else 'drop_'+drop
                                extra=['--mode','reset'] if drop is None else ['--drop',drop]
                                job(jid+'_'+suffix,'emmem.evaluate',baseargs+extra+['--out',str(out/'eval'/(jid+'_'+suffix))],[jid])
                            if spec['kind']=='classification':
                                for kind,values in taskentry.get('perturbations',{}).items():
                                    for value in values:
                                        sid=f'{jid}_{kind}_{value}'
                                        job(sid,'emmem.evaluate',baseargs+['--kind',kind,'--value',str(value),'--out',str(out/'eval'/sid)],[jid])
                if tag==reg['downstream'][0]['id']:
                    job(pid+'_diagnose','emmem.diagnostics',['--checkpoint',init,'--manifest',manifest,'--out',str(out/'diagnostics'/f'{pid}.json'),
                        '--history',str(v['history']),'--device',c['device'],'--limit','128' if profile=='smoke' else '20000'],[pid])
            for entry in reg.get('novelty',[]):
                job(pid+'_novelty_'+entry['id'],'emmem.novelty',['--checkpoint',init,'--normal',str(Path(entry['normal']).resolve()),
                    '--anomaly',str(Path(entry['anomaly']).resolve()),'--out',str(out/'novelty'/f"{pid}_{entry['id']}.json"),'--device',c['device']],[pid])
            if seed==seeds[0]:job(pid+'_benchmark','emmem.benchmark',['--checkpoint',init,'--out',str(out/'bench'/f'{pid}.json'),'--device',c['device']]+(['--lengths','256','--batches','1','--reps','2'] if profile=='smoke' else []),[pid])
    # Scratch runs for identical architectures may repeat intentionally; report groups by architecture.
    dump(out/'parameter_counts.json',counts)
    dump(out/'plan.json',dict(profile=profile,jobs=jobs,registry=reg,reference='new S1-style implementation, not historical S1/Q'))
    return len(jobs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--out',required=True);p.add_argument('--profile',choices=['full','smoke'],default='full')
    a=p.parse_args();print(build(a.registry,a.out,a.profile))
