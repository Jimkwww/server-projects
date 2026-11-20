"""Single-device reference trainer. Launch independent seeds on independent GPUs.
No automatic DDP or mixed precision: complex-state FP32 baseline is intentional.
"""
import argparse,hashlib,json,os,platform,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from .data import dataset,IQDataset
from .features import descriptors,patches,transform
from .models import Encoder,TaskModel
from .metrics import classification,regression

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(tmp,path)

def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def config_hash(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()

def pre_loss(model,x,c):
    n=x.shape[1]//model.patch
    mask=torch.rand(x.shape[0],n,device=x.device)<c.get('mask_ratio',.5)
    # Guarantee at least one hidden patch per example without using clean values.
    mask[:,0]=True
    o=model(x,mask=mask)
    tgt=patches(x,model.patch)
    rec=(o['reconstruction']-tgt).square().mean(-1)
    loss=rec[mask].mean()*c.get('recon_weight',1.)
    logs={'reconstruction':float(rec[mask].mean().detach())}
    aux_weight=c.get('aux_weight',1.)
    if aux_weight:
        with torch.no_grad():truth=descriptors(x,model.patch,c.get('history',8))
        for role,pred in o['aux'].items():
            target=truth[role]
            if c.get('shuffle_targets',False):target=target[torch.randperm(x.shape[0],device=x.device)]
            if role in o['logvar']:
                lv=o['logvar'][role];val=.5*((pred-target).square()*(-lv).exp()+lv).mean()
            else:val=F.smooth_l1_loss(pred,target)
            loss=loss+aux_weight*val/len(o['aux']);logs[role]=float(val.detach())
    eq=c.get('equiv_weight',0.)
    if eq or c.get('paired_augmentation',False):
        # Matched two-view auxiliary supervision in both augmentation controls.
        # Finite-transform consistency in named observable readouts; no false latent invariance.
        kind=random.choice(['phase','gain','cfo'])
        value={'phase':.7,'gain':1.3,'cfo':.007}[kind]
        xt=transform(x,kind,value);a=model(x);b=model(xt)
        with torch.no_grad():
            ta=descriptors(x,model.patch,c.get('history',8));tb=descriptors(xt,model.patch,c.get('history',8))
        paired=sum((F.smooth_l1_loss(a['aux'][k],ta[k])+F.smooth_l1_loss(b['aux'][k],tb[k]))*.5 for k in a['aux'])/len(a['aux'])
        loss=loss+c.get('paired_view_weight',.2)*paired
        logs['paired_view_aux']=float(paired.detach())
        el=sum(F.smooth_l1_loss(b['aux'][k]-a['aux'][k],tb[k]-ta[k]) for k in a['aux'])/len(a['aux'])
        loss=loss+eq*el;logs['equivariance']=float(el.detach())
    return loss,logs

def supervised_loss(pred,y,spec):
    kind=spec['kind']
    if kind in ('classification','segmentation'):
        return F.cross_entropy(pred.reshape(-1,spec['outputs']),y.long().reshape(-1))
    scale=torch.tensor(spec.get('loss_scale',[1.]*spec['outputs']),device=pred.device)
    return (((pred-y.float())/scale)**2).mean()

@torch.no_grad()
def evaluate(model,loader,device,spec=None):
    model.eval();losses=[];ys=[];ps=[]
    # Preserve training RNG: validation mask is fixed across epochs and variants.
    py=random.getstate();npstate=np.random.get_state();rng=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_all(99173)
    try:
        for x,y,_ in loader:
            x=x.to(device);y=y.to(device)
            if spec is None:loss,_=pre_loss(model,x,model._train_config)
            else:
                pred=model(x);loss=supervised_loss(pred,y,spec)
                ys.append(y.cpu().numpy());ps.append(pred.cpu().numpy())
            losses.append((float(loss),len(x)))
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(rng)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
    result={'loss':sum(v*n for v,n in losses)/sum(n for _,n in losses)}
    if spec:
        y=np.concatenate(ys);p=np.concatenate(ps)
        result.update(regression(y,p) if spec['kind']=='regression' else classification(y,p.argmax(-1),spec['outputs']))
    return result

def run(c,output,resume=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    fingerprint=config_hash(c);seed_all(c['seed'])
    torch.set_num_threads(c.get('cpu_threads',4))
    device=torch.device(c.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    if c.get('require_cuda',False) and device.type!='cuda':raise RuntimeError('Formal runs require CUDA')
    stage=c.get('stage','pretrain');task=c.get('task');spec=None
    if stage!='pretrain':
        if len(c['data'])!=1:raise ValueError('Downstream uses one manifest; pretrain may mix sources')
        meta=json.loads(Path(c['data'][0]).read_text());spec=meta['tasks'][task]
    train=dataset(c['data'],'train',task if spec else None,fraction=c.get('fraction',1.),seed=c['seed'])
    val=dataset(c['data'],'val',task if spec else None)
    # Epoch-boundary resume replays the exact next sampler using generator state.
    generator=torch.Generator().manual_seed(c['seed'])
    kwargs=dict(batch_size=c.get('batch_size',32),num_workers=c.get('workers',0),pin_memory=device.type=='cuda')
    loader=DataLoader(train,shuffle=True,generator=generator,**kwargs)
    vl=DataLoader(val,shuffle=False,**kwargs)
    enc=Encoder(c['model'])
    init=c.get('init')
    if init:
        ck=torch.load(init,map_location='cpu',weights_only=False)
        if ck['model_config']!=c['model']:raise ValueError('Init architecture mismatch')
        enc.load_state_dict(ck['encoder'],strict=True)
    model=enc if spec is None else TaskModel(enc,spec['kind'],spec['outputs'])
    if stage=='probe':
        for p in enc.parameters():p.requires_grad=False
    model.to(device);enc._train_config=c
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c.get('lr',3e-4),weight_decay=c.get('weight_decay',.01))
    epochs=c.get('epochs',50)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    start=0;best=-float('inf');step=0
    if resume and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['fingerprint']!=fingerprint:raise ValueError('Refusing resume with changed config')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['epoch']+1;best=ck['best'];step=ck['step']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'])
        generator.set_state(ck['loader_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'last.pt').exists():raise FileExistsError('Use --resume or a new output path')
    dump(out/'config.json',c)
    dump(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
         cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
         params=sum(p.numel() for p in model.parameters()),fingerprint=fingerprint))
    for epoch in range(start,epochs):
        tic=time.perf_counter();model.train()
        if stage=='probe':enc.eval()
        total=0.;count=0
        accum=c.get('grad_accum',1)
        opt.zero_grad(set_to_none=True)
        for bi,(x,y,_) in enumerate(loader):
            x=x.to(device);y=y.to(device)
            if spec is None:loss,_=pre_loss(enc,x,c)
            else:
                loss=supervised_loss(model(x),y,spec)
                if stage!='probe' and c.get('retain_aux',0.):
                    auxloss,_=pre_loss(enc,x,{**c,'equiv_weight':0.})
                    loss=loss+c['retain_aux']*auxloss
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch={epoch} step={step}')
            window_start=(bi//accum)*accum*kwargs['batch_size']
            window_size=min(accum*kwargs['batch_size'],len(train)-window_start)
            (loss*len(x)/window_size).backward()
            if (bi+1)%accum==0 or bi+1==len(loader):
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c.get('clip_grad',1.))
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients')
                opt.step();opt.zero_grad(set_to_none=True);step+=1
            total+=float(loss.detach())*len(x);count+=len(x)
        sched.step();vm=evaluate(model,vl,device,spec)
        # Within-run checkpoint selection only. Never rank different pretraining objectives by loss.
        score=vm.get('macro_f1',-vm['loss']);improved=score>best;best=max(best,score)
        record=dict(epoch=epoch,step=step,train_loss=total/count,val=vm,seconds=time.perf_counter()-tic,lr=opt.param_groups[0]['lr'])
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        ck=dict(model=model.state_dict(),encoder=enc.state_dict(),model_config=c['model'],train_config=c,
             optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,best=best,step=step,fingerprint=fingerprint,
             python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,loader_rng=generator.get_state(),spec=spec)
        if improved:save(out/'best.pt',ck)
        save(out/'last.pt',ck)
        print(json.dumps(record),flush=True)
    dump(out/'DONE.json',dict(fingerprint=fingerprint,epochs=epochs,best_validation_score=best,stage=stage))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.resume)
