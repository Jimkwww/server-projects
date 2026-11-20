import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .model import (Classifier, ComplexSSM, BidirectionalSSM, ComplexSelectiveSSM,
                    physical_features, perturb, max_abs_norm, parameter_count,
                    build_round5_model, build_global_qcorr_model,
                    raw_iq_scalar_features, SCALAR_VARIANTS)
from .data import IQDataset, read_manifest, audit
from .utils import (seed_all, seed_worker, save_json, save_checkpoint, grouped_metrics,
                    environment, rng_state, restore_rng)


def load_config(path):
    c=json.loads(Path(path).read_text())
    required={'model','train','protocol'}
    if not required <= c.keys(): raise ValueError('incomplete config')
    if c['train']['precision']!='fp32': raise ValueError('v1 supports fp32 only')
    if c['train']['scheduler'] not in ('cosine','constant'): raise ValueError('invalid scheduler')
    if not 0 <= c['train'].get('warmup_fraction', 0) < 1: raise ValueError('invalid warmup_fraction')
    return c


def make_loader(ds,batch,workers,shuffle=False,generator=None):
    return DataLoader(ds,batch_size=batch,num_workers=workers,shuffle=shuffle,
        generator=generator,worker_init_fn=seed_worker,pin_memory=torch.cuda.is_available(),
        persistent_workers=False,drop_last=False)


@torch.no_grad()
def fit_physics(model,ds,cfg,device):
    # Fixed random training subset, independent of training seed; avoids class-ordered files.
    size=min(len(ds),cfg['train']['physical_fit_samples'])
    positions=np.random.default_rng(0).choice(len(ds),size,replace=False).tolist()
    loader=make_loader(Subset(ds,positions),cfg['train']['batch_size'],0)
    total=torch.zeros(3,dtype=torch.float64,device=device); square=total.clone(); n=0
    for x,_,_,_ in loader:
        x=x.to(device)
        # Describe the IQ the model actually receives (max_abs-normalized), regardless
        # of whether the manifest already normalized (idempotent) or provides raw IQ.
        p=physical_features(max_abs_norm(x),cfg['model']['patch']).reshape(-1,3).double()
        total+=p.sum(0); square+=p.square().sum(0); n+=len(p)
    mean=total/n; std=(square/n-mean.square()).clamp_min(1e-6).sqrt()
    model.phys_mean.copy_(mean.float()); model.phys_std.copy_(std.float())
    return {'samples':size,'tokens':n,'mean':mean.tolist(),'std':std.tolist(),
            'ids_sha256':hashlib.sha256(ds.ids[positions].tobytes()).hexdigest()}


@torch.no_grad()
def fit_aux(model,ds,cfg,device):
    """Fit the frozen 16-dim aux-descriptor standardization (mean/std) on a fixed 8192-sample
    training subset (no CFO augmentation, max_abs-normalized input), using the initial branch
    parameters. Q_corr/R_corr share identical init so their initial stats match."""
    size=min(len(ds),cfg['train']['aux_fit_samples'])
    positions=np.random.default_rng(0).choice(len(ds),size,replace=False).tolist()
    loader=make_loader(Subset(ds,positions),cfg['train']['batch_size'],0)
    total=torch.zeros(16,dtype=torch.float64,device=device); square=total.clone(); n=0
    for x,_,_,_ in loader:
        x=x.to(device)
        f=model.aux_descriptor(max_abs_norm(x)).reshape(-1,16).double()
        total+=f.sum(0); square+=f.square().sum(0); n+=len(f)
    mean=total/n; std=(square/n-mean.square()).clamp_min(1e-6).sqrt()
    model.aux_mean.copy_(mean.float()); model.aux_std.copy_(std.float())
    return {'samples':size,'tokens':n,'mean':mean.tolist(),'std':std.tolist(),
            'ids_sha256':hashlib.sha256(ds.ids[positions].tobytes()).hexdigest()}


@torch.no_grad()
def fit_gate_scalars(model, ds, cfg, device):
    """Fit the frozen scalar standardization (c=|r|, d=|r_n-r_{n-1}|) on a fixed 8192-sample
    training subset (no CFO augmentation, max_abs-normalized input). c mean/std use every
    token; d mean/std exclude position 0 (no previous patch), which forward() then forces to
    zero after standardization. All scalar variants share this identical statistics."""
    size = min(len(ds), cfg['train']['gate_fit_samples'])
    positions = np.random.default_rng(0).choice(len(ds), size, replace=False).tolist()
    loader = make_loader(Subset(ds, positions), cfg['train']['batch_size'], 0)
    c_total = c_sq = d_total = d_sq = 0.0
    c_n = d_n = 0
    for x, _, _, _ in loader:
        x = x.to(device)
        c, d = raw_iq_scalar_features(max_abs_norm(x), cfg['model']['patch'])
        c = c.reshape(-1).double()
        d = d[:, 1:].reshape(-1).double()  # exclude first position
        c_total += c.sum().item(); c_sq += c.square().sum().item(); c_n += len(c)
        d_total += d.sum().item(); d_sq += d.square().sum().item(); d_n += len(d)
    c_mean = c_total / c_n
    c_std = math.sqrt(max(c_sq / c_n - c_mean * c_mean, 1e-12))
    d_mean = d_total / d_n
    d_std = math.sqrt(max(d_sq / d_n - d_mean * d_mean, 1e-12))
    model.gate_c_mean.copy_(torch.tensor([c_mean], device=device))
    model.gate_c_std.copy_(torch.tensor([c_std], device=device))
    model.gate_d_mean.copy_(torch.tensor([d_mean], device=device))
    model.gate_d_std.copy_(torch.tensor([d_std], device=device))
    return {'samples': size, 'c_tokens': c_n, 'd_tokens': d_n,
            'c_mean': c_mean, 'c_std': c_std, 'd_mean': d_mean, 'd_std': d_std,
            'ids_sha256': hashlib.sha256(ds.ids[positions].tobytes()).hexdigest()}


@torch.no_grad()
def frozen_branch_hash(model):
    """SHA256 of the frozen aux-branch parameters (Q_corr_frozen). Used to verify each epoch
    that the frozen recurrence (raw_alpha/theta/b) is untouched by gradients or weight decay."""
    parts = []
    for name, p in model.aux_branch.named_parameters():
        parts.append(name.encode())
        parts.append(p.detach().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(b''.join(parts)).hexdigest()


@torch.no_grad()
def evaluate(model,loader,device,frequency=0.,phase=0.,save_predictions=None,normalize=False):
    model.eval(); ys=[]; ps=[]; zs=[]; ids=[]; total_loss=0.; count=0
    for x,y,z,i in loader:
        x=x.to(device); y=y.to(device)
        if frequency!=0 or phase!=0: x=perturb(x,frequency,phase)
        if normalize: x=max_abs_norm(x)
        logits=model(x)
        if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite evaluation logits')
        total_loss+=F.cross_entropy(logits,y,reduction='sum').item(); count+=len(y)
        ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(-1).cpu().tolist())
        zs.extend(z.tolist()); ids.extend(i.tolist())
    if not count: raise ValueError('empty evaluation set')
    result=grouped_metrics(ys,ps,zs,model.cfg['classes']); result['loss']=total_loss/count
    if save_predictions:
        np.savez_compressed(save_predictions,y=np.array(ys),pred=np.array(ps),snr=np.array(zs),ids=np.array(ids))
    return result


def code_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def train(args):
    c=load_config(args.config)
    if args.variant: c['model']['variant']=args.variant
    if args.seed is not None: c['train']['seed']=args.seed
    if args.augment_max is not None: c['train']['augment_cfo_max']=args.augment_max
    if args.epochs is not None:
        if not args.smoke: raise ValueError('--epochs override only allowed with --smoke')
        c['train']['epochs']=args.epochs
    if not args.smoke and not c['protocol']['verified']:
        raise ValueError('Server must audit historical data/config and mark protocol.verified=true before full training')
    if not args.smoke and not c['protocol'].get('audit_report'):
        raise ValueError('protocol.audit_report is required')
    if not args.smoke and not Path(c['protocol']['audit_report']).is_file():
        raise ValueError('audit report does not exist')
    out=Path(args.out)
    if (out/'last.pt').exists() and not args.resume: raise FileExistsError('use --resume or a new run directory')
    if out.exists() and any(out.iterdir()) and not args.resume: raise FileExistsError('nonempty output directory')
    out.mkdir(parents=True,exist_ok=True)
    seed_all(c['train']['seed']); device=torch.device(args.device)
    m=read_manifest(args.manifest); data_audit=audit(m)
    if m['classes']!=c['model']['classes']: raise ValueError('class count mismatch')
    if c['train'].get('augment_on_raw', False) and m['normalization']!='none':
        raise ValueError('augment_on_raw requires a raw (normalization=none) manifest')
    ds=IQDataset(m,'train'); val=IQDataset(m,'val')
    if ds[0][0].shape[-1]!=c['model']['input_length']: raise ValueError('input length mismatch')
    if args.smoke:
        ds.ids=ds.ids[:min(128,len(ds))]; val.ids=val.ids[:min(128,len(val))]
    gen=torch.Generator().manual_seed(c['train']['seed'])
    aug_gen=torch.Generator(device=device).manual_seed(c['train']['seed']+100000)
    dl=make_loader(ds,c['train']['batch_size'],args.workers,True,gen)
    vl=make_loader(val,c['train']['batch_size'],args.workers)
    if c['model']['variant'] in ('R0','C0','Rb','Cb','Q_linear','Q_corr','R_corr','Q_corr_frozen',
                                 'Q_corr_gate_c','Q_corr_gate_d','Q_corr_gate_cd',
                                 'Q_corr_token_c','Q_corr_token_d','Q_corr_token_cd'):
        model=build_round5_model(c['model'],c['train']['seed']).to(device)
    elif c['model']['variant'] in ('G_Q_corr','G_Q_corr_frozen'):
        model=build_global_qcorr_model(c['model'],c['train']['seed']).to(device)
    else:
        model=Classifier(c['model']).to(device)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                         lr=c['train']['lr'],weight_decay=c['train']['weight_decay'],
                         betas=tuple(c['train']['betas']),eps=c['train']['adam_eps'])
    # Step-level warmup + cosine, matching the historical Global protocol (warmup is a
    # fraction of total optimizer steps, not epochs).
    total_steps=c['train']['epochs']*len(dl)
    warmup_steps=max(1,int(total_steps*c['train']['warmup_fraction']))
    def lr_factor(step):
        if step<warmup_steps: return (step+1)/max(1,warmup_steps)
        if c['train']['scheduler']=='constant': return 1.
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup_steps)/max(1,total_steps-warmup_steps))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    provenance={'data':data_audit,'code_sha256':code_hash(),'environment':environment(),'smoke':args.smoke}
    AUX_VARIANTS=('Q_linear','Q_corr','R_corr','Q_corr_frozen','G_Q_corr','G_Q_corr_frozen',
                  'Q_corr_gate_c','Q_corr_gate_d','Q_corr_gate_cd',
                  'Q_corr_token_c','Q_corr_token_d','Q_corr_token_cd')
    FROZEN_VARIANTS=('Q_corr_frozen','G_Q_corr_frozen')
    start=0; best=-1.; best_epoch=-1; phys=None; aux=None; frozen_hash=None; gate=None
    if args.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['config']!=c or ck['provenance']!=provenance:
            raise ValueError('resume config/data/code/environment differs; create a new run')
        # Frozen aux standardization buffers must come from the checkpoint, never silently
        # refit. Checked before load_state_dict so a missing buffer raises a clear error.
        if c['model']['variant'] in AUX_VARIANTS:
            for nm in ('aux_mean','aux_std'):
                if nm not in ck['model']:
                    raise ValueError(f'resume: checkpoint lacks frozen buffer {nm}; refusing to silently refit')
            aux=ck.get('aux')
            if aux is None:
                raise ValueError('resume: checkpoint lacks aux stats; refusing to silently refit')
        if c['model']['variant'] in SCALAR_VARIANTS:
            for nm in ('gate_c_mean','gate_c_std','gate_d_mean','gate_d_std'):
                if nm not in ck['model']:
                    raise ValueError(f'resume: checkpoint lacks frozen buffer {nm}; refusing to silently refit')
            gate=ck.get('gate_scalars')
            if gate is None:
                raise ValueError('resume: checkpoint lacks gate scalar stats; refusing to silently refit')
        if c['model']['variant'] in FROZEN_VARIANTS:
            frozen_hash=ck.get('frozen_hash')
            if frozen_hash is None:
                raise ValueError('resume: checkpoint lacks frozen branch hash')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        start=ck['epoch']+1; best=ck['best']; best_epoch=ck['best_epoch']; phys=ck['physics']
        restore_rng(ck['rng'],gen)
        aug_gen.set_state(ck['augment_rng'])
    else:
        if c['model']['variant'] in ('S2','S2_concat','S1_energy','S1_corr','G_full','G_corr'):
            phys=fit_physics(model,ds,c,device)
        if c['model']['variant'] in AUX_VARIANTS:
            aux=fit_aux(model,ds,c,device)
        if c['model']['variant'] in SCALAR_VARIANTS:
            gate=fit_gate_scalars(model,ds,c,device)
        if c['model']['variant'] in FROZEN_VARIANTS:
            frozen_hash=frozen_branch_hash(model)
    save_json(out/'config.json',c); save_json(out/'provenance.json',provenance)
    trainable_params = sum(p.numel()*(2 if p.is_complex() else 1)
                           for p in model.parameters() if p.requires_grad)
    save_json(out/'model_info.json',{'real_scalar_parameters':parameter_count(model),
                                     'trainable_real_scalar_parameters':trainable_params,
                                     'physics':phys,'aux':aux,'gate_scalars':gate,'frozen_hash':frozen_hash})
    # In interrupted runs discard uncommitted history lines beyond last checkpoint.
    history=out/'history.jsonl'
    if args.resume and history.exists():
        lines=[json.loads(x) for x in history.read_text().splitlines() if x.strip()]
        history.write_text(''.join(json.dumps(x)+'\n' for x in lines if x['epoch']<start))
    for epoch in range(start,c['train']['epochs']):
        model.train(); total=0.; n=0; correct=0; max_grad=0.; diagnostics=[]; t0=time.perf_counter()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        for batch,(x,y,_,_) in enumerate(dl):
            x=x.to(device); y=y.to(device)
            extent=c['train']['augment_cfo_max']
            if c['train'].get('augment_on_raw', False):
                # Order B: perturb raw IQ, then max_abs normalize; model describes
                # the normalized IQ it actually receives.
                if extent>0: x=perturb(x,(2*torch.rand(len(x),device=device,generator=aug_gen)-1)*extent)
                x=max_abs_norm(x)
            elif extent>0:
                # Legacy order A: manifest already normalized; perturb normalized IQ.
                x=perturb(x,(2*torch.rand(len(x),device=device,generator=aug_gen)-1)*extent)
            monitor=batch % c['train']['diagnostic_every']==0
            for module in model.modules():
                if isinstance(module,(ComplexSSM,BidirectionalSSM,ComplexSelectiveSSM)): module.collect_stats=monitor
            opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['train']['grad_clip'],error_if_nonfinite=True)
            max_grad=max(max_grad,float(norm)); opt.step(); scheduler.step()
            total+=loss.item()*len(y); n+=len(y); correct+=(logits.argmax(-1)==y).sum().item()
            if monitor:
                diagnostics.append({'batch':batch,'layers':[module.last_stats for module in model.modules() if isinstance(module,(ComplexSSM,BidirectionalSSM,ComplexSelectiveSSM))]})
        for module in model.modules():
            if isinstance(module,(ComplexSSM,BidirectionalSSM,ComplexSelectiveSSM)): module.collect_stats=False
        train_seconds=time.perf_counter()-t0
        scores=evaluate(model,vl,device,normalize=(m['normalization']=='none'))
        improved=scores['overall']['oa']>best
        if improved: best=scores['overall']['oa']; best_epoch=epoch
        row={'epoch':epoch,'lr':opt.param_groups[0]['lr'],'train_loss':total/n,'train_oa':correct/n,
             'val':scores,'train_seconds':train_seconds,'max_grad_norm_before_clip':max_grad,
             'peak_training_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
             'step':scheduler.last_epoch+1,
             'diagnostics':diagnostics}
        if c['model']['variant'] in FROZEN_VARIANTS and frozen_branch_hash(model)!=frozen_hash:
            raise RuntimeError('frozen aux branch parameters changed during training')
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'config':c,'provenance':provenance,'epoch':epoch,'best':best,'best_epoch':best_epoch,
            'rng':rng_state(gen),'augment_rng':aug_gen.get_state(),'physics':phys,
            'aux':aux,'gate_scalars':gate,'frozen_hash':frozen_hash}
        save_checkpoint(out/'last.pt',ck)
        if improved: save_checkpoint(out/'best.pt',ck)
        with history.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'epoch':epoch,'train_loss':total/n,'val_oa':scores['overall']['oa'],
                          'val_f1':scores['overall']['macro_f1'],'best_epoch':best_epoch}),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model'])
    result=evaluate(model,vl,device,save_predictions=out/'val_predictions.npz',
                    normalize=(m['normalization']=='none'))
    save_json(out/'best_val.json',{'best_epoch':ck['epoch'],'selection':'val OA, earliest tie',
                                 'metrics':result,'smoke':args.smoke})
    save_json(out/'DONE.json',{'epochs':c['train']['epochs'],'best_epoch':ck['epoch'],'smoke':args.smoke})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    p.add_argument('--variant',choices=['G','S0','S1','S2','S2_concat','S1_energy','S1_corr','G_full','G_corr','A64','A128','B_front','C_bi_front','D_complex','R0','C0','Rb','Cb','Q_linear','Q_corr','R_corr','Q_corr_frozen','G_Q_corr','G_Q_corr_frozen','Q_corr_gate_c','Q_corr_gate_d','Q_corr_gate_cd','Q_corr_token_c','Q_corr_token_d','Q_corr_token_cd']); p.add_argument('--seed',type=int)
    p.add_argument('--device',default='cuda'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--augment-max',type=float); p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true'); p.add_argument('--epochs',type=int)
    args=p.parse_args()
    if args.augment_max is not None and not 0<=args.augment_max<.5: p.error('invalid CFO range')
    train(args)

if __name__=='__main__': main()
