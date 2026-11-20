"""Frozen cross-branch probes and interventions; train fit, val ridge selection, test once."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import IQDataset
from .features import descriptors,transform
from .models import Encoder
from .train import dump,seed_all

@torch.no_grad()
def collect(enc,manifest,split,device,history=8,mode='normal',limit=20000):
    hh={};tt={};headerr={};count=0
    for x,_,_ in DataLoader(IQDataset(manifest,split),batch_size=32):
        x=x.to(device);o=enc(x,mode=mode);t=descriptors(x,enc.patch,history)
        # Fixed regularly-spaced positions including late history; cap per split independent of values.
        sel=torch.arange(0,o['sequence'].shape[1],4,device=device)
        for k,v in o['branches'].items():hh.setdefault(k,[]).append(v[:,sel].reshape(-1,v.shape[-1]).cpu().numpy())
        for k,v in t.items():tt.setdefault(k,[]).append(v[:,sel].reshape(-1,v.shape[-1]).cpu().numpy())
        for k,v in o['aux'].items():headerr.setdefault(k,[]).append(((v[:,sel]-t[k][:,sel])**2).mean(-1).cpu().numpy().ravel())
        count+=len(x)*len(sel)
        if count>=limit:break
    return ({k:np.concatenate(v)[:limit] for k,v in hh.items()},
            {k:np.concatenate(v)[:limit] for k,v in tt.items()},
            {k:float(np.concatenate(v)[:limit].mean()) for k,v in headerr.items()})

def run(checkpoint,manifest,out,device='cpu',history=8,limit=20000):
    seed_all(77);torch.set_num_threads(4)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);enc=Encoder(ck['model_config'])
    enc.load_state_dict(ck['encoder']);enc.to(device).eval()
    train=collect(enc,manifest,'train',device,history,limit=limit)
    val=collect(enc,manifest,'val',device,history,limit=limit)
    test=collect(enc,manifest,'test',device,history,limit=limit)
    reset=collect(enc,manifest,'test',device,history,mode='reset',limit=limit)
    matrix={};reset_matrix={}
    for branch,h in train[0].items():
        mu=h.mean(0);std=h.std(0)+1e-4
        def design(z):return np.c_[(z-mu)/std,np.ones(len(z))]
        a=design(h);v=design(val[0][branch]);te=design(test[0][branch]);re=design(reset[0][branch])
        matrix[branch]={};reset_matrix[branch]={}
        for role,target in train[1].items():
            tm=target.mean(0);ts=target.std(0)+1e-4;y=(target-tm)/ts
            best=None
            for alpha in [.01,.1,1.,10.,100.]:
                reg=np.eye(a.shape[1])*alpha;reg[-1,-1]=0
                w=np.linalg.solve(a.T@a+reg,a.T@y)
                err=float(((v@w-(val[1][role]-tm)/ts)**2).mean())
                if best is None or err<best[0]:best=(err,w,alpha)
            w=best[1]
            matrix[branch][role]=dict(standardized_mse=float(((te@w-(test[1][role]-tm)/ts)**2).mean()),alpha=best[2])
            reset_matrix[branch][role]=float(((re@w-(reset[1][role]-tm)/ts)**2).mean())
    sensitivities={}
    x=next(iter(DataLoader(IQDataset(manifest,'test'),batch_size=16)))[0].to(device)
    with torch.no_grad():
        base=enc(x)
        for kind,value in [('phase',.7),('gain',1.3),('cfo',.007),('noise',0.),('iq_imbalance',.15)]:
            xx=transform(x,kind,value);changed=enc(xx)
            ta=descriptors(x,enc.patch,history);tb=descriptors(xx,enc.patch,history)
            sensitivities[kind]={k:dict(relative_state_change=float(((changed['branches'][k]-h)**2).mean()/(h.square().mean()+1e-8))) for k,h in base['branches'].items()}
            sensitivities[kind]['readout_transport_mse']={k:float(((changed['aux'][k]-base['aux'][k]-(tb[k]-ta[k]))**2).mean()) for k in base['aux']}
    calibration={}
    if enc.logvar:
        variances={};errors={}
        with torch.no_grad():
            for bi,(cx,_,_) in enumerate(DataLoader(IQDataset(manifest,'test'),batch_size=32)):
                cx=cx.to(device);co=enc(cx);ct=descriptors(cx,enc.patch,history)
                for k,lv in co['logvar'].items():
                    variances.setdefault(k,[]).append(lv.exp().cpu().numpy().ravel())
                    errors.setdefault(k,[]).append((co['aux'][k]-ct[k]).cpu().numpy().ravel())
                if (bi+1)*32>=limit:break
        for k in variances:
            vv=np.concatenate(variances[k]);ee=np.concatenate(errors[k]);order=np.argsort(vv)
            bins=[dict(predicted_variance=float(vv[ii].mean()),observed_mse=float((ee[ii]**2).mean()),n=len(ii))
                  for ii in np.array_split(order,10) if len(ii)]
            calibration[k]=dict(bins=bins,coverage_1sigma=float((abs(ee)<=np.sqrt(vv)).mean()),
                                coverage_2sigma=float((abs(ee)<=2*np.sqrt(vv)).mean()),
                                gaussian_nll=float((.5*(np.log(vv)+ee**2/vv)).mean()))
    result=dict(calibration=calibration,probe_matrix=matrix,reset_with_same_probes=reset_matrix,aux_head_test_mse=test[2],
         aux_head_reset_mse=reset[2],sensitivity=sensitivities,
         warning='Readout decodability is not exclusive semantic disentanglement. Reset removes learned recurrence, not causal descriptor windows. Do not treat low off-diagonal probe scores as mandatory.')
    dump(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--history',type=int,default=8);p.add_argument('--limit',type=int,default=20000)
    a=p.parse_args();run(**vars(a))
