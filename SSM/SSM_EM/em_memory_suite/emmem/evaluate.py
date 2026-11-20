import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import IQDataset
from .models import Encoder,TaskModel
from .features import transform
from .metrics import classification,regression,binary_auc
from .train import dump,seed_all

@torch.no_grad()
def evaluate_checkpoint(checkpoint,manifest,task,out,split='test',kind='identity',value=0.,mode='normal',drop=None,device='cpu'):
    seed_all(887);torch.set_num_threads(4)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    spec=json.loads(Path(manifest).read_text())['tasks'][task]
    enc=Encoder(ck['model_config']);model=TaskModel(enc,spec['kind'],spec['outputs'])
    model.load_state_dict(ck['model']);model.to(device).eval()
    ds=IQDataset(manifest,split,task);ys=[];ps=[];ids=[]
    # Perturbation policy belongs to task. Classification/noise generally label preserving;
    # regression/segmentation transforms require caller's explicit target policy.
    if spec['kind']!='classification' and kind!='identity':
        raise ValueError('Regression/segmentation perturbations require target transport; use identity or implement documented policy')
    for x,y,i in DataLoader(ds,batch_size=64):
        pred=model(transform(x.to(device),kind,value),mode,drop)
        ys.append(y.numpy());ps.append(pred.cpu().numpy());ids.append(i.numpy())
    y=np.concatenate(ys);p=np.concatenate(ps);idx=np.concatenate(ids)
    metrics=regression(y,p) if spec['kind']=='regression' else classification(y,p.argmax(-1),spec['outputs'])
    if spec['kind']=='classification':
        prob=torch.tensor(p).softmax(-1).numpy();confidence=prob.max(-1);correct=(p.argmax(-1)==y)
        metrics['nll']=float(-np.log(np.maximum(prob[np.arange(len(y)),y.astype(int)],1e-12)).mean())
        metrics['ece']=sum(float(abs(confidence[m].mean()-correct[m].mean())*m.mean())
            for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:])
            if (m:=((confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi))).any())
        if ds.snr is not None:
            snr=np.asarray(ds.snr)[idx];metrics['by_snr']={str(s):classification(y[snr==s],p[snr==s].argmax(-1),spec['outputs']) for s in np.unique(snr)}
            metrics['snr_groups']={name:classification(y[m],p[m].argmax(-1),spec['outputs']) for name,m in
                                  [('le0',snr<=0),('gt0',snr>0)] if m.any()}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'predictions.npz',y=y,pred=p,indices=idx)
    dump(out/'metrics.json',dict(metrics=metrics,split=split,perturbation=kind,value=value,mode=mode,drop=drop,
                                  checkpoint=str(checkpoint),task=task,manifest=str(manifest)))
    return metrics

@torch.no_grad()
def novelty(checkpoint,normal_manifest,anomaly_manifest,out,device='cpu'):
    # Training-normal only diagonal Gaussian. Val normals calibrate 95% threshold.
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);enc=Encoder(ck['model_config'])
    enc.load_state_dict(ck['encoder']);enc.to(device).eval()
    def get(path,split):
        return np.concatenate([enc(x.to(device))['sequence'].mean(1).cpu().numpy()
                               for x,_,_ in DataLoader(IQDataset(path,split),batch_size=64)])
    a=get(normal_manifest,'train');mu=a.mean(0);var=a.var(0)+1e-4
    def score(v):return (((v-mu)**2)/var).mean(1)
    cal=score(get(normal_manifest,'val'));threshold=float(np.quantile(cal,.95))
    neg=score(get(normal_manifest,'test'));pos=score(get(anomaly_manifest,'test'))
    y=np.r_[np.zeros(len(neg)),np.ones(len(pos))];s=np.r_[neg,pos]
    result=dict(auroc=binary_auc(y,s),threshold=threshold,test_fpr=float((neg>threshold).mean()),
        test_tpr=float((pos>threshold).mean()),method='train-normal diagonal Gaussian distance',
        warning='Require source-matched normal/anomaly test data; dataset-source separation alone is not anomaly understanding.')
    dump(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--task',required=True);p.add_argument('--out',required=True);p.add_argument('--split',default='test')
    p.add_argument('--kind',default='identity');p.add_argument('--value',type=float,default=0)
    p.add_argument('--mode',default='normal',choices=['normal','reset']);p.add_argument('--drop');p.add_argument('--device',default='cpu')
    a=p.parse_args();evaluate_checkpoint(a.checkpoint,a.manifest,a.task,a.out,a.split,a.kind,a.value,a.mode,a.drop,a.device)
