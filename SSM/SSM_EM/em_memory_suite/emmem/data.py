"""Memmapped dataset schema, strict identity checks, deterministic sample selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,ConcatDataset

class IQDataset(Dataset):
    def __init__(self,manifest,split,task=None,fraction=1.,seed=17):
        self.path=Path(manifest).resolve();self.meta=json.loads(self.path.read_text())
        self.spec=self.meta['splits'][split];root=self.path.parent
        self.x=np.load(root/self.spec['x'],mmap_mode='r',allow_pickle=False)
        if self.x.ndim!=3 or self.x.shape[-1]!=2:raise ValueError('Expected x [N,L,2]')
        self.y=None
        if task is not None:
            if task not in self.spec.get('targets',{}):raise ValueError(f'Missing {task} in {manifest}/{split}')
            self.y=np.load(root/self.spec['targets'][task],mmap_mode='r',allow_pickle=False)
        self.idx=np.arange(len(self.x));rng=np.random.default_rng(seed)
        if fraction<1:
            if self.y is not None and self.meta['tasks'][task]['kind']=='classification':
                self.idx=np.concatenate([rng.permutation(np.flatnonzero(self.y==c))[:max(1,int(np.sum(self.y==c)*fraction))]
                                         for c in np.unique(self.y)])
            else:self.idx=rng.permutation(self.idx)[:max(1,int(len(self.idx)*fraction))]
        self.snr=np.load(root/self.spec['snr'],mmap_mode='r') if 'snr' in self.spec else None
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        j=self.idx[i];x=torch.from_numpy(np.array(self.x[j],dtype=np.float32))
        if not torch.isfinite(x).all():raise ValueError('nonfinite IQ')
        y=torch.tensor(0) if self.y is None else torch.from_numpy(np.array(self.y[j]))
        return x,y,int(j)

def dataset(paths,split,task=None,**kwargs):
    ds=[IQDataset(p,split,task,**kwargs) for p in paths]
    lengths={d.x.shape[1] for d in ds}
    if len(lengths)!=1:raise ValueError('Use equal-length dataset buckets; no silent cropping')
    return ds[0] if len(ds)==1 else ConcatDataset(ds)

def audit(paths,output,hash_samples=True):
    seen_ids={};seen_groups={};seen_wave={};issues=[];summaries=[]
    for path in paths:
        p=Path(path).resolve();m=json.loads(p.read_text());root=p.parent
        if not m.get('dataset_id') or not m.get('group_namespace'):raise ValueError('dataset_id/group_namespace required')
        if not m.get('provenance'):issues.append(f'{path}: missing provenance')
        for split,s in m['splits'].items():
            x=np.load(root/s['x'],mmap_mode='r');n=len(x)
            ids=np.load(root/s['ids'],allow_pickle=False)
            groups=np.load(root/s['groups'],allow_pickle=False)
            if len(ids)!=n or len(groups)!=n:raise ValueError('metadata length mismatch')
            if x.ndim!=3 or x.shape[-1]!=2:raise ValueError('x shape')
            for task,t in m.get('tasks',{}).items():
                if task not in s.get('targets',{}):continue
                y=np.load(root/s['targets'][task],mmap_mode='r')
                if len(y)!=n or not np.isfinite(y).all():raise ValueError('bad target')
                if t['kind']=='classification' and (y.ndim!=1 or y.min()<0 or y.max()>=t['outputs']):raise ValueError('label range')
            for i in range(n):
                key=(m['dataset_id'],str(ids[i]));group=(m['group_namespace'],str(groups[i]))
                loc=f'{m["dataset_id"]}/{split}'
                if key in seen_ids:issues.append(f'duplicate id {key}')
                seen_ids[key]=loc
                if group in seen_groups and seen_groups[group]!=loc:issues.append(f'group leakage {group}: {seen_groups[group]} -> {loc}')
                seen_groups[group]=loc
                if not np.isfinite(x[i]).all():raise ValueError('nonfinite x')
                if hash_samples:
                    h=hashlib.sha256(np.ascontiguousarray(x[i],dtype=np.float32).tobytes()).hexdigest()
                    if h in seen_wave and seen_wave[h]!=loc:issues.append(f'waveform duplicate across splits/datasets {seen_wave[h]} -> {loc}')
                    seen_wave[h]=loc
            summaries.append(dict(dataset=m['dataset_id'],split=split,n=n,length=x.shape[1]))
    result=dict(ok=not issues,issues=list(dict.fromkeys(issues)),summary=summaries,
                warning='Exact hashes do not detect cropped, resampled or transformed source duplicates; source/group provenance review is mandatory.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2))
    if issues:raise RuntimeError(f'Data audit failed; see {output}')
    return result
