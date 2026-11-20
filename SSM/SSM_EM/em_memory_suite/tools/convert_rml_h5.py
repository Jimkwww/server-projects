"""Explicit indices only: no silent random split of captures. h5py optional server dependency.
Group array must identify source capture/transmission, not merely sample row ID.
"""
import argparse,json
from pathlib import Path
import h5py
import numpy as np

def convert(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    groups=np.load(a.groups,allow_pickle=False)
    splitfiles=json.loads(Path(a.splits).read_text());allidx=[]
    with h5py.File(a.h5,'r') as f:
        x=f[a.x_key];y=f[a.y_key];snr=f[a.snr_key]
        if len(groups)!=len(x):raise ValueError('Groups length mismatch')
        manifest=dict(dataset_id=a.dataset_id,group_namespace=a.dataset_id,provenance=a.provenance,
            tasks={'modulation':{'kind':'classification','outputs':y.shape[1] if len(y.shape)>1 else int(np.max(y[:]))+1}},splits={})
        for split,file in splitfiles.items():
            idx=np.load(file,allow_pickle=False).astype('int64')
            if len(np.unique(idx))!=len(idx) or idx.min()<0 or idx.max()>=len(x):raise ValueError('Bad split indices')
            allidx.extend(idx.tolist());idx=np.sort(idx)
            xx=np.lib.format.open_memmap(out/f'{split}_x.npy',mode='w+',dtype='float32',shape=(len(idx),x.shape[1],2))
            yy=[];zz=[]
            for start in range(0,len(idx),1024):
                sub=idx[start:start+1024];xx[start:start+len(sub)]=x[sub]
                label=y[sub];yy.append(label.argmax(-1) if label.ndim>1 else label);zz.append(snr[sub].reshape(-1))
            xx.flush();del xx
            np.save(out/f'{split}_y.npy',np.concatenate(yy).astype('int64'))
            np.save(out/f'{split}_snr.npy',np.concatenate(zz))
            np.save(out/f'{split}_ids.npy',idx.astype(str));np.save(out/f'{split}_groups.npy',groups[idx].astype(str))
            manifest['splits'][split]=dict(x=f'{split}_x.npy',ids=f'{split}_ids.npy',groups=f'{split}_groups.npy',
                                         snr=f'{split}_snr.npy',targets={'modulation':f'{split}_y.npy'})
        if len(set(allidx))!=len(allidx):raise ValueError('Split overlap')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['h5','splits','groups','out','dataset-id','provenance']:p.add_argument('--'+k,required=True)
    p.add_argument('--x-key',default='X');p.add_argument('--y-key',default='Y');p.add_argument('--snr-key',default='Z')
    convert(p.parse_args())
