import os
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    path=Path(path).resolve()
    m=json.loads(path.read_text())
    root=path.parent
    def absolute(p):
        return str((root/p).resolve()) if not Path(p).is_absolute() else p
    for k in ('file','x','y','snr'):
        if k in m: m[k]=absolute(m[k])
    m['splits']={k:absolute(v) for k,v in m['splits'].items()}
    m['_manifest_path']=str(path)
    if m.get('normalization') not in ('none','sample_rms','max_abs'):
        raise ValueError('manifest must specify normalization: none, sample_rms or max_abs')
    if m.get('layout') not in ('NLC','NCL'):
        raise ValueError('layout must be NLC or NCL')
    if m.get('label_format') not in ('integer','one_hot'):
        raise ValueError('label_format must be explicit')
    return m


class IQDataset(Dataset):
    def __init__(self,manifest,split):
        self.m=manifest
        self.ids=np.load(manifest['splits'][split],allow_pickle=False)
        if self.ids.ndim != 1 or self.ids.dtype.kind not in 'iu':
            raise ValueError('split must be 1D integer original row indices')
        self.handles=None

    def _open(self):
        if self.handles is None or getattr(self, '_pid', None)!=os.getpid():
            if hasattr(self, 'file'): self.file.close()
            self._pid=os.getpid()
            if self.m['format']=='h5':
                self.file=h5py.File(self.m['file'],'r')
                self.handles=[self.file[self.m.get(k+'_key',d)] for k,d in [('x','X'),('y','Y'),('snr','Z')]]
            elif self.m['format']=='npy':
                self.handles=[np.load(self.m[k],mmap_mode='r',allow_pickle=False) for k in ('x','y','snr')]
            else: raise ValueError('format must be h5 or npy')
        return self.handles

    def __getstate__(self):
        state=self.__dict__.copy()
        state['handles']=None
        state.pop('file',None)
        return state

    def __len__(self): return len(self.ids)

    def __getitem__(self,i):
        x,y,snr=self._open()
        idx=int(self.ids[i])
        a=np.array(x[idx],dtype=np.float32,copy=True)
        if self.m['layout']=='NLC': a=a.T.copy()
        if a.shape[0]!=2 or not np.isfinite(a).all(): raise ValueError('invalid IQ')
        if self.m['normalization']=='sample_rms':
            a/=max(float(np.sqrt(np.mean(np.sum(a*a,axis=0)))),1e-8)
        elif self.m['normalization']=='max_abs':
            # Historical Global: joint max absolute COMPONENT over I and Q, not per-channel
            # and not the complex-envelope magnitude.
            a/=max(float(np.abs(a).max()),1e-8)
        label=np.asarray(y[idx])
        if self.m['label_format']=='one_hot':
            if label.size!=self.m['classes'] or not np.isclose(label.sum(),1) or not np.all((label==0)|(label==1)):
                raise ValueError('invalid one-hot labels')
            label=int(label.argmax())
        else:
            value=float(label.squeeze())
            if value!=int(value): raise ValueError('fractional label')
            label=int(value)
        z=float(np.asarray(snr[idx]).squeeze())
        if not 0<=label<self.m['classes'] or not np.isfinite(z): raise ValueError('invalid label/SNR')
        return torch.from_numpy(a),label,z,idx


def audit(manifest):
    """Full split disjointness and bounds; data metadata, not a full data-content hash."""
    ds=IQDataset(manifest,'train')
    arrays=ds._open()
    count=len(arrays[0])
    if any(len(a)!=count for a in arrays): raise ValueError('array length mismatch')
    all_ids=[]; report={}
    for name,path in manifest['splits'].items():
        ids=np.load(path,allow_pickle=False)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)==0: raise ValueError('invalid split '+name)
        if ids.min()<0 or ids.max()>=count or len(np.unique(ids))!=len(ids): raise ValueError('bad index '+name)
        all_ids.append(ids)
        report[name]={'count':len(ids),'sha256':file_hash(path)}
    joined=np.concatenate(all_ids)
    if len(np.unique(joined))!=len(joined): raise ValueError('split overlap detected')
    for j in (0,len(ds)//2,len(ds)-1): ds[j]
    paths=[manifest['file']] if manifest['format']=='h5' else [manifest[k] for k in ('x','y','snr')]
    data_files=[{'path':p,'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in paths]
    return {'splits':report,'shape':[list(a.shape) for a in arrays], 'data_files':data_files,
            'manifest':manifest,'manifest_sha256':file_hash(manifest['_manifest_path'])}
