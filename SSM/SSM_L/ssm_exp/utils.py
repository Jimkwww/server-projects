import json
import os
import random
import platform
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed); random.seed(seed)


def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    os.replace(tmp,path)


def save_checkpoint(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)


def metrics(y,p,c):
    cm=np.bincount(np.asarray(y,dtype=int)*c+np.asarray(p,dtype=int),minlength=c*c).reshape(c,c)
    tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(c,dtype=float),where=den>0)
    return {'n':int(cm.sum()),'oa':float(tp.sum()/max(1,cm.sum())),
            'macro_f1':float(f1.mean()),'class_f1':f1.tolist(),'confusion':cm.tolist()}


def grouped_metrics(y,p,snr,c):
    y,p,snr=np.asarray(y),np.asarray(p),np.asarray(snr)
    out={'overall':metrics(y,p,c),'snr':{}}
    for key,mask in [('low',snr<=0),('high',snr>0)]:
        out[key]=metrics(y[mask],p[mask],c) if mask.any() else None
    for z in np.unique(snr):
        mask=snr==z; out['snr'][str(float(z))]=metrics(y[mask],p[mask],c)
    return out


def environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
            'tf32':False,'deterministic':'warn_only','precision':'fp32/complex64'}


def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'loader':generator.get_state()}


def restore_rng(r,generator):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
    if r['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(r['cuda'])
    generator.set_state(r['loader'])
