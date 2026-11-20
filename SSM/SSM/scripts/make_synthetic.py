"""Synthetic fixture ONLY; never a substitute for RML2018A results."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--output',default='scratch/synthetic'); a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(123); count=240; length=64; classes=4
labels=np.arange(count)%classes; t=np.arange(length)
freq=(labels+1)/16
z=np.exp(2j*np.pi*freq[:,None]*t+1j*rng.uniform(-np.pi,np.pi,(count,1)))
z+=.1*(rng.normal(size=z.shape)+1j*rng.normal(size=z.shape))
np.save(out/'x.npy',np.stack((z.real,z.imag),-1).astype('float32'))
np.save(out/'y.npy',labels); np.save(out/'snr.npy',np.where(np.arange(count)%3,10.,-4.))
for split,start,end in [('train',0,160),('val',160,200),('test',200,240)]:np.save(out/(split+'.npy'),np.arange(start,end))
m={'format':'npy','x':'x.npy','y':'y.npy','snr':'snr.npy','layout':'NLC','label_format':'integer',
   'normalization':'none','classes':classes,'splits':{s:s+'.npy' for s in ('train','val','test')}}
(out/'manifest.json').write_text(json.dumps(m,indent=2))
c=json.loads(Path('configs/base.json').read_text()); c['model'].update(dim=16,depth=2,state=12,selector_hidden=8,input_length=64,classes=classes)
c['train'].update(batch_size=16,epochs=3,physical_fit_samples=80,diagnostic_every=5)
(out/'config.json').write_text(json.dumps(c,indent=2))
print(out)
