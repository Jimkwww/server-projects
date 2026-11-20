import argparse
import json
from pathlib import Path
import torch
from .data import read_manifest, audit, IQDataset
from .engine import make_loader,evaluate
from .model import Classifier
from .utils import seed_all,save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True); p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=4); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--frequencies',default='-0.01,-0.005,-0.002,-0.001,0,0.001,0.002,0.005,0.01')
    p.add_argument('--phases',default='0,0.7853981634,1.5707963268,3.1415926536,4.7123889804')
    p.add_argument('--output',default=None); p.add_argument('--save-predictions',action='store_true')
    a=p.parse_args(); run=Path(a.run)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    if ck['provenance']['smoke']: raise ValueError('do not evaluate smoke checkpoints as research runs')
    m=read_manifest(a.manifest)
    if audit(m)!=ck['provenance']['data']: raise ValueError('manifest/data/splits changed since training')
    seed_all(ck['config']['train']['seed']); device=torch.device(a.device)
    model=Classifier(ck['config']['model']).to(device); model.load_state_dict(ck['model'])
    loader=make_loader(IQDataset(m,a.split),a.batch_size,a.workers)
    out=Path(a.output) if a.output else run/f'eval_{a.split}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists(): raise FileExistsError('evaluation output exists; choose a new output')
    frequencies=[float(v) for v in a.frequencies.split(',')]
    if any(abs(v)>=.5 for v in frequencies): raise ValueError('CFO >= Nyquist')
    cases=[('clean',0.,0.)]+[('cfo',f,0.) for f in frequencies]+[('phase',0.,float(v)) for v in a.phases.split(',')]
    rows=[]
    for j,(kind,f,phase) in enumerate(cases):
        r=evaluate(model,loader,device,f,phase,out/f'pred_{j:03d}.npz' if a.save_predictions else None)
        rows.append({'kind':kind,'frequency_cycles_per_sample':f,'phase_radians':phase,'metrics':r})
        save_json(out/'partial.json',rows)
        print(kind,f,phase,r['overall']['oa'],flush=True)
    save_json(out/'results.json',{'split':a.split,'checkpoint_epoch':ck['epoch'],
        'checkpoint_selection':'clean validation OA','cases':rows,
        'warning':'CFO range is a candidate diagnostic grid; server must verify label validity and occupied bandwidth.'})

if __name__=='__main__': main()
