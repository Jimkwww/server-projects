import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .engine import load_config
from .model import Classifier,parameter_count
from .utils import seed_all,save_json,environment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--lengths',default='1024,2048,4096,8192'); p.add_argument('--batches',default='1,64')
    p.add_argument('--warmup',type=int,default=50); p.add_argument('--repeats',type=int,default=1000)
    p.add_argument('--output',default=None)
    a=p.parse_args(); seed_all(0); device=torch.device(a.device)
    ck=torch.load(Path(a.run)/'best.pt',map_location='cpu',weights_only=False)
    if ck['provenance']['smoke']: raise ValueError('smoke checkpoint')
    model=Classifier(ck['config']['model']).to(device); model.load_state_dict(ck['model']); model.eval()
    if a.repeats<2 or a.warmup<1: raise ValueError('insufficient repetitions')
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    rows=[]
    with torch.inference_mode():
        for length in map(int,a.lengths.split(',')):
            for batch in map(int,a.batches.split(',')):
                if length%model.cfg['patch']: raise ValueError('length not divisible by patch')
                if model.cfg['position']=='learned' and length!=model.cfg['input_length']:
                    rows.append({'length':length,'batch':batch,'status':'unsupported learned positional length'}); continue
                try:
                    x=torch.randn(batch,2,length,device=device)
                    for _ in range(a.warmup): model(x)
                    sync()
                    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                    times=[]
                    for _ in range(a.repeats):
                        sync(); start=time.perf_counter(); model(x); sync()
                        times.append((time.perf_counter()-start)*1000)
                    # Separate throughput without per-iteration synchronization.
                    sync(); start=time.perf_counter()
                    for _ in range(a.repeats): model(x)
                    sync(); elapsed=time.perf_counter()-start
                    rows.append({'length':length,'batch':batch,'status':'ok','latency_ms_mean':float(np.mean(times)),
                        'latency_ms_p50':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),
                        'throughput_samples_s':batch*a.repeats/elapsed,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None})
                    del x
                except torch.cuda.OutOfMemoryError:
                    if 'x' in locals(): del x
                    torch.cuda.empty_cache(); rows.append({'length':length,'batch':batch,'status':'OOM'})
    output=Path(a.output) if a.output else Path(a.run)/'benchmark.json'
    if output.exists(): raise FileExistsError(output)
    save_json(output,{'environment':environment(),'parameters':parameter_count(model),'warmup':a.warmup,
        'repeats':a.repeats,'scan':model.cfg['scan'],'model':model.cfg,'rows':rows,
        'scope':'resident input; includes physical features and entire model; excludes H2D/data loading; inference only',
        'implementation':'eager PyTorch; parallel affine prefix O(N log N) work, not fused linear-work SSM'})
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
