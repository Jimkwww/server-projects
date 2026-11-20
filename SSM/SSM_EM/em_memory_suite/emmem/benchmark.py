import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .models import Encoder
from .train import dump

def run(checkpoint,out,device='cpu',lengths=(256,512,1024,2048,4096,8192),batches=(1,16,64),reps=50):
    torch.set_num_threads(4);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    enc=Encoder(ck['model_config']).to(device).eval();enc.load_state_dict(ck['encoder']);results=[]
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    for length in lengths:
        for batch in batches:
            try:
                x=torch.randn(batch,length,2,device=device)
                with torch.no_grad():
                    for _ in range(10):enc(x)
                    sync()
                    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                    times=[]
                    for _ in range(reps):
                        sync();tic=time.perf_counter();enc(x);sync();times.append(time.perf_counter()-tic)
                results.append(dict(length=length,batch=batch,median_ms=1000*float(np.median(times)),p95_ms=1000*float(np.quantile(times,.95)),
                    samples_per_second=batch/float(np.median(times)),peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None))
            except torch.cuda.OutOfMemoryError:
                results.append(dict(length=length,batch=batch,error='CUDA_OOM'));torch.cuda.empty_cache()
    dump(out,dict(params=sum(p.numel() for p in enc.parameters()),device=device,rows=results,
        precision='FP32/complex64',includes='descriptor extraction + encoder + auxiliary heads; excludes loading and H2D',
        implementation='reference unfused log-depth scan; do not infer production SSM vs Transformer speed from this alone'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--lengths',default='256,512,1024,2048,4096,8192');p.add_argument('--batches',default='1,16,64');p.add_argument('--reps',type=int,default=50)
    a=p.parse_args();a.lengths=tuple(map(int,a.lengths.split(',')));a.batches=tuple(map(int,a.batches.split(',')));run(**vars(a))
