import sys, torch
sys.argv=['smoke']
import train
for task,L,nc in [('rml',1024,24),('radar',512,5)]:
    for name in ['q','structured','emind','wavesfm']:
        m,fwd,params = train.make_model(name,L,nc,'cuda:0')
        x=torch.randn(4,L,2,device='cuda:0')
        logits=fwd(x)
        npar=sum(p.numel()*(2 if p.is_complex() else 1) for p in params)
        print(f'{name:11s}/{task}: logits {tuple(logits.shape)} params {npar:,}')
        del m,params
