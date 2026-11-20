"""Sequential subprocess runner; no hidden GPU selection or shell interpolation."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--output',required=True);p.add_argument('--variants',default='G,S0,S1,S2');p.add_argument('--seeds',default='17,42,2026')
p.add_argument('--device',default='cuda');p.add_argument('--workers',default='4');p.add_argument('--augment-max',type=float,default=0.)
p.add_argument('--resume',action='store_true');a=p.parse_args()
for seed in a.seeds.split(','):
    for variant in a.variants.split(','):
        run=Path(a.output)/f'{variant}_seed{seed}'
        if (run/'DONE.json').exists():
            if not a.resume:raise FileExistsError(run)
            old=json.loads((run/'config.json').read_text()); expected=json.loads(Path(a.config).read_text())
            expected['model']['variant']=variant;expected['train']['seed']=int(seed);expected['train']['augment_cfo_max']=a.augment_max
            if old!=expected:raise ValueError('completed run config mismatch '+str(run))
            print('Completed, skipping',run,flush=True);continue
        cmd=[sys.executable,'-m','ssm_exp.engine','--config',a.config,'--manifest',a.manifest,'--out',str(run),
             '--variant',variant,'--seed',seed,'--device',a.device,'--workers',a.workers,'--augment-max',str(a.augment_max)]
        if a.resume and (run/'last.pt').exists():cmd+=['--resume']
        print('Launching:',cmd,flush=True);subprocess.run(cmd,check=True)
