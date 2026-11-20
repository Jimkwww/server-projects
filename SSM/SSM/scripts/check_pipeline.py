"""Local integration verification using SYNTHETIC data; writes only scratch/qa."""
import json
import os
import subprocess
import sys
from pathlib import Path

def call(args):
    subprocess.run([sys.executable,*args],check=True,env={**os.environ,'OMP_NUM_THREADS':'1'})

root=Path('scratch/qa')
if root.exists():raise FileExistsError('Use a fresh scratch/qa or inspect the previous run first')
call(['scripts/make_synthetic.py','--output',str(root/'data')])
c=json.loads((root/'data/config.json').read_text())
(root/'SYNTHETIC_ONLY.md').write_text('Integration QA only. No real RML data, no research conclusion.\n')
c['protocol'].update(verified=True,audit_report=str((root/'SYNTHETIC_ONLY.md').resolve()),name='SYNTHETIC_QA_ONLY')
c['train']['epochs']=2
(root/'config.json').write_text(json.dumps(c,indent=2))
call(['-m','ssm_exp.audit','--manifest',str(root/'data/manifest.json'),'--config',str(root/'config.json'),'--output',str(root/'audit.json')])
call(['scripts/run_campaign.py','--config',str(root/'config.json'),'--manifest',str(root/'data/manifest.json'),
      '--output',str(root/'runs'),'--variants','G,S0,S1,S2,S2_concat','--seeds','17','--device','cpu','--workers','0'])
for v in ['G','S0','S1','S2','S2_concat']:
    run=str(root/'runs'/f'{v}_seed17')
    call(['-m','ssm_exp.eval','--run',run,'--manifest',str(root/'data/manifest.json'),
          '--device','cpu','--workers','0','--frequencies=-0.002,0,0.002','--phases=0,1.57','--save-predictions'])
    call(['-m','ssm_exp.benchmark','--run',run,'--device','cpu','--lengths','64,128',
          '--batches','1,4','--warmup','2','--repeats','3'])
call(['-m','ssm_exp.summarize','--runs',str(root/'runs'),'--output',str(root/'summary')])
expected=['runs.csv','summary.csv','paired_deltas.csv','curves.csv','efficiency.csv']
assert all((root/'summary'/name).is_file() for name in expected)
(root/'PASSED.json').write_text(json.dumps({'synthetic_only':True,'models':5,'training':True,'evaluation':True,'benchmark':True,'summary':True},indent=2))
print('SYNTHETIC integration pipeline passed')
