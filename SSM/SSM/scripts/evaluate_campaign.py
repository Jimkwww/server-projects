"""Evaluate completed runs; requires an explicit frozen CFO grid."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--manifest',required=True)
p.add_argument('--frequencies',required=True);p.add_argument('--split',choices=['val','test'],default='val')
p.add_argument('--device',default='cuda');p.add_argument('--workers',default='4');p.add_argument('--batch-size',default='128')
p.add_argument('--save-predictions',action='store_true');a=p.parse_args()
for done in sorted(Path(a.runs).rglob('DONE.json')):
    if json.loads(done.read_text())['smoke']:continue
    run=done.parent
    if (run/f'eval_{a.split}'/'results.json').exists():
        raise FileExistsError('Existing evaluation: do not silently mix grids: '+str(run))
    cmd=[sys.executable,'-m','ssm_exp.eval','--run',str(run),'--manifest',a.manifest,'--split',a.split,
         '--frequencies='+a.frequencies,'--device',a.device,'--workers',a.workers,'--batch-size',a.batch_size]
    if a.save_predictions:cmd+=['--save-predictions']
    subprocess.run(cmd,check=True)
