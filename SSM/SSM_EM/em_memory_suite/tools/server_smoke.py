"""Run ONLY on server: end-to-end small test, not scientific results."""
import json,subprocess,sys
from pathlib import Path
from emmem.synthetic import generate
from emmem.train import dump
root=Path('smoke_workspace').resolve();manifest=generate(root/'data',n=48,length=256)
registry=dict(pretrain=[str(manifest)],downstream=[dict(id='synthetic_waveform',manifest=str(manifest),task='waveform',perturbations={}),
    dict(id='synthetic_regression',manifest=str(manifest),task='parameters',perturbations={}),
    dict(id='synthetic_segmentation',manifest=str(manifest),task='activity',perturbations={})],device='cpu',workers=0,seeds=[17])
dump(root/'registry.json',registry)
for cmd in [
    [sys.executable,'-m','unittest','discover','-s','tests','-v'],
    [sys.executable,'-m','tools.audit_data','--registry',str(root/'registry.json'),'--out',str(root/'audit.json')],
    [sys.executable,'-m','emmem.plan','--registry',str(root/'registry.json'),'--out',str(root/'plan'),'--profile','smoke'],
    [sys.executable,'-m','emmem.runner','--plan',str(root/'plan'/'plan.json'),'--fail-fast'],
    [sys.executable,'-m','emmem.report','--root',str(root/'plan')]]:
    subprocess.run(cmd,check=True)
print('SERVER SMOKE PASSED (synthetic, not research evidence)')
