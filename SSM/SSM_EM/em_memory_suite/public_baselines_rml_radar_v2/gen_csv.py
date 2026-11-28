import json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent/'public_baselines_rml_radar_v1'/'runs'
V2 = ROOT/'runs'
rows = []
# v1 seed17 lr3e-4 30ep
for task in ['rml','radar']:
    for m in ['q','structured','emind','wavesfm']:
        r = json.loads((V1/task/m/'result.json').read_text())
        rows.append(dict(task=task, model=m, seed=17, lr=3e-4, epochs=30, source='v1',
                         test_oa=r['test_oa'], test_macro_f1=r['test_macro_f1'],
                         best_val_f1=r['best_val_f1'], params=r['params']))
# v2 all
for p in sorted(V2.glob('*/*/*/result.json')):
    r = json.loads(p.read_text())
    rows.append(dict(task=r['task'], model=r['model'], seed=r['seed'], lr=r['lr'], epochs=r['epochs'],
                     source='v2', test_oa=r['test_oa'], test_macro_f1=r['test_macro_f1'],
                     best_val_f1=r['best_val_f1'], params=r['params']))
rows.sort(key=lambda x: (x['task'], x['model'], x['seed'], x['epochs'], x['lr']))
with open(ROOT/'summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'summary.csv: {len(rows)} rows')
# also a compact per-run json
(ROOT/'summary_runs.json').write_text(json.dumps(rows, indent=2))
print('summary_runs.json written')
