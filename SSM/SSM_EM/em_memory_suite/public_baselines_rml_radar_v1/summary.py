import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
models = ['q', 'structured', 'emind', 'wavesfm']
tasks = ['rml', 'radar']
rows = []
for task in tasks:
    for m in models:
        r = json.loads((ROOT/'runs'/task/m/'result.json').read_text())
        rows.append({'task': task, 'model': m, 'test_oa': r['test_oa'], 'test_macro_f1': r['test_macro_f1'],
                     'best_val_f1': r['best_val_f1'], 'params': r['params'], 'epochs': r['epochs']})
print('=== public_baselines_rml_radar_v1 汇总（~10M, seed17, lr3e-4, val macro-F1 best）===')
for task in tasks:
    print(f'\n[{task}]')
    print(f'{"model":12s} {"test_OA":>9s} {"test_F1":>9s} {"best_valF1":>10s} {"params":>10s}')
    for r in rows:
        if r['task']==task:
            print(f'{r["model"]:12s} {r["test_oa"]:>9.4f} {r["test_macro_f1"]:>9.4f} {r["best_val_f1"]:>10.4f} {r["params"]:>10,}')
(ROOT/'runs'/'summary.json').write_text(json.dumps(rows, indent=2))
print('\n[saved] runs/summary.json')
