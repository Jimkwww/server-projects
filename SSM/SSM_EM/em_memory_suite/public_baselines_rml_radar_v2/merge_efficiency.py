import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
rows = []
for p in sorted((ROOT/'efficiency').glob('*.json')):
    rows.append(json.loads(p.read_text()))
env = rows[0] if rows else {}
(ROOT/'runs'/'efficiency.json').write_text(json.dumps({'env': {k: env[k] for k in
    ('gpu','torch','cuda','precision','allow_tf32_matmul','allow_tf32_cudnn','note')}, 'rows': rows}, indent=2))
print(f'merged {len(rows)} efficiency rows -> runs/efficiency.json')
for r in sorted(rows, key=lambda x: (x['task'], x['model'])):
    print(f"  {r['task']}/{r['model']}: tok{r['tokens']} b1 {r['batch1_ms_median']:.2f}ms "
          f"thr {r['throughput_batch64_samples_s']:.0f} samples/s mem_b64 {r['peak_allocated_MB_batch64']:.0f}MB")
