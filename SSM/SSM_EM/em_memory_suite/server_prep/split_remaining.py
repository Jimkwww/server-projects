import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / 'configs/registry.server.json'
base = json.loads(REG.read_text())
base['batch_size'] = 128
base['variants'] = ['shuffled_aux', 'structured_recon']
for gpu, seed in [('cuda:0', 17), ('cuda:1', 42), ('cuda:2', 2026)]:
    r = dict(base)
    r['device'] = gpu
    r['seeds'] = [seed]
    out = ROOT / f'configs/registry.remaining.{gpu.replace(":", "")}.json'
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f'{gpu}: seed {seed} -> {out.name}')
