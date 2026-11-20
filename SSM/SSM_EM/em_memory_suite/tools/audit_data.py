import argparse,json
from pathlib import Path
from emmem.data import audit
p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--out',required=True)
a=p.parse_args();r=json.loads(Path(a.registry).read_text())
paths=sorted(set(r['pretrain']+[d['manifest'] for d in r['downstream']]+[d[k] for d in r.get('novelty',[]) for k in ['normal','anomaly']]))
print(audit(paths,a.out))
