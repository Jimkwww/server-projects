"""Create a small result archive, excluding checkpoints, raw IQ and giant predictions by default."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();index=[]
for file in root.rglob('*.pt'):
    h=hashlib.sha256()
    with file.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    index.append(dict(path=str(file.relative_to(root)),sha256=h.hexdigest(),size=file.stat().st_size))
(root/'checkpoint_index.json').write_text(json.dumps(index,indent=2))
allowed={'.md','.csv','.json','.jsonl','.png','.pdf','.log','.txt','.patch'}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for file in root.rglob('*'):
        if file.is_file() and file.resolve()!=out and file.suffix in allowed:z.write(file,str(file.relative_to(root)))
print(out)
