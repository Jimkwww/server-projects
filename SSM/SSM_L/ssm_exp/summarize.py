import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .utils import save_json


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def protocol_key(cfg,provenance):
    # Model-specific architecture remains in each run config; group common protocol only.
    train={k:v for k,v in cfg['train'].items() if k!='seed'}
    model={k:cfg['model'][k] for k in ('patch','dim','depth','classes','heads','ffn_ratio','dropout','position','input_length','direction')}
    return hashlib.sha256(json.dumps({'train':train,'common_model':model,'data':provenance['data'],
        'code':provenance['code_sha256'],'protocol':cfg['protocol']},sort_keys=True).encode()).hexdigest()[:12]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]; curves=[]; efficiency=[]; missing=[]
    for path in sorted(Path(a.runs).rglob('config.json')):
        run=path.parent
        if not (run/'DONE.json').exists(): missing.append(str(run)); continue
        if json.loads((run/'DONE.json').read_text())['smoke']:continue
        c=json.loads(path.read_text()); prov=json.loads((run/'provenance.json').read_text())
        val=json.loads((run/'best_val.json').read_text()); m=val['metrics']
        key=protocol_key(c,prov)
        info=json.loads((run/'model_info.json').read_text())
        base={'run':str(run),'protocol':key,'variant':c['model']['variant'],'seed':c['train']['seed'],
              'augment_max':c['train']['augment_cfo_max']}
        row={**base,'epoch':val['best_epoch'],'oa':m['overall']['oa'],'macro_f1':m['overall']['macro_f1'],
             'low_oa':m['low']['oa'] if m['low'] else None,'high_oa':m['high']['oa'] if m['high'] else None,
             'parameters':info['real_scalar_parameters']}
        rows.append(row)
        for snr,score in m['snr'].items():
            curves.append({**base,'split':'val','kind':'snr_clean','x':float(snr),
                           'oa':score['oa'],'macro_f1':score['macro_f1']})
        for ep in sorted(run.glob('eval_*/results.json')):
            ev=json.loads(ep.read_text())
            for case in ev['cases']:
                kind=case['kind']; score=case['metrics']['overall']
                x=case['phase_radians'] if kind=='phase' else case['frequency_cycles_per_sample']
                curves.append({**base,'split':ev['split'],'kind':kind,'x':x,
                               'oa':score['oa'],'macro_f1':score['macro_f1']})
        if (run/'benchmark.json').exists():
            bench=json.loads((run/'benchmark.json').read_text())
            for b in bench['rows']:efficiency.append({**base,**b,'device':bench['environment']['device'],'scan':bench['scan']})
    groups=defaultdict(list)
    seen=set()
    for r in rows:
        ident=(r['protocol'],r['variant'],r['seed'])
        if ident in seen: raise ValueError('duplicate protocol/variant/seed; separate campaigns: '+str(ident))
        seen.add(ident); groups[ident[:2]].append(r)
    summary=[]
    for (key,variant),rs in groups.items():
        item={'protocol':key,'variant':variant,'n_seeds':len(rs),'seeds':','.join(str(r['seed']) for r in rs)}
        for metric in ('oa','macro_f1','low_oa','high_oa'):
            vals=[r[metric] for r in rs if r[metric] is not None]
            item[metric+'_mean']=float(np.mean(vals)) if vals else None
            item[metric+'_sample_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summary.append(item)
    paired=[]
    for key in {r['protocol'] for r in rows}:
        for target,reference in [('S0','G'),('S1','S0'),('S2','S1'),('S2','S2_concat')]:
            aa={r['seed']:r for r in rows if r['protocol']==key and r['variant']==target}
            bb={r['seed']:r for r in rows if r['protocol']==key and r['variant']==reference}
            for seed in sorted(aa.keys()&bb.keys()):
                paired.append({'protocol':key,'target':target,'reference':reference,'seed':seed,
                    'delta_oa_pp':100*(aa[seed]['oa']-bb[seed]['oa']),
                    'delta_macro_f1_pp':100*(aa[seed]['macro_f1']-bb[seed]['macro_f1'])})
    csv_write(out/'runs.csv',rows); csv_write(out/'summary.csv',summary)
    csv_write(out/'paired_deltas.csv',paired); csv_write(out/'curves.csv',curves); csv_write(out/'efficiency.csv',efficiency)
    save_json(out/'incomplete_runs.json',missing)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key,split,kind in sorted({(r['protocol'],r['split'],r['kind']) for r in curves}):
            if kind=='clean':continue
            fig,ax=plt.subplots(figsize=(6,4))
            subset=[r for r in curves if (r['protocol'],r['split'],r['kind'])==(key,split,kind)]
            for v in sorted({r['variant'] for r in subset}):
                buckets=defaultdict(list)
                for r in subset:
                    if r['variant']==v:buckets[r['x']].append(r['oa'])
                xs=sorted(buckets); mean=np.array([np.mean(buckets[x]) for x in xs]); std=np.array([
                    np.std(buckets[x],ddof=1) if len(buckets[x])>1 else 0 for x in xs])
                ax.plot(xs,mean,marker='.',label=v); ax.fill_between(xs,mean-std,mean+std,alpha=.12)
            ax.set(xlabel={'snr_clean':'SNR (dB)','cfo':'CFO (cycles/sample)','phase':'Phase (rad)'}[kind],ylabel='OA',title=f'{split}: {kind}')
            ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f'{key}_{split}_{kind}.png',dpi=160); plt.close(fig)
    (out/'README.md').write_text('# Results\n\nOnly completed non-smoke runs are included. Means and sample standard deviations use available seeds; verify n_seeds=3.\n\nDo not mix protocol hashes. Paired differences are in percentage points. Inspect individual seed directions, clean accuracy, and full CFO curves; three seeds do not establish significance. Test results never select checkpoints.\n')
    print(f'Collected {len(rows)} completed runs, {len(missing)} incomplete runs')

if __name__=='__main__':main()
