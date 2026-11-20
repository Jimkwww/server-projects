"""Resumable DAG executor. One process per plan shard; no concurrent shared-state writes."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .train import dump

def run(plan,keep_going=True,retry_failed=False,seed=None):
    path=Path(plan).resolve();root=path.parent
    lock=root/'RUNNING.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'{lock} exists. Check owner process; remove only after confirming it is dead.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        payload=json.loads(path.read_text());jobs=payload['jobs'];statepath=root/'status.json'
        state=json.loads(statepath.read_text()) if statepath.exists() else {}
        known={j['id'] for j in jobs}
        if len(known)!=len(jobs):raise ValueError('Duplicate job id')
        for j in jobs:
            if any(d not in known for d in j['deps']):raise ValueError('Missing dependency')
        pending=jobs[:]
        while pending:
            progressed=False
            for j in pending[:]:
                jid=j['id'];signature=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
                old=state.get(jid,{})
                if old.get('signature') not in (None,signature):raise ValueError(f'Job changed: {jid}; use new plan directory')
                if old.get('status')=='done' or (old.get('status')=='failed' and not retry_failed):
                    pending.remove(j);progressed=True;continue
                if any(state.get(d,{}).get('status') in ('failed','blocked') for d in j['deps']):
                    state[jid]=dict(status='blocked',signature=signature,reason='failed dependency');dump(statepath,state)
                    pending.remove(j);progressed=True;continue
                if not all(state.get(d,{}).get('status')=='done' for d in j['deps']):continue
                logs=root/'logs';logs.mkdir(exist_ok=True);tic=time.time()
                state[jid]=dict(status='running',signature=signature,start=tic);dump(statepath,state)
                with (logs/f'{jid}.log').open('a') as log:
                    result=subprocess.run([sys.executable,'-m',j['module'],*j['args']],stdout=log,stderr=subprocess.STDOUT)
                state[jid]=dict(status='done' if result.returncode==0 else 'failed',signature=signature,
                                returncode=result.returncode,seconds=time.time()-tic)
                dump(statepath,state);print(jid,state[jid]['status'],flush=True)
                pending.remove(j);progressed=True
                if result.returncode and not keep_going:raise RuntimeError(f'Failed: {jid}')
            if not progressed:raise RuntimeError('DAG cycle or unresolved dependency')
        failures=[k for k,v in state.items() if v['status']!='done']
        dump(root/'completion.json',dict(all_done=not failures,failures=failures,total=len(jobs)))
        if failures:raise RuntimeError(f'{len(failures)} jobs failed/blocked; see status.json')
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--retry-failed',action='store_true');p.add_argument('--fail-fast',action='store_true')
    a=p.parse_args();run(a.plan,not a.fail_fast,a.retry_failed)
