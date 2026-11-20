"""Deterministic synthetic mechanism benchmark, NOT RadioML or an OTA substitute.
Pulse-shaped digital symbols; several waveform families; independent source groups.
Absolute CFO is known by construction only. Segmentation labels refer to gating.
"""
import argparse,json
from pathlib import Path
import numpy as np

NAMES=['BPSK','QPSK','8PSK','16QAM','64QAM','2FSK','4FSK','AM','FM','tone','LFM','OFDM']

def generate_one(rng,length,cls,snr,ood=False):
    sps=int(rng.choice([5,7,9] if ood else [4,6,8]));t=np.arange(length)
    ns=length//sps+16
    if cls<=4:
        if cls<=2:
            order=[2,4,8][cls];a=np.exp(2j*np.pi*rng.integers(order,size=ns)/order)
        else:
            q=[4,8][cls-3];a=(2*rng.integers(q,size=ns)-q+1)+1j*(2*rng.integers(q,size=ns)-q+1)
            a=a/np.sqrt(np.mean(abs(a)**2))
        impulse=np.zeros(ns*sps,complex);impulse[::sps]=a
        # Raised-cosine pulse, not RRC; continuous with analytic singular limits.
        u=np.arange(-4*sps,4*sps+1)/sps;beta=.35
        den=1-(2*beta*u)**2
        pulse=np.sinc(u)*np.cos(np.pi*beta*u)/np.where(abs(den)<1e-8,1,den)
        near=abs(den)<1e-8;pulse[near]=np.pi/4*np.sinc(1/(2*beta))
        z=np.convolve(impulse,pulse,mode='same')[8*sps:8*sps+length]
    elif cls in (5,6):
        order=2 if cls==5 else 4
        f=np.repeat((rng.integers(order,size=ns)-(order-1)/2)*.035,sps)[:length]
        z=np.exp(2j*np.pi*np.cumsum(f))
    elif cls in (7,8):
        msg=np.sin(2*np.pi*rng.uniform(.002,.01)*t+rng.uniform(-np.pi,np.pi))
        z=(1+.7*msg).astype(complex) if cls==7 else np.exp(2j*np.pi*np.cumsum(.04*msg))
    elif cls==9:z=np.exp(2j*np.pi*rng.uniform(-.1,.1)*t)
    elif cls==10:
        slope=rng.uniform(-.00015,.00015);z=np.exp(2j*np.pi*(rng.uniform(-.05,.05)*t+.5*slope*t*t))
    else:
        chunks=[]
        for _ in range(length//80+2):
            bins=np.zeros(64,complex);bins[8:56]=np.exp(.5j*np.pi*rng.integers(4,size=48))
            v=np.fft.ifft(bins);chunks.append(np.r_[v[-16:],v])
        z=np.concatenate(chunks)[:length]
    z=z/np.sqrt(np.mean(abs(z)**2)+1e-9)
    start=int(rng.integers(0,length//4));end=int(rng.integers(3*length//4,length+1))
    gate=((t>=start)&(t<end)).astype(float)
    z=z*gate
    # Channel parameters vary independently of class; record origin, not infer identifiability.
    if ood:z=z+.35*np.r_[np.zeros(5,complex),z[:-5]]
    gain=float(np.exp(rng.uniform(-.7,.7)));cfo=float(rng.uniform(-.04,.04) if not ood else rng.choice([-1,1])*rng.uniform(.05,.08))
    z=z*gain*np.exp(1j*(2*np.pi*cfo*t+rng.uniform(-np.pi,np.pi)))
    noise=(rng.normal(size=length)+1j*rng.normal(size=length))*gain/np.sqrt(2*10**(snr/10))
    return np.stack(((z+noise).real,(z+noise).imag),-1).astype('float32'),np.array([np.log(gain),cfo],dtype='float32'),gate.astype('int64')

def generate(out,n=2400,length=1024,seed=31415,patch=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(dataset_id='synthetic_v1',group_namespace='synthetic_v1_sources',
        provenance='Independent generated examples, independent RNG streams by split; not a real-data foundation benchmark.',
        sample_rate_hz=None,center_frequency_hz=None,amplitude_policy='absolute gain retained',
        tasks=dict(waveform=dict(kind='classification',outputs=len(NAMES),classes=NAMES),
                   parameters=dict(kind='regression',outputs=2,names=['log_gain','injected_cfo_cycles_per_sample']),
                   activity=dict(kind='segmentation',outputs=2)),splits={})
    for j,split in enumerate(['train','val','test','ood']):
        rng=np.random.default_rng(seed+100003*j);count=n if split=='train' else max(24,n//4)
        xx=[];yy=[];pp=[];ss=[];snrs=[]
        for i in range(count):
            cls=i%len(NAMES);snr=float(rng.choice([-10,-5,0,5,10,20]))
            x,p,s=generate_one(rng,length,cls,snr,split=='ood');xx.append(x);yy.append(cls);pp.append(p)
            ss.append((s.reshape(-1,patch).mean(-1)>=.5).astype('int64'));snrs.append(snr)
        arrays=dict(x=np.stack(xx),waveform=np.array(yy),parameters=np.stack(pp),activity=np.stack(ss),snr=np.array(snrs),
            ids=np.array([f'{split}_{i}' for i in range(count)]),groups=np.array([f'{split}_source_{i}' for i in range(count)]))
        for k,v in arrays.items():np.save(out/f'{split}_{k}.npy',v,allow_pickle=False)
        manifest['splits'][split]=dict(x=f'{split}_x.npy',ids=f'{split}_ids.npy',groups=f'{split}_groups.npy',snr=f'{split}_snr.npy',
                 targets={k:f'{split}_{k}.npy' for k in manifest['tasks']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return out/'manifest.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=2400)
    p.add_argument('--length',type=int,default=1024);p.add_argument('--seed',type=int,default=31415)
    a=p.parse_args();print(generate(a.out,a.n,a.length,a.seed))
