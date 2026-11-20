"""Causal descriptors. Every target at n uses samples <= end of patch n.
No raw-phase unwrap, no absolute RF-frequency claims, no clean side channel.
"""
import torch
import torch.nn.functional as F

ROLES = ('energy', 'coherence', 'spectrum', 'period')
DIMS = dict(energy=4, coherence=8, spectrum=16, period=8, statistics=6)

def complex_iq(x):
    return torch.complex(x[..., 0].float(), x[..., 1].float())

def patches(x, patch):
    if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] % patch:
        raise ValueError('IQ must be [B,L,2], L divisible by patch')
    return x.reshape(x.shape[0], -1, patch*2)

def history_average(x, window):
    # Inclusive rolling window, including startup partial windows.
    if window <= 1:
        return x
    cs = F.pad(x.cumsum(1), (0, 0, 1, 0))
    end = torch.arange(1, x.shape[1]+1, device=x.device)
    start = (end-window).clamp_min(0)
    return (cs[:, end]-cs[:, start])/(end-start)[None, :, None]

def descriptors(x, patch=8, history=8):
    z = complex_iq(x)
    b,l = z.shape
    if l % patch:
        raise ValueError('length must be divisible by patch')
    n=l//patch
    power=z.abs().square().reshape(b,n,patch)
    mean=power.mean(-1)
    energy=torch.stack((torch.log1p(mean),torch.log1p(power.std(-1,unbiased=False)),
                        torch.log1p(power.amax(-1)),torch.log1p(power.square().mean(-1))),-1)
    def lag_corr(lag):
        if lag >= l:
            return torch.zeros(b,n,dtype=z.dtype,device=z.device)
        prev=F.pad(z[:,:-lag],(lag,0))
        valid=(torch.arange(l,device=z.device)>=lag).float()[None,:]
        num=(z*prev.conj()).reshape(b,n,patch).sum(-1)
        den=((z.abs().square()*valid).reshape(b,n,patch).sum(-1)*
             prev.abs().square().reshape(b,n,patch).sum(-1)).sqrt()
        return num/(den+1e-6)
    c=torch.stack([lag_corr(k) for k in (1,2,4,8)],-1)
    coherence=torch.cat((c.real,c.imag),-1)
    # Trailing 64-sample Hann window at each patch end; no future context.
    win=64
    frames=F.pad(z,(win-1,0)).unfold(1,win,1)[:,patch-1::patch,:]
    spec=torch.fft.fftshift(torch.fft.fft(frames*torch.hann_window(win,device=z.device),dim=-1),dim=-1)
    ps=spec.abs().square().reshape(b,n,16,4).mean(-1)/win
    spectrum=torch.log1p(ps)
    period=torch.stack([lag_corr(k).abs() for k in (8,16,24,32,48,64,96,128)],-1)
    zp=z.reshape(b,n,patch)
    norm=power.mean(-1).clamp_min(1e-6)
    m2=zp.square().mean(-1)/norm
    m4=(zp**4).mean(-1)/norm.square()
    stats=torch.stack((m2.real,m2.imag,m4.real,m4.imag,power.square().mean(-1)/norm.square(),
                       (zp.abs()**3).mean(-1)/norm.pow(1.5)),-1)
    raw=dict(energy=energy,coherence=coherence,spectrum=spectrum,period=period,statistics=stats)
    return {k:history_average(v,history) for k,v in raw.items()}

def transform(x, kind, value, generator=None):
    z=complex_iq(x)
    t=torch.arange(z.shape[1],device=x.device)
    if kind=='phase': z=z*torch.exp(torch.tensor(1j*value,device=x.device))
    elif kind=='cfo': z=z*torch.exp(2j*torch.pi*value*t) # cycles/sample
    elif kind=='gain': z=z*value
    elif kind=='noise':
        p=z.abs().square().mean(-1,keepdim=True)
        noise=torch.randn(x.shape,device=x.device,generator=generator)
        z=z+complex_iq(noise)*(p/(2*10**(value/10))).sqrt()
    elif kind=='iq_imbalance': z=z+value*z.conj()
    elif kind=='clip':
        threshold=value*z.abs().square().mean(-1,keepdim=True).sqrt()
        z=z*(threshold/z.abs().clamp_min(1e-6)).clamp_max(1)
    elif kind=='multipath': z=z+value*F.pad(z[:,:-3],(3,0))
    elif kind=='identity': pass
    else: raise ValueError(kind)
    return torch.stack((z.real,z.imag),-1)
