import math
import torch
from torch import nn
from .features import ROLES,DIMS,patches,descriptors

def affine_scan(a,b,initial=None):
    """Inclusive diagonal affine scan, O(log T) stages, differentiable complex64.
    Hillis-Steele reference, not a fused Mamba kernel. O(T log T) work.
    """
    aa,bb=a,b
    step=1
    while step<a.shape[1]:
        bb=torch.cat((bb[:,:step],bb[:,step:]+aa[:,step:]*bb[:,:-step]),1)
        aa=torch.cat((aa[:,:step],aa[:,step:]*aa[:,:-step]),1)
        step*=2
    return bb if initial is None else bb+aa*initial[:,None]

class ComplexSSM(nn.Module):
    def __init__(self,dim,states,selective=True,real=False,scan='parallel',bidirectional=False):
        super().__init__()
        self.states,self.real,self.selective,self.scan=states,real,selective,scan
        self.bidirectional=bidirectional
        self.B=nn.Linear(dim,states*(1 if real else 2),bias=False)
        self.C=nn.Linear(states*(1 if real else 2),dim,bias=False)
        self.log_alpha=nn.Parameter(torch.full((states,),-2.))
        self.theta=nn.Parameter(torch.linspace(-math.pi,math.pi,states)) if not real else None
        self.gates=nn.Linear(dim,2*states) if selective else None
        self.skip=nn.Parameter(torch.ones(dim))
    def forward(self,u,mode='normal'):
        v=self.B(u)
        if not self.real:
            vr,vi=v.chunk(2,-1);v=torch.complex(vr,vi)
        if self.selective:
            delta,g=self.gates(u).chunk(2,-1)
            delta=torch.nn.functional.softplus(delta)+1e-4;g=g.sigmoid()
        else:
            delta=torch.ones_like(v.real);g=torch.ones_like(v.real)
        rho=torch.exp(-torch.nn.functional.softplus(self.log_alpha)*delta)
        a=rho if self.real else torch.polar(rho,self.theta.expand_as(rho))
        w=g*v
        if mode=='reset': h=w
        elif self.scan=='serial':
            prev=torch.zeros_like(w[:,0]);out=[]
            for i in range(u.shape[1]):
                prev=a[:,i]*prev+w[:,i];out.append(prev)
            h=torch.stack(out,1)
        else: h=affine_scan(a,w)
        if self.bidirectional and mode!='reset':
            backward=affine_scan(a.flip(1),w.flip(1)).flip(1)
            h=(h+backward)*.5
        out=h if self.real else torch.cat((h.real,h.imag),-1)
        return self.C(out)+self.skip*u

class Block(nn.Module):
    def __init__(self,dim,states,kind='ssm',residual=True,selective=True,real=False,scan='parallel',bidirectional=False):
        super().__init__();self.kind,self.residual=kind,residual
        self.bidirectional=bidirectional
        self.n1=nn.LayerNorm(dim);self.n2=nn.LayerNorm(dim)
        if kind=='transformer':
            heads=4 if dim%4==0 else 1
            self.core=nn.MultiheadAttention(dim,heads,batch_first=True,dropout=0)
        elif kind=='pointwise': self.core=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        else: self.core=ComplexSSM(dim,states,selective,real,scan,bidirectional)
        self.ff=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
    def forward(self,x,mode='normal'):
        z=self.n1(x)
        if self.kind=='transformer':
            mask=None if self.bidirectional else torch.ones(z.shape[1],z.shape[1],dtype=torch.bool,device=z.device).triu(1)
            if mode=='reset':mask=~torch.eye(z.shape[1],dtype=torch.bool,device=z.device)
            y=self.core(z,z,z,attn_mask=mask,need_weights=False)[0]
        elif self.kind=='pointwise': y=self.core(z)
        else:y=self.core(z,mode)
        x=x+y if self.residual else y
        y=self.ff(self.n2(x))
        return x+y if self.residual else y

class Encoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=dict(cfg);self.patch=cfg.get('patch',8)
        self.roles=tuple(cfg.get('roles',list(ROLES)))
        self.branched=cfg.get('branched',True)
        self.dim=cfg.get('dim',64);self.names=self.roles+(('general',) if cfg.get('general',True) else ()) if self.branched else ('general',)
        self.front=nn.ModuleDict();self.stacks=nn.ModuleDict()
        for name in self.names:
            physical=cfg.get('physical_input',True) and name!='general'
            size=DIMS[name] if physical else self.patch*2
            self.front[name]=nn.Linear(size,self.dim)
            self.stacks[name]=nn.ModuleList([Block(self.dim,cfg.get('states',32),
                cfg.get('kind','ssm'),cfg.get('residual',True),cfg.get('selective',True),
                cfg.get('real',False),cfg.get('scan','parallel'),cfg.get('bidirectional',False)) for _ in range(cfg.get('depth',2))])
        self.mask_token=nn.ParameterDict({k:nn.Parameter(torch.zeros(self.dim)) for k in self.names})
        self.out_dim=self.dim*len(self.names)
        # Concatenation remains accessible; no irreversible low-dimensional bottleneck.
        self.fusion=nn.Sequential(nn.LayerNorm(self.out_dim),nn.Linear(self.out_dim,self.out_dim),nn.GELU())
        self.reconstruct=nn.Linear(self.out_dim,self.patch*2)
        if cfg.get('freeze_front',False):
            for p in self.front.parameters():p.requires_grad=False
        self.aux=nn.ModuleDict()
        self.logvar=nn.ModuleDict()
        for role in self.roles:
            src=self.out_dim if cfg.get('aux_route','local')=='fused' else self.dim
            self.aux[role]=nn.Linear(src,DIMS[role])
            if cfg.get('uncertainty',False):self.logvar[role]=nn.Linear(src,DIMS[role])
    def forward(self,x,mask=None,mode='normal',drop=None):
        p=patches(x,self.patch)
        if mask is not None:
            x=x.clone().reshape(x.shape[0],p.shape[1],self.patch,2)
            x=x.masked_fill(mask[:,:,None,None],0).reshape(x.shape[0],-1,2)
            p=patches(x,self.patch)
        physical=descriptors(x,self.patch,history=1) if self.cfg.get('physical_input',True) and self.branched else {}
        hs={}
        for name in self.names:
            inp=physical[name] if name in physical else p
            h=self.front[name](inp)
            if self.cfg.get('kind','ssm')=='transformer':
                pos=torch.arange(h.shape[1],device=h.device,dtype=h.dtype)[:,None]
                freq=torch.exp(torch.arange(0,self.dim,2,device=h.device,dtype=h.dtype)*(-math.log(10000.)/self.dim))
                pe=torch.zeros(h.shape[1],self.dim,device=h.device,dtype=h.dtype)
                pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq[:self.dim//2])
                h=h+pe[None]
            if mask is not None:h=torch.where(mask[...,None],self.mask_token[name][None,None],h)
            for block in self.stacks[name]:h=block(h,mode)
            if name==drop:h=torch.zeros_like(h)
            hs[name]=h
        joined=torch.cat(list(hs.values()),-1)
        fused=joined+self.fusion(joined)
        aux={role:self.aux[role](fused if self.cfg.get('aux_route','local')=='fused'
             else (hs[role] if role in hs else hs['general'])) for role in self.roles}
        logvar={role:head(fused if self.cfg.get('aux_route','local')=='fused' else (hs[role] if role in hs else hs['general'])).clamp(-8,6) for role,head in self.logvar.items()}
        return dict(sequence=fused,branches=hs,aux=aux,logvar=logvar,reconstruction=self.reconstruct(fused))

class TaskModel(nn.Module):
    def __init__(self,encoder,task,outputs):
        super().__init__();self.encoder=encoder;self.task=task
        self.head=nn.Linear(encoder.out_dim,outputs)
    def forward(self,x,mode='normal',drop=None):
        h=self.encoder(x,mode=mode,drop=drop)['sequence']
        return self.head(h if self.task=='segmentation' else h.mean(1))
