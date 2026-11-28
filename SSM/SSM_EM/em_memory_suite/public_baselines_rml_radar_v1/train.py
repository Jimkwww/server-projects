"""public_baselines_rml_radar_v1: 4-model (Q/structured/EMind/WavesFM) supervised comparison.

Two tasks: RML2018A modulation (24-class, L=1024) and RadChar-Tiny radar (5-class, L=512).
~10M params each, seed17, unified Q lr=3e-4 wd=0.01, cosine+5% warmup, grad_clip 1.0, fp32,
effective batch 128, val macro-F1 best checkpoint, full train set.

--task rml|radar --model q|structured|emind|wavesfm --gpu N [--timing]
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'emind_long_compare_v1'
sys.path.insert(0, str(V1)); sys.path.insert(0, str(V1/'EMind')); sys.path.insert(0, str(V1/'wavesfm'))
sys.path.insert(0, str(ROOT.parent)); sys.path.insert(0, '/home/user_wzk/wzk/projects/SSM/SSM_L')

from transformers.modeling_utils import PreTrainedModel
if not hasattr(PreTrainedModel, 'get_head_mask'):
    PreTrainedModel.get_head_mask = lambda self,h,n,i=False: h if h is not None else [None]*n
from models.block import SiTMAESdpaAttention
def _patch_sdpa():
    def fwd(self, hidden_states, head_mask=None, output_attentions=False, mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        am = None
        if mask is not None:
            am = mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
        ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=0.0)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(hidden_states.shape[0], -1, self.all_head_size)
        return (self.dropout(self.dense(ctx)),)
    SiTMAESdpaAttention.forward = fwd
_patch_sdpa()

from models.configuration_em import SiTMAEConfig
from models.modeling_em import SiTMAEForClassification
from models_vit import ModalityAdapterViT
from emmem.models import Encoder as StructEnc
from ssm_exp.model import max_abs_norm

DATA = ROOT.parent / 'data'
TASKS = {'rml': ('rml2018_clsdev', 24, 1024, 'modulation'),
         'radar': ('radchar_tiny', 5, 512, 'radar_waveform')}

# ~10M configs (params verified in match10.py)
CFG = {
    'emind':     dict(hidden_size=256, num_hidden_layers=12, num_attention_heads=8, intermediate_size=1024),
    'structured': dict(dim=152, depth=12),
    'wavesfm':    dict(embed_dim=320, depth=8),
    'q':          dict(dim=288, depth=12, state=144),
}

def make_model(name, L, nc, device):
    if name == 'emind':
        c = CFG['emind']
        cfg = SiTMAEConfig(**c, max_seq_len=(L,1), patch_size=(4,1), num_channels=2,
                           num_labels=nc, attention_type='sdpa', use_fs=False)
        m = SiTMAEForClassification(cfg).to(device)
        def fwd(x):  # x [B,L,2]
            xx = x / (x.abs().max(dim=1,keepdim=True).values.max(dim=-1,keepdim=True).values + 1e-6)
            xx = xx.clamp(-5,5).permute(0,2,1).unsqueeze(-1)
            patch = 4; B = x.shape[0]
            cls = torch.zeros(B,2,patch,1,device=x.device)
            xx = torch.cat([cls,xx],dim=2)
            n = (L+patch)//patch
            pat = xx.reshape(B,2,n,patch).permute(0,2,1,3).reshape(B,n,2*patch)
            pos = torch.arange(n,device=x.device).unsqueeze(0).repeat(B,1)
            sids = [torch.zeros(n,dtype=torch.long,device=x.device) for _ in range(B)]
            return m(pat,pos,sids).logits
        return m, fwd, [p for p in m.parameters() if p.requires_grad]
    if name == 'structured':
        c = CFG['structured']
        cfg = dict(patch=8, dim=c['dim'], states=c['dim']//2, depth=c['depth'], branched=True,
                   physical_input=True, roles=['energy','coherence','spectrum','period'], kind='ssm',
                   aux_route='local', residual=True, selective=True, real=False, scan='parallel')
        enc = StructEnc(cfg).to(device)
        head = nn.Linear(enc.out_dim, nc).to(device)
        for p in list(enc.aux.parameters())+list(enc.logvar.parameters())+list(enc.reconstruct.parameters()):
            p.requires_grad_(False)
        def fwd(x):
            xx = x / (x.abs().max(dim=1,keepdim=True).values.max(dim=-1,keepdim=True).values + 1e-6)
            xx = xx.clamp(-5,5)
            return head(enc(xx)['sequence'].mean(1))
        return (enc,head), fwd, [p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad]
    if name == 'wavesfm':
        c = CFG['wavesfm']
        m = ModalityAdapterViT(modality='iq', num_outputs=nc, embed_dim=c['embed_dim'], depth=c['depth'],
                               num_heads=8, mlp_ratio=4, iq_segment_len=16, iq_hop=16, iq_max_tokens=L//16,
                               iq_max_antennas=1, global_pool='token').to(device)
        def fwd(x):
            xx = x / (x.abs().max(dim=1,keepdim=True).values.max(dim=-1,keepdim=True).values + 1e-6)
            xx = xx.clamp(-5,5).permute(0,2,1).unsqueeze(2)
            return m(xx)
        return m, fwd, [p for p in m.parameters() if p.requires_grad]
    # q
    from pretrain.model import Encoder as QEnc
    c = CFG['q']
    cfg = dict(patch=8, dim=c['dim'], depth=c['depth'], heads=4, state=c['state'], selector_hidden=64,
               classes=nc, input_length=L, ffn_ratio=4, dropout=0.0, position='none', direction='forward',
               scan='parallel', ssm_gain=1.0)
    enc = QEnc(cfg, 'Q', seed=17).to(device)
    head = nn.Linear(c['dim'], nc).to(device)
    def fwd(x):
        xx = max_abs_norm(x.permute(0,2,1))
        return head(enc.forward_tokens(xx).mean(1))
    return (enc,head), fwd, [p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad]

def state_of(m): return (m[0].state_dict(), m[1].state_dict()) if isinstance(m,tuple) else m.state_dict()
def load_of(m,sd):
    if isinstance(m,tuple): m[0].load_state_dict(sd[0]); m[1].load_state_dict(sd[1])
    else: m.load_state_dict(sd)
def train_mode(m,on):
    if isinstance(m,tuple): m[0].train(on); m[1].train(on)
    else: m.train(on)

def macro_f1(y,p,nc):
    y,p=np.array(y),np.array(p)
    return float(np.mean([2*((p==c)&(y==c)).sum()/(2*((p==c)&(y==c)).sum()+((p==c)&(y!=c)).sum()+((p!=c)&(y==c)).sum()+1e-9) for c in range(nc)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['rml','radar'])
    ap.add_argument('--model', required=True, choices=['q','structured','emind','wavesfm'])
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--grad-accum', type=int, default=2)  # eff 128
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--out', default='runs')
    a = ap.parse_args()
    device = a.gpu
    seed = 17
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dname, nc, L, _ = TASKS[a.task]
    base = DATA / dname
    Xtr=np.load(base/'train_x.npy',mmap_mode='r'); Ytr=np.load(base/'train_y.npy',mmap_mode='r')
    Xva=np.load(base/'val_x.npy',mmap_mode='r');   Yva=np.load(base/'val_y.npy',mmap_mode='r')
    Xte=np.load(base/'test_x.npy',mmap_mode='r');  Yte=np.load(base/'test_y.npy',mmap_mode='r')
    print(f'[data] {dname}: train {len(Xtr)} val {len(Xva)} test {len(Xte)}, L={L}, nc={nc}', flush=True)

    model, fwd, params = make_model(a.model, L, nc, device)
    npar = sum(p.numel()*(2 if p.is_complex() else 1) for p in params)
    print(f'[model] {a.model}: {npar:,} trainable params', flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    eff = a.batch*a.grad_accum
    steps_per_epoch = (len(Xtr)+eff-1)//eff
    total_steps = a.epochs*steps_per_epoch
    warmup = int(0.05*total_steps)
    def lr_lambda(s):
        if s < warmup: return s/max(1,warmup)
        return 0.5*(1+math.cos(math.pi*(s-warmup)/max(1,total_steps-warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def run_epoch(sx,sy,train):
        if train: train_mode(model,True); perm=np.random.permutation(len(sx))
        else: train_mode(model,False); perm=np.arange(len(sx))
        opt.zero_grad(set_to_none=True)
        ws,total,correct,ay,ap=0,0,0,[],[]
        for i in range(0,len(perm),a.batch):
            idx=perm[i:i+a.batch]
            x=torch.from_numpy(np.asarray(sx[idx],dtype=np.float32)).to(device)
            y=torch.from_numpy(sy[idx].astype(np.int64)).to(device)
            if train: logits=fwd(x)
            else:
                with torch.inference_mode(): logits=fwd(x)
            loss=nn.functional.cross_entropy(logits,y,reduction='sum')
            if train: loss.backward(); ws+=len(idx)
            total+=loss.item(); correct+=(logits.argmax(-1)==y).sum().item()
            ay.extend(y.cpu().tolist()); ap.extend(logits.argmax(-1).cpu().tolist())
            if train and (i+a.batch>=len(perm) or ((i//a.batch+1)%a.grad_accum==0)):
                for p in params:
                    if p.grad is not None: p.grad/=ws
                torch.nn.utils.clip_grad_norm_(params,1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); ws=0
        return total/len(perm), correct/len(perm), ay, ap

    if a.timing:
        # 1 full train epoch + 1 full val epoch, report seconds
        t0=time.time(); run_epoch(Xtr,Ytr,True); t_tr=time.time()-t0
        t0=time.time(); run_epoch(Xva,Yva,False); t_va=time.time()-t0
        print(f'[timing] {a.model}/{a.task}: train_epoch={t_tr:.1f}s val_epoch={t_va:.1f}s -> {a.epochs}ep est={(t_tr+t_va)*a.epochs/60:.1f}min', flush=True)
        return

    out = Path(a.out)/a.task/a.model
    out.mkdir(parents=True,exist_ok=True)
    best=-1; hist=[]
    for ep in range(a.epochs):
        t0=time.time()
        trl,tra,_,_=run_epoch(Xtr,Ytr,True)
        val,vaa,vy,vp=run_epoch(Xva,Yva,False)
        vf1=macro_f1(vy,vp,nc)
        hist.append({'ep':ep,'tr_loss':trl,'tr_acc':tra,'va_acc':vaa,'va_f1':vf1,'secs':time.time()-t0})
        if vf1>best:
            best=vf1
            torch.save({'model':state_of(model),'epoch':ep,'best':best}, out/'best.pt')
        torch.save({'model':state_of(model),'optimizer':opt.state_dict(),'epoch':ep,'best':best}, out/'last.pt')
        print(f'[ep{ep}] tr_loss {trl:.4f} tr_acc {tra:.4f} | va_acc {vaa:.4f} va_f1 {vf1:.4f} | {time.time()-t0:.0f}s', flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    load_of(model,ck['model']); train_mode(model,False)
    tel,tea,ty,tp=run_epoch(Xte,Yte,False)
    tef1=macro_f1(ty,tp,nc)
    result={'task':a.task,'model':a.model,'test_oa':tea,'test_macro_f1':tef1,'best_val_f1':best,
            'params':npar,'epochs':a.epochs,'lr':a.lr,'batch':a.batch,'grad_accum':a.grad_accum,
            'history':hist}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'test_pred.npz',y=np.array(ty),pred=np.array(tp))
    print(f'[TEST] {a.model}/{a.task}: oa={tea:.4f} f1={tef1:.4f} best_val={best:.4f}')

if __name__=='__main__':
    main()
