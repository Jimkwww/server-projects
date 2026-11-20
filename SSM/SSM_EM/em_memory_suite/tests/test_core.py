import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from emmem.models import affine_scan,Encoder,ComplexSSM,TaskModel
from emmem.features import descriptors,history_average,transform,patches
from emmem.synthetic import generate
from emmem.data import audit,IQDataset
from emmem.metrics import classification,binary_auc
from emmem.train import pre_loss,run

# Small deterministic CPU tests, no CUDA allocation and no experiment matrix.
torch.set_num_threads(1)
SMALL=dict(patch=8,dim=8,states=4,depth=1,branched=True,physical_input=True,
           roles=['energy','coherence','spectrum','period'],aux_route='local')

class Core(unittest.TestCase):
    def setUp(self):torch.manual_seed(8)
    def test_scan_forward_backward(self):
        a=(torch.rand(2,13,3)*.7+0j).requires_grad_();b=torch.randn(2,13,3,dtype=torch.complex64,requires_grad=True)
        initial=torch.randn(2,3,dtype=torch.complex64)
        h=initial;steps=[]
        for n in range(13):h=a[:,n]*h+b[:,n];steps.append(h)
        serial=torch.stack(steps,1);parallel=affine_scan(a,b,initial)
        torch.testing.assert_close(serial,parallel,atol=1e-6,rtol=1e-5)
        gs=torch.autograd.grad(serial.abs().square().sum(),(a,b),retain_graph=True)
        gp=torch.autograd.grad(parallel.abs().square().sum(),(a,b))
        for x,y in zip(gs,gp):torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-5)
    def test_scan_single(self):
        a=torch.rand(2,1,3);b=torch.rand(2,1,3);torch.testing.assert_close(affine_scan(a,b),b)
    def test_ssm_serial(self):
        a=ComplexSSM(8,4,scan='serial');b=copy.deepcopy(a);b.scan='parallel';x=torch.randn(2,9,8)
        torch.testing.assert_close(a(x),b(x))
    def test_descriptors_causal(self):
        x=torch.randn(2,256,2);y=x.clone();y[:,128:]=torch.randn_like(y[:,128:])
        a=descriptors(x);b=descriptors(y)
        for k in a:torch.testing.assert_close(a[k][:,:16],b[k][:,:16])
    def test_descriptors_zero(self):
        for v in descriptors(torch.zeros(1,256,2)).values():self.assertTrue(torch.isfinite(v).all())
    def test_phase_invariance(self):
        x=torch.randn(2,256,2);a=descriptors(x);b=descriptors(transform(x,'phase',.9))
        for k in ['energy','coherence','spectrum','period']:torch.testing.assert_close(a[k],b[k],atol=2e-6,rtol=2e-5)
    def test_cfo_correlation_transport(self):
        x=torch.randn(2,256,2);v=.013;a=descriptors(x,history=1)['coherence'];b=descriptors(transform(x,'cfo',v),history=1)['coherence']
        ca=torch.complex(a[...,:4],a[...,4:]);cb=torch.complex(b[...,:4],b[...,4:])
        expected=ca*torch.exp(2j*torch.pi*v*torch.tensor([1,2,4,8]))
        torch.testing.assert_close(cb,expected,atol=2e-6,rtol=1e-4)
    def test_history(self):
        x=torch.arange(5.).reshape(1,5,1);torch.testing.assert_close(history_average(x,3),torch.tensor([0,.5,1,2,3]).reshape(1,5,1))
    def test_mask_no_leak(self):
        m=Encoder(SMALL).eval();x=torch.randn(2,256,2);mask=torch.zeros(2,32,dtype=torch.bool);mask[:,5:9]=True
        y=x.clone();y[:,40:72]=100
        a=m(x,mask)['sequence'];b=m(y,mask)['sequence'];torch.testing.assert_close(a,b)
    def test_model_causal(self):
        m=Encoder(SMALL).eval();x=torch.randn(2,256,2);y=x.clone();y[:,128:]+=7
        torch.testing.assert_close(m(x)['sequence'][:,:16],m(y)['sequence'][:,:16])
    def test_batch_isolation(self):
        m=Encoder(SMALL).eval();x=torch.randn(2,256,2)
        torch.testing.assert_close(m(x)['sequence'][:1],m(x[:1])['sequence'],atol=1e-6,rtol=1e-5)
    def test_gradient(self):
        m=Encoder(SMALL);loss,_=pre_loss(m,torch.randn(2,256,2),dict(aux_weight=1.,equiv_weight=.1));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        for k in m.names:self.assertGreater(m.front[k].weight.grad.abs().sum(),0)
    def test_uncertainty_gradient(self):
        m=Encoder({**SMALL,'uncertainty':True})
        loss,_=pre_loss(m,torch.randn(2,256,2),dict(aux_weight=1.))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for head in m.logvar.values():self.assertTrue(torch.isfinite(head.weight.grad).all())
    def test_aux_isolation(self):
        m=Encoder(SMALL);o=m(torch.randn(2,256,2));o['aux']['energy'].square().mean().backward()
        self.assertIsNotNone(m.front['energy'].weight.grad)
        self.assertIsNone(m.front['coherence'].weight.grad)
        self.assertIsNone(m.front['general'].weight.grad)
    def test_variants(self):
        for change in [dict(branched=False,aux_route='fused'),dict(general=False),dict(kind='pointwise'),dict(real=True),dict(kind='transformer'),dict(residual=False),dict(uncertainty=True),dict(bidirectional=True),dict(roles=['energy','coherence','spectrum','period','statistics'])]:
            m=Encoder({**SMALL,**change});o=m(torch.randn(1,256,2));self.assertEqual(o['sequence'].shape[1],32)
    def test_transformer_reset(self):
        m=Encoder({**SMALL,'branched':False,'physical_input':False,'kind':'transformer'}).eval()
        x=torch.randn(1,256,2);y=x.clone();y[:,:64]+=3
        torch.testing.assert_close(m(x,mode='reset')['sequence'][:,8:],m(y,mode='reset')['sequence'][:,8:])
    def test_task_shapes(self):
        x=torch.randn(2,256,2)
        for task,shape in [('classification',(2,3)),('regression',(2,3)),('segmentation',(2,32,3))]:
            self.assertEqual(TaskModel(Encoder(SMALL),task,3)(x).shape,shape)
    def test_metrics(self):
        self.assertEqual(classification([0,1],[0,1],2)['macro_f1'],1.)
        self.assertEqual(binary_auc([0,1],[1,1]),.5);self.assertEqual(binary_auc([0,1],[0,1]),1.)
    def test_audit_and_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=generate(d,n=24,length=256);audit([p],Path(d)/'audit.json')
            ds=IQDataset(p,'train','waveform',fraction=.1);self.assertEqual(len(ds),12)
            m=json.loads(p.read_text());g=Path(d)/m['splits']['val']['groups'];a=np.load(g).astype('U64');a[0]='train_source_0';np.save(g,a)
            with self.assertRaises(RuntimeError):audit([p],Path(d)/'bad.json')

if __name__=='__main__':unittest.main()
