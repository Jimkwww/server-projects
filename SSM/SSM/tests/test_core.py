import math
import numpy as np
import pytest
import torch
from ssm_exp.model import (serial_scan,parallel_scan,physical_features,perturb,ComplexSSM,Classifier,parameter_count)
from ssm_exp.utils import metrics

torch.set_num_threads(1)

@pytest.mark.parametrize('length',[1,2,7,16,33])
def test_scan_values_and_gradients(length):
    torch.manual_seed(3)
    a=(torch.randn(2,length,5,dtype=torch.cdouble)*.2).requires_grad_()
    b=torch.randn_like(a,requires_grad=True)
    serial=serial_scan(a,b);parallel=parallel_scan(a,b)
    torch.testing.assert_close(serial,parallel,rtol=1e-10,atol=1e-10)
    ga=torch.autograd.grad(serial.abs().square().sum(),(a,b),retain_graph=True)
    gb=torch.autograd.grad(parallel.abs().square().sum(),(a,b))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-9)


def test_complex_expansion_and_hand_recurrence():
    a=torch.tensor([[[.2+.4j],[.3-.1j],[.8+.2j]]],dtype=torch.cdouble)
    b=torch.tensor([[[1.+2j],[3.-1j],[2.+0j]]],dtype=torch.cdouble)
    h=serial_scan(a,b); re=0.;im=0.; expected=[]
    for n in range(3):
        re,im=a[0,n].real*re-a[0,n].imag*im+b[0,n].real,a[0,n].real*im+a[0,n].imag*re+b[0,n].imag
        expected.append(torch.complex(re,im))
    torch.testing.assert_close(h,torch.stack(expected)[None])


def test_physical_transform_alignment_and_zero():
    t=torch.arange(24);omega=.13
    s=torch.exp(1j*omega*t)[None]
    x=torch.stack((s.real,s.imag),1)
    p=physical_features(x,8)
    torch.testing.assert_close(p[0,:,1],torch.full((3,),math.cos(omega)))
    torch.testing.assert_close(p[0,:,2],torch.full((3,),math.sin(omega)))
    torch.testing.assert_close(p,physical_features(perturb(x,phase=1.7),8),atol=1e-6,rtol=1e-6)
    q=physical_features(perturb(x,frequency=.02),8)
    torch.testing.assert_close(q[...,1]+1j*q[...,2],(p[...,1]+1j*p[...,2])*complex(np.exp(2j*np.pi*.02)),atol=1e-6,rtol=1e-6)
    assert torch.isfinite(physical_features(torch.zeros_like(x),8)).all()
    x[:,:,8:16]*=3
    torch.testing.assert_close(physical_features(x,8)[0,:,0],torch.tensor([0.,math.log(9),0.]),atol=1e-6,rtol=1e-6)


def test_no_cross_token_pairs():
    x=torch.zeros(1,2,16);x[:,0,:8]=1;x[:,0,8:]=-1
    torch.testing.assert_close(physical_features(x,8)[...,1],torch.ones(1,2))


@pytest.mark.parametrize('variant',['S0','S1','S2'])
def test_state_reset_and_trainable_gradients(variant):
    torch.manual_seed(4)
    layer=ComplexSSM(8,6,variant,5)
    u=torch.randn(3,9,8,requires_grad=True);p=torch.randn(3,9,3)
    full=layer(u,p)
    parts=torch.cat([layer(u[i:i+1],p[i:i+1]) for i in range(3)])
    torch.testing.assert_close(full,parts)
    layer(torch.randn_like(u),p)
    torch.testing.assert_close(full,layer(u,p))
    full.square().mean().backward()
    for name,param in layer.named_parameters():
        assert param.grad is not None,name
        assert torch.isfinite(param.grad).all(),name
        assert param.grad.abs().sum()>0,name


def test_parameter_budget_and_metrics():
    a=ComplexSSM(8,6,'S1',5);b=ComplexSSM(8,6,'S2',5)
    assert parameter_count(a)==parameter_count(b)
    scores=metrics([0,0,1,1],[0,1,1,1],2)
    assert scores['oa']==.75
    assert scores['macro_f1']==pytest.approx((2/3+4/5)/2)


def test_model_shape_backward():
    import json
    from pathlib import Path
    c=json.loads((Path(__file__).parents[1]/'configs/base.json').read_text())['model']
    c.update(dim=16,state=8,depth=2,input_length=64,classes=4,selector_hidden=8)
    for variant in ['G','S0','S1','S2','S2_concat']:
        model=Classifier(dict(c,variant=variant));out=model(torch.randn(3,2,64))
        assert out.shape==(3,4)
        torch.nn.functional.cross_entropy(out,torch.tensor([0,1,2])).backward()
        assert torch.isfinite(model.head.weight.grad).all()
