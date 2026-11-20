import argparse
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from ssm_exp import engine

def fixture(tmp_path):
    rng=np.random.default_rng(5)
    np.save(tmp_path/'x.npy',rng.normal(size=(32,32,2)).astype('float32'))
    np.save(tmp_path/'y.npy',np.arange(32)%4);np.save(tmp_path/'snr.npy',np.arange(32)%3-1)
    np.save(tmp_path/'train.npy',np.arange(24));np.save(tmp_path/'val.npy',np.arange(24,32))
    m=dict(format='npy',x='x.npy',y='y.npy',snr='snr.npy',layout='NLC',label_format='integer',
           normalization='none',classes=4,splits=dict(train='train.npy',val='val.npy'))
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    c=json.loads((Path(__file__).parents[1]/'configs/base.json').read_text())
    c['model'].update(dim=8,depth=1,state=4,selector_hidden=4,input_length=32,classes=4,variant='S2',dropout=.1)
    c['train'].update(epochs=3,batch_size=8,physical_fit_samples=20,diagnostic_every=2,augment_cfo_max=.002)
    (tmp_path/'config.json').write_text(json.dumps(c))
    return argparse.Namespace(config=str(tmp_path/'config.json'),manifest=str(tmp_path/'manifest.json'),
        out=str(tmp_path/'full'),variant=None,seed=None,augment_max=None,epochs=None,smoke=True,
        resume=False,device='cpu',workers=0)


def test_checkpoint_resume_exact(tmp_path,monkeypatch):
    args=fixture(tmp_path);engine.train(args)
    expected=torch.load(Path(args.out)/'last.pt',weights_only=False,map_location='cpu')
    args.out=str(tmp_path/'interrupted')
    original=engine.evaluate;calls=0
    def interrupt(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated interruption')
        return original(*a,**kw)
    monkeypatch.setattr(engine,'evaluate',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):engine.train(args)
    monkeypatch.setattr(engine,'evaluate',original);args.resume=True;engine.train(args)
    actual=torch.load(Path(args.out)/'last.pt',weights_only=False,map_location='cpu')
    for key in expected['model']:torch.testing.assert_close(expected['model'][key],actual['model'][key],rtol=0,atol=0)
    assert expected['best']==actual['best']


def test_formal_training_guard(tmp_path):
    args=fixture(tmp_path);args.smoke=False
    with pytest.raises(ValueError,match='audit'):engine.train(args)
