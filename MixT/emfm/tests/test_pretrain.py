"""SERVER-only correctness and end-to-end resume tests, no real data needed."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import h5py
import numpy as np
import torch
from emfm.data import sha256
from emfm.models import Classifier
from emfm.masked_model import MaskedIQModel, make_patch_mask, masked_error, load_pretrained_encoder
from emfm.prepare_pretrain import prepare
from emfm.pretrain_data import load_bundle, PretrainIQ


class PretrainCorrectness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_fixed_validation_masks_are_batch_order_independent(self):
        valid = torch.ones(3, 39, dtype=torch.bool)
        valid[0, 17:] = False
        ids = torch.tensor([97, 2, 45])
        a = make_patch_mask(valid, 4, .5, sample_ids=ids)
        order = torch.tensor([2,0,1])
        b = make_patch_mask(valid[order], 4, .5, sample_ids=ids[order])
        torch.testing.assert_close(a[order], b)
        self.assertEqual(a.sum(1).tolist(), [2,5,5])
        self.assertFalse(a[0, 5:].any())
        c = make_patch_mask(valid[:1], 4, .5, sample_ids=ids[:1])
        torch.testing.assert_close(a[:1], c)

    def test_masked_loss_excludes_visible_and_padding(self):
        target = torch.ones(1, 7, 2)
        prediction = torch.zeros_like(target, requires_grad=True)
        valid = torch.tensor([[True]*5+[False]*2])
        mask = torch.tensor([[False, True]])
        error, count, zero = masked_error(prediction, target, valid, mask, 4)
        self.assertEqual(int(count), 2)
        self.assertEqual(float(error), 2)
        self.assertEqual(float(zero), 2)
        (error/count).backward()
        self.assertEqual(prediction.grad.nonzero()[:,1].unique().tolist(), [4])

    def test_no_hidden_patch_leakage_and_encoder_transfer(self):
        for kind in ['global','swin']:
            cfg = {'kind':kind,'dims':[8] if kind=='global' else [8,16],
                   'depths':[2] if kind=='global' else [2,2],
                   'heads':[2] if kind=='global' else [2,4], 'patch_size':4,'window':4}
            model = MaskedIQModel(cfg).eval()
            x, valid = torch.randn(2, 37, 2), torch.ones(2,37,dtype=torch.bool)
            mask = make_patch_mask(valid,4,.5,sample_ids=[10,20])
            changed = x.clone()
            points = mask.repeat_interleave(4,1)[:,:37]
            changed[points] = 1000 * torch.randn_like(changed[points])
            pred = model(x,valid,mask)
            torch.testing.assert_close(pred,model(changed,valid,mask))
            loss,n,_ = masked_error(pred,x,valid,mask,4)
            (loss/n).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            classifier = Classifier(cfg,3)
            original_head = classifier.head.weight.detach().clone()
            ckpt = {'format':'emfm_masked_iq_v1','metadata':{'stage':'dev','config':{'model':cfg,'data':{'normalization':'max_abs'}},'classification_manifest_sha256':'fixture'},'encoder':model.encoder.state_dict()}
            load_pretrained_encoder(classifier,ckpt,cfg,'max_abs','fixture')
            for k,v in model.encoder.state_dict().items():
                torch.testing.assert_close(v,classifier.encoder.state_dict()[k])
            torch.testing.assert_close(original_head,classifier.head.weight)

    def test_data_exclusions_cache_and_resumed_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            x=np.random.default_rng(5).normal(size=(40,1024,2)).astype(np.float32)
            x[3]=x[1]  # duplicate of downstream val, must be excluded
            with h5py.File(root/'data.h5','w') as f:
                f['X']=x; f['Y']=np.eye(3,dtype=np.int64)[np.arange(40)%3]
                f['Z']=np.full((40,1),10)
            with (root/'cls.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['subset','split','sample_id','class','snr'])
                for sid,split in enumerate(['train','val','test']):
                    w.writerow(['cls_dev',split,sid,sid,10])
            digest=sha256(root/'cls.csv')
            bundle=prepare(root/'data.h5',root/'cls.csv',root/'data',8,4,4,2,2024,digest)
            _,rows=load_bundle(root/'data/dataset.json')
            tr={r['sample_id'] for r in rows['train_dev']}; va={r['sample_id'] for r in rows['val_dev']}
            self.assertFalse(tr & {1,2,3});self.assertFalse(va & {0,1,2,3});self.assertFalse(tr&va)
            streamed=PretrainIQ(root/'data.h5',rows['train_dev'],False)
            cached=PretrainIQ(root/'data.h5',rows['train_dev'],True)
            for i in range(len(streamed)):
                torch.testing.assert_close(streamed[i]['iq'],cached[i]['iq'])
                self.assertNotIn('label',cached[i])
            streamed.close()
            cfg={'model':{'kind':'swin','dims':[8,16],'depths':[2,2],'heads':[2,4], 'patch_size':8,'window':8,'dropout':0.0},
                 'data':{'normalization':'max_abs','classification_manifest_sha256':digest},
                 'objective':{'mask_ratio':.5,'validation_mask_seed':32024},
                 'training':{'epochs':{'smoke':2,'dev':2},'batch_size':4,'lr':.0003,'weight_decay':.01,'warmup_fraction':.05,'grad_clip':1.,'precision':'fp32'}}
            (root/'cfg.json').write_text(json.dumps(cfg))
            cmd=[sys.executable,'-m','emfm.pretrain','--config',str(root/'cfg.json'),'--h5',str(root/'data.h5'),
                 '--bundle',str(root/'data/dataset.json'),'--stage','dev','--device','cpu']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            def run(args):
                result=subprocess.run(cmd+args,text=True,capture_output=True,env=env)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            run(['--output',str(root/'full')])
            run(['--output',str(root/'resume'),'--stop-after-epochs','1'])
            run(['--output',str(root/'resume'),'--resume',str(root/'resume/last.pt')])
            a=torch.load(root/'full/last.pt',weights_only=False,map_location='cpu')
            b=torch.load(root/'resume/last.pt',weights_only=False,map_location='cpu')
            self.assertEqual(a['step'],b['step'])
            for k,v in a['model'].items():
                torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
