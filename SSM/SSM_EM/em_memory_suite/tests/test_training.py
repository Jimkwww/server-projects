"""Tiny CPU integration test: only 24 examples, 2 epochs, 8-dim branches."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from emmem.synthetic import generate
from emmem.train import run
from emmem.evaluate import evaluate_checkpoint
from test_core import SMALL

class Integration(unittest.TestCase):
    def test_pretrain_transfer_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=generate(root/'data',n=24,length=256)
            c=dict(stage='pretrain',seed=17,data=[str(manifest)],model=SMALL,epochs=1,batch_size=12,device='cpu',cpu_threads=1)
            run(c,root/'pre');run(c,root/'pre',resume=True)
            for task,stage in [('waveform','probe'),('parameters','finetune'),('activity','scratch')]:
                dc={**c,'stage':stage,'task':task,'init':str(root/'pre'/'best.pt')}
                if stage=='scratch':dc.pop('init')
                run(dc,root/task)
                result=evaluate_checkpoint(root/task/'best.pt',manifest,task,root/(task+'_eval'))
                self.assertTrue('mse' in result or 'macro_f1' in result)
    def test_exact_epoch_resume(self):
        import emmem.train as t
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=generate(root/'data',n=24,length=256)
            c=dict(stage='pretrain',seed=17,data=[str(manifest)],model=SMALL,epochs=2,batch_size=12,grad_accum=2,device='cpu',cpu_threads=1)
            run(c,root/'continuous');original=t.save
            def interrupted(path,obj):
                original(path,obj)
                if Path(path).name=='last.pt' and obj['epoch']==0:raise RuntimeError('simulated interruption')
            with patch.object(t,'save',side_effect=interrupted):
                with self.assertRaises(RuntimeError):run(c,root/'resumed')
            run(c,root/'resumed',resume=True)
            a=torch.load(root/'continuous'/'last.pt',weights_only=False)['model'];b=torch.load(root/'resumed'/'last.pt',weights_only=False)['model']
            for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
