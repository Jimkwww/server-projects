"""Run on SERVER: python -m unittest emfm.tests.test_correctness -v"""
import csv
import io
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from emfm.data import RML2018A, collate
from emfm.models import Attention, Block, Classifier, Merge


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_attention_matches_independent_reference(self):
        a = Attention(8, 2).eval()
        x = torch.randn(2, 5, 8)
        valid = torch.tensor([[True, True, True, False, False], [False] * 5])
        got = a(x, valid)
        q, k, v = a.qkv(x[:1, :3]).reshape(1, 3, 3, 2, 4).permute(2, 0, 3, 1, 4)
        out = ((q @ k.transpose(-1, -2) / 2).softmax(-1) @ v).transpose(1, 2).reshape(1, 3, 8)
        torch.testing.assert_close(got[:1, :3], a.proj(out))
        self.assertTrue(torch.isfinite(got).all())
        self.assertTrue((got[~valid] == 0).all())

    def test_shift_windows_against_slow_reference_and_no_wrap(self):
        b = Block(8, 2, window=4, shift=2).eval()
        x = torch.randn(1, 11, 8)
        mask = torch.ones(1, 11, dtype=torch.bool)
        got = b.attend(x, mask)
        expected = torch.zeros_like(x)
        groups = (torch.arange(11) + 2) // 4
        for g in groups.unique():
            idx = torch.nonzero(groups == g).flatten()
            expected[:, idx] = b.attn(x[:, idx], mask[:, idx])
        torch.testing.assert_close(got, expected)
        changed = x.clone()
        changed[:, -1] += 100
        torch.testing.assert_close(got[:, 0], b.attend(changed, mask)[:, 0])

    def test_merge_order_and_odd_length(self):
        m = Merge(2, 4)
        m.norm = torch.nn.Identity()
        with torch.no_grad():
            m.proj.weight.copy_(torch.eye(4))
        x = torch.arange(6).float().reshape(1, 3, 2)
        y, mask = m(x, torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(y, torch.tensor([[[0., 1., 2., 3.], [4., 5., 0., 0.]]]))
        self.assertTrue(mask.all())

    def test_models_padding_gradients_and_state(self):
        for kind in ('global', 'swin'):
            cfg = dict(kind=kind, dims=[8] if kind == 'global' else [8, 16, 32],
                       depths=[2] if kind == 'global' else [2, 2, 2],
                       heads=[2] if kind == 'global' else [2, 2, 4], patch_size=4, window=4)
            model = Classifier(cfg, classes=3).eval()
            x = torch.randn(1, 13, 2)
            mask = torch.ones(1, 13, dtype=torch.bool)
            one = model(x, mask)
            padded = torch.cat((x, torch.randn(1, 54, 2) * 100), dim=1)
            padded_mask = torch.arange(67)[None, :] < 13
            torch.testing.assert_close(one, model(padded, padded_mask), atol=2e-6, rtol=2e-5)
            one.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            buf.seek(0)
            other = Classifier(cfg, classes=3).eval()
            other.load_state_dict(torch.load(buf, weights_only=True))
            torch.testing.assert_close(one, other(x, mask))

    def test_reader_indices_normalization_and_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            x = np.arange(8 * 7 * 2, dtype=np.float32).reshape(8, 7, 2) - 17
            x[5] = 0
            y = np.eye(3, dtype=np.int64)[np.arange(8) % 3]
            with h5py.File(tmp / 'data.h5', 'w') as f:
                f['X'], f['Y'], f['Z'] = x, y, np.arange(8)[:, None]
            with (tmp / 'manifest.csv').open('w') as f:
                w = csv.writer(f)
                w.writerow(['subset', 'split', 'sample_id', 'class', 'snr'])
                for sid in (7, 1, 5, 3):
                    w.writerow(['smoke', 'train', sid, sid % 3, sid])
                w.writerow(['cls_dev', 'val', 0, 0, 0])
            ds = RML2018A(tmp / 'data.h5', tmp / 'manifest.csv', 'smoke', 'train')
            self.assertEqual(len(ds), 4)
            for i, sid in enumerate((7, 1, 5, 3)):
                r = ds[i]
                np.testing.assert_allclose(r['iq'].numpy() * r['scale'], x[sid], atol=1e-5)
                self.assertEqual(r['sample_id'], sid)
            # Parent has already opened a handle; spawn must still work.
            results = []
            for workers in (0, 2):
                kw = {'multiprocessing_context': 'spawn'} if workers else {}
                batches = list(DataLoader(ds, batch_size=3, num_workers=workers, collate_fn=collate, **kw))
                results.append({k: torch.cat([b[k] for b in batches]) for k in batches[0]})
            for key in results[0]:
                torch.testing.assert_close(results[0][key], results[1][key])
            self.assertEqual(results[0]['sample_id'].tolist(), [7, 1, 5, 3])
            self.assertFalse(results[0]['fs_valid'].any())
            ds.close()


if __name__ == '__main__':
    unittest.main()
