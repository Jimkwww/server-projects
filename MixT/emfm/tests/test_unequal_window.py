"""Correctness checks for unequal-window attention (brief §6.3).

Run on SERVER from the MixT root:  python -m unittest emfm.tests.test_unequal_window -v
"""
import unittest

import torch

from emfm.models import Classifier
from emfm.unequal_window_models import (
    UnequalWindowBlock,
    UnequalWindowClassifier,
    allowed_matrix,
    rotate_right,
    window_intervals,
)


class UnequalWindow(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_rotation_and_partition_geometry(self):
        w = [3, 5, 7, 13]
        self.assertEqual(rotate_right(w, 1), [13, 3, 5, 7])
        self.assertEqual(rotate_right(w, 2), [7, 13, 3, 5])
        self.assertEqual(rotate_right(w, 3), [5, 7, 13, 3])
        self.assertEqual(rotate_right(w, 0), [3, 5, 7, 13])
        self.assertEqual(rotate_right(w, 4), [3, 5, 7, 13])
        n = sum(w)
        iv = window_intervals(w, n)
        self.assertEqual(iv[0][0], 0)
        self.assertEqual(iv[-1][1], n)
        self.assertTrue(all(iv[i][1] == iv[i + 1][0] for i in range(len(iv) - 1)))
        a = allowed_matrix(w, n)
        self.assertTrue(a.diag().all())
        self.assertTrue(torch.equal(a, a.T))
        self.assertTrue(a[0, 2])       # same window
        self.assertFalse(a[0, 3])      # across window boundary
        with self.assertRaises(ValueError):
            window_intervals([3, 5, 7], n)

    def test_single_window_equals_global(self):
        g = Classifier(dict(kind="global", dims=[176], depths=[6], heads=[4],
                            patch_size=8, window=8, shifted=True, ratio=4,
                            dropout=0.0), classes=3).eval()
        u = UnequalWindowClassifier(dict(kind="unequal_window", dims=[176],
                                         depths=[6], heads=[4], patch_size=8,
                                         window_list=[128], rotation="cyclic",
                                         window_impl="dense_mask", ratio=4,
                                         dropout=0.0), classes=3).eval()
        u.load_state_dict(g.state_dict())
        x = torch.randn(2, 1024, 2)
        valid = torch.ones(2, 1024, dtype=torch.bool)
        torch.testing.assert_close(u(x, valid), g(x, valid), atol=1e-5, rtol=1e-5)
        # gradients also match (dropout off, eval)
        x2 = x.clone().requires_grad_(True)
        u(x2, valid).square().sum().backward()
        self.assertTrue(torch.isfinite(x2.grad).all())

    def test_dense_mask_equals_per_window(self):
        base = dict(kind="unequal_window", dims=[176], depths=[6], heads=[4],
                    patch_size=8, window_list=[4, 8, 16, 24, 32, 44],
                    rotation="cyclic", ratio=4, dropout=0.0)
        m_dense = UnequalWindowClassifier(dict(base, window_impl="dense_mask"), classes=3).eval()
        m_win = UnequalWindowClassifier(dict(base, window_impl="per_window"), classes=3).eval()
        m_win.load_state_dict(m_dense.state_dict())
        x1 = torch.randn(2, 1024, 2, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        valid = torch.ones(2, 1024, dtype=torch.bool)
        o1 = m_dense(x1, valid)
        o2 = m_win(x2, valid)
        torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)
        o1.square().sum().backward()
        o2.square().sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-6, rtol=1e-6)
        for (n1, p1), (n2, p2) in zip(m_dense.named_parameters(), m_win.named_parameters()):
            self.assertEqual(n1, n2)
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-6, rtol=1e-6)

    def test_locality_single_block(self):
        dim, heads, n = 16, 2, 12
        wl = [3, 4, 5]
        intervals = window_intervals(wl, n)
        allowed = allowed_matrix(wl, n)
        blk = UnequalWindowBlock(dim, heads, impl="dense_mask",
                                 allowed=allowed, intervals=intervals).eval()
        x = torch.randn(1, n, dim)
        valid = torch.ones(1, n, dtype=torch.bool)
        out = blk(x, valid)
        x2 = x.clone()
        x2[:, 10] += 100            # token 10 is in window [7,12), token 4 in [3,7)
        out2 = blk(x2, valid)
        torch.testing.assert_close(out[:, 4], out2[:, 4])

    def test_finite_gradients_and_no_masked_rows(self):
        cfg = dict(kind="unequal_window", dims=[176], depths=[6], heads=[4],
                   patch_size=8, window_list=[4, 8, 16, 24, 32, 44],
                   rotation="cyclic", window_impl="dense_mask", ratio=4, dropout=0.0)
        m = UnequalWindowClassifier(cfg, classes=3).eval()
        for stage in m.encoder.stages:
            for blk in stage:
                allowed = blk.allowed
                self.assertTrue(allowed.diag().all())
                self.assertTrue((allowed.sum(1) > 0).all())
                self.assertTrue((allowed.sum(0) > 0).all())
        x = torch.randn(2, 1024, 2)
        valid = torch.ones(2, 1024, dtype=torch.bool)
        o = m(x, valid)
        self.assertTrue(torch.isfinite(o).all())
        o.sum().backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
