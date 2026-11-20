"""Round-5 resume consistency test (Section 一 of the review task).

Verifies the fit_aux() resume fix end-to-end: a run resumed from a checkpoint must be
bit-identical to an uninterrupted run of the same total length, across model params,
frozen aux stats (aux_mean/aux_std), optimizer state, scheduler/lr, and the data + CFO
RNG streams.

Protocol (smoke, CPU, Q_corr_frozen — the variant that exercises every resume path):
  run A: train fresh --epochs 3, run to completion  -> last.pt at epoch 2
  run B: train fresh --epochs 3, interrupted after epoch 1 -> last.pt at epoch 1
  run C: train --resume --epochs 3 on B's dir (same config) -> last.pt at epoch 2
Then assert ck(A, epoch 2) == ck(C, epoch 2) bit-for-bit. Because Q_corr_frozen carries
aux_mean/aux_std buffers + frozen_hash, this also proves the frozen stats are restored
from the checkpoint and never silently re-fit on resume.

Run:  python -m pytest tests/test_round5_resume.py -v
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_MODULE = 'ssm_exp.engine'
CONFIG = ROOT / 'configs' / 'resolved_round5.json'
MANIFEST = ROOT / 'configs' / 'data_raw.json'
VARIANT = 'Q_corr_frozen'
EPOCHS = 3  # total epochs for the uninterrupted reference and the interrupted run


def _cmd(out, epochs, resume=False):
    cmd = [sys.executable, '-m', ENGINE_MODULE,
           '--config', str(CONFIG), '--manifest', str(MANIFEST),
           '--out', str(out), '--variant', VARIANT, '--smoke', '--epochs', str(epochs),
           '--device', 'cpu', '--workers', '0']
    if resume:
        cmd.append('--resume')
    return cmd


def _run_to_completion(out, epochs):
    r = subprocess.run(_cmd(out, epochs), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'engine failed (fresh epochs={epochs})\n'
                           f'STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}')
    return r


def _run_interrupt_after_epoch(out, epochs, stop_after):
    """Start a fresh run and terminate it right after `stop_after` epoch finishes
    (its checkpoint has already been saved and flushed by then)."""
    proc = subprocess.Popen(_cmd(out, epochs), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    seen = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        seen.append(line)
        try:
            if json.loads(line).get('epoch', -1) >= stop_after:
                break
        except json.JSONDecodeError:
            continue
    proc.terminate()
    proc.wait(timeout=60)
    assert (out / 'last.pt').exists(), f'no checkpoint after interrupt; saw {seen}'


def _load(out):
    return torch.load(out / 'last.pt', map_location='cpu', weights_only=False)


def _byte_equal(a, b):
    """Byte-exact recursive comparison for torch tensors, numpy arrays, and nested
    dict/list/tuple/scalar structures (checkpoint state_dicts and RNG snapshots)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return a.dtype == b.dtype and a.shape == b.shape and bool(np.array_equal(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        return set(a) == set(b) and all(_byte_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(_byte_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_resume_bit_identical_to_uninterrupted_run():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_a = td / 'a'
        run_b = td / 'b'
        _run_to_completion(run_a, EPOCHS)                        # epoch 0,1,2
        _run_interrupt_after_epoch(run_b, EPOCHS, stop_after=1)   # epoch 0,1
        ck_mid = _load(run_b)
        assert ck_mid['epoch'] == 1, ck_mid['epoch']
        # resume the interrupted run to completion (same config: epochs=3)
        r = subprocess.run(_cmd(run_b, EPOCHS, resume=True), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'resume failed\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}')

        ck_a = _load(run_a)   # uninterrupted epoch 2
        ck_b = _load(run_b)   # resumed epoch 2

        assert ck_a['epoch'] == ck_b['epoch'] == 2
        assert ck_a['best'] == ck_b['best']
        assert ck_a['best_epoch'] == ck_b['best_epoch']
        assert ck_a['config'] == ck_b['config']
        assert ck_a['provenance'] == ck_b['provenance']
        assert ck_a['aux'] == ck_b['aux'], 'aux fit stats differ'
        assert ck_a['frozen_hash'] == ck_b['frozen_hash']
        assert _byte_equal(ck_a['model'], ck_b['model']), 'model state_dict differs'
        assert _byte_equal(ck_a['optimizer'], ck_b['optimizer']), 'optimizer differs'
        assert _byte_equal(ck_a['scheduler'], ck_b['scheduler']), 'scheduler differs'
        assert _byte_equal(ck_a['rng'], ck_b['rng']), 'rng streams differ'
        assert _byte_equal(ck_a['augment_rng'], ck_b['augment_rng']), 'augment rng differs'

        # frozen stats bit-exact to the fit-time record, and unchanged by resume
        aux = ck_b['aux']
        assert _byte_equal(ck_b['model']['aux_mean'], torch.tensor(aux['mean']))
        assert _byte_equal(ck_b['model']['aux_std'], torch.tensor(aux['std']))
        model_info = json.loads((run_b / 'model_info.json').read_text())
        assert model_info['aux'] == aux, 'resume overwrote aux in model_info.json'


def test_resume_refuses_missing_aux_stats():
    """A checkpoint without aux buffers must raise, not silently re-fit."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / 'c'
        _run_to_completion(out, EPOCHS)
        ck = _load(out)
        del ck['model']['aux_mean']
        del ck['model']['aux_std']
        ck['aux'] = None
        torch.save(ck, out / 'last.pt')
        r = subprocess.run(_cmd(out, EPOCHS, resume=True), capture_output=True, text=True)
        assert r.returncode != 0, 'resume should have failed on missing aux buffers'
        assert 'refusing to silently refit' in r.stderr or 'lacks frozen buffer' in r.stderr


if __name__ == '__main__':
    test_resume_bit_identical_to_uninterrupted_run()
    test_resume_refuses_missing_aux_stats()
    print('RESUME CONSISTENCY TESTS PASSED')
