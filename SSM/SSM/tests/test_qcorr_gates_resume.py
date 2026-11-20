"""Q_corr_gate* resume consistency test (mirrors test_global_qcorr_resume.py).

A run resumed from a checkpoint must be bit-identical to an uninterrupted run of the same
total length (model params, frozen aux stats aux_mean/aux_std, the new gate scalar buffers
gate_c_mean/std + gate_d_mean/std, optimizer, scheduler/lr, data + CFO RNG streams). Because
the gate variants carry both aux buffers and the gate scalar stats, this proves the frozen
standardization is restored from the checkpoint and never silently re-fit on resume.

Run:  python -m pytest tests/test_qcorr_gates_resume.py -v
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
CONFIG = ROOT / 'configs' / 'resolved_qcorr_gates.json'
MANIFEST = ROOT / 'configs' / 'data_raw.json'
VARIANT = 'Q_corr_gate_cd'
EPOCHS = 3


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
        raise RuntimeError(f'engine failed (fresh epochs={epochs})\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}')
    return r


def _run_interrupt_after_epoch(out, epochs, stop_after):
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
        _run_to_completion(run_a, EPOCHS)
        _run_interrupt_after_epoch(run_b, EPOCHS, stop_after=1)
        ck_mid = _load(run_b)
        assert ck_mid['epoch'] == 1, ck_mid['epoch']
        r = subprocess.run(_cmd(run_b, EPOCHS, resume=True), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'resume failed\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}')

        ck_a = _load(run_a)
        ck_b = _load(run_b)
        assert ck_a['epoch'] == ck_b['epoch'] == 2
        assert ck_a['best'] == ck_b['best']
        assert ck_a['best_epoch'] == ck_b['best_epoch']
        assert ck_a['config'] == ck_b['config']
        assert ck_a['provenance'] == ck_b['provenance']
        assert ck_a['aux'] == ck_b['aux']
        assert ck_a['gate_scalars'] == ck_b['gate_scalars']
        assert _byte_equal(ck_a['model'], ck_b['model']), 'model state_dict differs'
        assert _byte_equal(ck_a['optimizer'], ck_b['optimizer']), 'optimizer differs'
        assert _byte_equal(ck_a['scheduler'], ck_b['scheduler']), 'scheduler differs'
        assert _byte_equal(ck_a['rng'], ck_b['rng']), 'rng streams differ'
        assert _byte_equal(ck_a['augment_rng'], ck_b['augment_rng']), 'augment rng differs'

        aux = ck_b['aux']
        gate = ck_b['gate_scalars']
        assert _byte_equal(ck_b['model']['aux_mean'], torch.tensor(aux['mean']))
        assert _byte_equal(ck_b['model']['aux_std'], torch.tensor(aux['std']))
        assert _byte_equal(ck_b['model']['gate_c_mean'], torch.tensor([gate['c_mean']]))
        assert _byte_equal(ck_b['model']['gate_c_std'], torch.tensor([gate['c_std']]))
        assert _byte_equal(ck_b['model']['gate_d_mean'], torch.tensor([gate['d_mean']]))
        assert _byte_equal(ck_b['model']['gate_d_std'], torch.tensor([gate['d_std']]))
        model_info = json.loads((run_b / 'model_info.json').read_text())
        assert model_info['aux'] == aux, 'resume overwrote aux in model_info.json'
        assert model_info['gate_scalars'] == gate, 'resume overwrote gate_scalars in model_info.json'


def test_resume_refuses_missing_gate_scalars():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / 'c'
        _run_to_completion(out, EPOCHS)
        ck = _load(out)
        for nm in ('gate_c_mean', 'gate_c_std', 'gate_d_mean', 'gate_d_std'):
            del ck['model'][nm]
        ck['gate_scalars'] = None
        torch.save(ck, out / 'last.pt')
        r = subprocess.run(_cmd(out, EPOCHS, resume=True), capture_output=True, text=True)
        assert r.returncode != 0, 'resume should have failed on missing gate scalar buffers'
        assert ('refusing to silently refit' in r.stderr) or ('lacks gate scalar stats' in r.stderr)


def test_resume_refuses_missing_aux_stats():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / 'd'
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
    test_resume_refuses_missing_gate_scalars()
    test_resume_refuses_missing_aux_stats()
    print('Q_CORR GATE RESUME CONSISTENCY TESTS PASSED')
