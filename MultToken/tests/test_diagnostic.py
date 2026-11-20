"""Tests for the Phase-0 reconstruction / boundary / freq-order / IQ-pairing checks."""
from mtk import diagnostic as D


def test_wpt_inverse_within_tolerance():
    r = D.check_wpt_inverse("haar", 2)
    assert r["critical_sampling"] is True
    assert r["rel_l2_float32"] < 1e-5
    r2 = D.check_wpt_inverse("db4", 2)
    assert r2["rel_l2_float32"] < 1e-5


def test_dwt_inverse_within_tolerance():
    r = D.check_dwt_inverse("db4", 3)
    assert r["rel_l2_float32"] < 1e-5


def test_stft_inverse_within_tolerance():
    r = D.check_stft_inverse()
    assert r["rel_l2_complex"] < 1e-5


def test_wpt_low_freq_in_natural_leaf_0():
    r = D.check_wpt_freq_order("haar", 2)
    assert r["low_freq_in_natural_leaf_0"] is True


def test_iq_pairing_same_transform():
    r = D.check_iq_pairing("haar", 2)
    assert r["i_of_swap_equals_q"] < 1e-6
    assert r["q_of_swap_equals_i"] < 1e-6


def test_all_tokenizers_emit_valid_batch():
    r = D.check_token_shapes()
    assert set(r) == {"t0", "t1", "t2", "t3-haar", "t3-db4", "t4", "t5"}
