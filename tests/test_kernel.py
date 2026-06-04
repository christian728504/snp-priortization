"""Kernel tests: pin the numba port to a pure-numpy reference + hand-computed Δ.

The reference functions below are the original vectorized numpy implementation the
package was first validated against Picard with. The numba kernel must reproduce
them to ~1e-12, so it cannot silently drift.
"""

from __future__ import annotations

import math

import numpy as np

from snp_prioritization.kernel import (
    MAX_PROB_BELOW_ONE,
    block_deltas,
    block_deltas_parallel,
)

CAP = 3.0


# --------------------------------------------------------------------------- #
# Pure-numpy reference (rows = blocks, cols = 3 genotypes)
# --------------------------------------------------------------------------- #
def _gl_from_pl(pl):
    return -pl.astype(np.float64) / 10.0


def _set_loglikelihoods(ll):
    mr = ll - ll.max(axis=1, keepdims=True)
    s = np.power(10.0, mr).sum(axis=1, keepdims=True)
    return mr - np.log10(s)


def _cap(ll_norm, cap):
    capped = np.maximum(ll_norm - ll_norm.max(axis=1, keepdims=True), -cap)
    return _set_loglikelihoods(capped)


def _p_normalize(ll):
    bump = 300.0 - ll.max(axis=1, keepdims=True)
    tmp = np.power(10.0, ll + bump)
    p = tmp / tmp.sum(axis=1, keepdims=True)
    n = ll.shape[1]
    max_p = MAX_PROB_BELOW_ONE
    min_p = (1.0 - MAX_PROB_BELOW_ONE) / (n - 1)
    p = np.where(p > max_p, max_p, p)
    return np.where(p < min_p, min_p, p)


def _capped_likelihoods(pl, swap, cap):
    gl = _gl_from_pl(pl)
    if swap.any():
        gl = gl.copy()
        gl[swap] = gl[swap][:, ::-1]
    return _p_normalize(_cap(_set_loglikelihoods(gl), cap))


def _hwe(maf):
    p = 1.0 - maf
    return np.stack([p * p, 2.0 * p * maf, maf * maf], axis=1)


def _ref_deltas(pl1, pl2, sw1, sw2, maf, cap):
    L1 = _capped_likelihoods(pl1, sw1, cap)
    L2 = _capped_likelihoods(pl2, sw2, cap)
    pi = _hwe(maf)
    no_swap = np.log10((L1 * L2 * pi).sum(axis=1))
    swap = np.log10((L1 * pi).sum(axis=1)) + np.log10((L2 * pi).sum(axis=1))
    return no_swap - swap


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    pl1 = rng.integers(0, 800, size=(n, 3)).astype(np.int64)
    pl2 = rng.integers(0, 800, size=(n, 3)).astype(np.int64)
    sw1 = rng.integers(0, 2, size=n).astype(np.bool_)
    sw2 = rng.integers(0, 2, size=n).astype(np.bool_)
    maf = rng.uniform(0.01, 0.5, size=n)
    return pl1, pl2, sw1, sw2, maf


def test_kernel_matches_numpy_reference():
    pl1, pl2, sw1, sw2, maf = _random_inputs(500, seed=1)
    got = block_deltas(pl1, pl2, sw1, sw2, maf, CAP)
    ref = _ref_deltas(pl1, pl2, sw1, sw2, maf, CAP)
    assert np.allclose(got, ref, atol=1e-12, rtol=0)


def test_parallel_kernel_matches_serial():
    pl1, pl2, sw1, sw2, maf = _random_inputs(500, seed=2)
    a = block_deltas(pl1, pl2, sw1, sw2, maf, CAP)
    b = block_deltas_parallel(pl1, pl2, sw1, sw2, maf, CAP)
    assert np.array_equal(a, b)


def test_cap_limits_extreme_confidence():
    # An arbitrarily confident block is capped; downstream Δ stays bounded.
    pl = np.array([[0, 10000, 20000]], dtype=np.int64)
    sw = np.array([False])
    d = block_deltas(pl, pl, sw, sw, np.array([0.5]), CAP)[0]
    ref = _ref_deltas(pl, pl, sw, sw, np.array([0.5]), CAP)[0]
    assert abs(d - ref) < 1e-12
    assert abs(d) < 2 * CAP  # bounded by the per-block cap


def test_orientation_swap_equals_reversed_pl():
    # Swapping a sample's orientation (REF/ALT reversed vs MAJOR/MINOR) is exactly
    # reversing its PL triple and not swapping.
    pl_a = np.array([[0, 84, 739]], dtype=np.int64)
    pl_b = np.array([[12, 0, 230]], dtype=np.int64)
    maf = np.array([0.3])
    F, T = np.array([False]), np.array([True])
    swapped = block_deltas(pl_a, pl_b, T, F, maf, CAP)[0]
    reversed_pl = block_deltas(pl_a[:, ::-1].copy(), pl_b, F, F, maf, CAP)[0]
    assert abs(swapped - reversed_pl) < 1e-12


def test_hand_computed_single_block():
    # Two confident, opposite homozygotes at MAF=0.5; Δ computed by hand.
    maf = 0.5
    pl1 = np.array([[0, 1000, 2000]], dtype=np.int64)  # hom-major
    pl2 = np.array([[2000, 1000, 0]], dtype=np.int64)  # hom-minor
    sw = np.array([False])
    got = block_deltas(pl1, pl2, sw, sw, np.array([maf]), CAP)[0]
    eps = 1e-3 / (1 + 2e-3)
    main = 1.0 / (1 + 2e-3)
    exp_no = math.log10(main * eps * 0.25 + eps * eps * 0.5 + eps * main * 0.25)
    exp_sw = math.log10(main * 0.25 + eps * 0.5 + eps * 0.25) + math.log10(
        eps * 0.25 + eps * 0.5 + main * 0.25
    )
    assert abs(got - (exp_no - exp_sw)) < 1e-9
