"""Numba per-haplotype-block LOD math.

The hot path of ``pair_lod``: for each haplotype block, turn the two samples' PL
triples into capped, normalized linear likelihoods and compute the per-block LOD
contribution Δ. This is a 1:1 port of the numpy pipeline the package was first
written in (and validated against Picard 3.4.0); ``tests/test_kernel.py`` pins the
port to a pure-numpy reference to ~1e-12 so it cannot drift.

Per block, genotype order [HOM_MAJOR, HET, HOM_MINOR], log10 throughout:
  1. gl   = -PL / 10
  2. swap gl[0]<->gl[2] if the VCF (REF,ALT) is reversed vs map (MAJOR,MINOR)
  3. normalize  (HaplotypeProbabilitiesUsingLogLikelihoods.setLogLikelihoods)
  4. cap at MAX_EFFECT  (CappedHaplotypeProbabilities)
  5. L = pNormalizeLogProbability  (MathUtil, incl. the MAX_PROB_BELOW_ONE clamp)
  6. π = [(1-maf)², 2(1-maf)maf, maf²]   (HaplotypeBlock HWE frequencies)
  7. Δ = log10(Σ L1·L2·π) − log10(Σ L1·π) − log10(Σ L2·π)
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# Picard defaults (CrosscheckFingerprints.java)
MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK = 3.0
# MathUtil.MAX_PROB_BELOW_ONE
MAX_PROB_BELOW_ONE = 0.9999999999999999


@njit(cache=True, inline="always")
def _capped_likelihoods_one(pl0, pl1, pl2, swap, cap, out):
    """PL triple -> capped, pNormalized linear likelihoods (written into out[3]).

    Mirrors fingerprint_lod._capped_likelihoods for a single block:
    setLogLikelihoods -> cap -> pNormalizeLogProbability. `swap` flips genotype
    indices 0<->2 (REF/ALT reversed vs map MAJOR/MINOR).
    """
    # 1. gl = -PL/10
    g0 = -pl0 / 10.0
    g1 = -pl1 / 10.0
    g2 = -pl2 / 10.0
    # 2. orientation swap
    if swap:
        g0, g2 = g2, g0

    # 3. setLogLikelihoods: subtract max, normalize so Σ 10^ll == 1
    m = g0
    if g1 > m:
        m = g1
    if g2 > m:
        m = g2
    n0 = g0 - m
    n1 = g1 - m
    n2 = g2 - m
    s = 10.0**n0 + 10.0**n1 + 10.0**n2
    ls = math.log10(s)
    n0 -= ls
    n1 -= ls
    n2 -= ls

    # 4. cap: subtract max, floor at -cap, then setLogLikelihoods again
    m = n0
    if n1 > m:
        m = n1
    if n2 > m:
        m = n2
    c0 = n0 - m
    c1 = n1 - m
    c2 = n2 - m
    if c0 < -cap:
        c0 = -cap
    if c1 < -cap:
        c1 = -cap
    if c2 < -cap:
        c2 = -cap
    m = c0
    if c1 > m:
        m = c1
    if c2 > m:
        m = c2
    c0 -= m
    c1 -= m
    c2 -= m
    s = 10.0**c0 + 10.0**c1 + 10.0**c2
    ls = math.log10(s)
    c0 -= ls
    c1 -= ls
    c2 -= ls

    # 5. pNormalizeLogProbability: bump near 300, linearize, normalize, clamp
    m = c0
    if c1 > m:
        m = c1
    if c2 > m:
        m = c2
    bump = 300.0 - m
    t0 = 10.0 ** (c0 + bump)
    t1 = 10.0 ** (c1 + bump)
    t2 = 10.0 ** (c2 + bump)
    ts = t0 + t1 + t2
    p0 = t0 / ts
    p1 = t1 / ts
    p2 = t2 / ts
    max_p = MAX_PROB_BELOW_ONE
    min_p = (1.0 - MAX_PROB_BELOW_ONE) / 2.0  # (1 - max) / (n - 1), n = 3
    if p0 > max_p:
        p0 = max_p
    elif p0 < min_p:
        p0 = min_p
    if p1 > max_p:
        p1 = max_p
    elif p1 < min_p:
        p1 = min_p
    if p2 > max_p:
        p2 = max_p
    elif p2 < min_p:
        p2 = min_p

    out[0] = p0
    out[1] = p1
    out[2] = p2


@njit(cache=True)
def _delta_one(pl1, pl2, swap1, swap2, maf, cap, L1, L2):
    """Per-block Δ for one block. L1/L2 are length-3 scratch buffers."""
    _capped_likelihoods_one(pl1[0], pl1[1], pl1[2], swap1, cap, L1)
    _capped_likelihoods_one(pl2[0], pl2[1], pl2[2], swap2, cap, L2)
    p = 1.0 - maf
    pi0 = p * p
    pi1 = 2.0 * p * maf
    pi2 = maf * maf
    no_swap = math.log10(L1[0] * L2[0] * pi0 + L1[1] * L2[1] * pi1 + L1[2] * L2[2] * pi2)
    swap = math.log10(L1[0] * pi0 + L1[1] * pi1 + L1[2] * pi2) + math.log10(
        L2[0] * pi0 + L2[1] * pi1 + L2[2] * pi2
    )
    return no_swap - swap


@njit(cache=True)
def block_deltas(pl1, pl2, swap1, swap2, maf, cap):
    """Per-block Δ for every block (serial). Σ delta == genome-wide LOD.

    pl1/pl2: (N, 3) int genotype PLs; swap1/swap2: (N,) bool; maf: (N,) float;
    cap: scalar (Picard MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK). Returns (N,) float64.
    """
    n = pl1.shape[0]
    delta = np.empty(n, dtype=np.float64)
    L1 = np.empty(3, dtype=np.float64)
    L2 = np.empty(3, dtype=np.float64)
    for i in range(n):
        delta[i] = _delta_one(
            pl1[i], pl2[i], swap1[i], swap2[i], maf[i], cap, L1, L2
        )
    return delta


@njit(cache=True, parallel=True)
def block_deltas_parallel(pl1, pl2, swap1, swap2, maf, cap):
    """prange twin of block_deltas (blocks are independent). Same result."""
    n = pl1.shape[0]
    delta = np.empty(n, dtype=np.float64)
    for i in prange(n):
        L1 = np.empty(3, dtype=np.float64)
        L2 = np.empty(3, dtype=np.float64)
        delta[i] = _delta_one(
            pl1[i], pl2[i], swap1[i], swap2[i], maf[i], cap, L1, L2
        )
    return delta
