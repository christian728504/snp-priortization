"""End-to-end tests for pair_lod + rank_blocks on the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from snp_prioritization import PairLOD, pair_lod, rank_blocks
from snp_prioritization.lod import _orientation_swap
from snp_prioritization.io import MapEntry

FIX = Path(__file__).resolve().parent / "fixtures"
MAP = str(FIX / "mini.map")


def _ab():
    return pair_lod(str(FIX / "A.vcf"), str(FIX / "B.vcf"), MAP)


def test_sum_of_deltas_equals_total():
    res = _ab()
    assert np.isclose(res.delta.sum(), res.total, atol=1e-12)


def test_no_evidence_blocks_excluded():
    # pos 4000 has a flat PL in A; pos 12000 flat in A -> both dropped from A-vs-*.
    res = _ab()
    positions = set(int(p) for p in res.pos)
    assert 4000 not in positions
    assert 12000 not in positions
    # 13 blocks minus the 2 flat-in-A blocks
    assert len(res) == 11


def test_parallel_matches_serial():
    a = pair_lod(str(FIX / "A.vcf"), str(FIX / "C.vcf"), MAP)
    b = pair_lod(str(FIX / "A.vcf"), str(FIX / "C.vcf"), MAP, parallel=True)
    assert np.isclose(a.total, b.total, atol=1e-12)


def test_rank_directions_are_sorted():
    res = _ab()
    swap = [t[4] for t in rank_blocks(res, "swap")]
    assert swap == sorted(swap)  # most-negative first
    match = [t[4] for t in rank_blocks(res, "match")]
    assert match == sorted(match, reverse=True)  # most-positive first
    mag = [abs(t[4]) for t in rank_blocks(res, "magnitude")]
    assert mag == sorted(mag, reverse=True)


def test_rank_top_limits_and_tuple_shape():
    res = _ab()
    top = rank_blocks(res, "swap", top=3)
    assert len(top) == 3
    chrom, pos, name, maf, delta = top[0]
    assert isinstance(pos, int) and isinstance(maf, float) and isinstance(delta, float)


def test_rank_rejects_unknown_direction():
    import pytest

    with pytest.raises(ValueError):
        rank_blocks(_ab(), "sideways")


def test_empty_when_no_shared_blocks(tmp_path):
    # A map with a position no VCF has -> empty result, total 0.
    m = tmp_path / "empty.map"
    m.write_text(
        "@HD\tVN:1.5\n@SQ\tSN:chr9\tLN:1000\n"
        "#CHROMOSOME\tPOSITION\tNAME\tMAJOR_ALLELE\tMINOR_ALLELE\tMAF\tANCHOR_SNP\tPANELS\n"
        "chr9\t500\tchr9:1\tA\tG\t0.3\t\t\n"
    )
    res = pair_lod(str(FIX / "A.vcf"), str(FIX / "B.vcf"), str(m))
    assert isinstance(res, PairLOD)
    assert len(res) == 0 and res.total == 0.0


def test_orientation_swap_helper():
    e = MapEntry(name="x", major="T", minor="G", maf=0.3)
    assert _orientation_swap("T", "G", e) is False
    assert _orientation_swap("G", "T", e) is True
    assert _orientation_swap("A", "C", e) is None
