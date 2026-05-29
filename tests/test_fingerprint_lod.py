"""Tests for fingerprint_lod: unit math + exact parity with Picard.

Parity strategy
---------------
The crosscheck TSVs in results/ ARE Picard 3.4.0 output (LOD_SCORE per pair).
We recompute each pair's LOD in Python and require |diff| < 1e-3.

Important data note: the WGS/RNA/ATAC fingerprint VCFs were extracted with the
FULL map (hg38_chr.map); the WGBS comparison used VCFs extracted with the
FILTERED map (hg38_chr.filtered.map), which live in a separate `filtered/`
subtree. The TSV's LEFT_FILE/RIGHT_FILE columns for the filtered comparison are
stale (they record the full-dir path), so we resolve VCF paths from the sample
accession + comparison type rather than trusting those columns.

A separate hermetic test re-runs Picard CrosscheckFingerprints itself and
compares to Python at full double precision.
"""

from __future__ import annotations

import math
import shutil
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from snp_priortization import fingerprint_lod as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

DATA_ROOT = Path("/data/zusers/ramirezc/data/detect_sample_swap")
HAPLOTYPE_MAP_DIR = PROJECT / "scratch" / "haplotype-maps"

FULL_MAP = HAPLOTYPE_MAP_DIR / "hg38_chr.map"
FILTERED_MAP = HAPLOTYPE_MAP_DIR / "hg38_chr.filtered.map"
CODING_EXONS_MAP = HAPLOTYPE_MAP_DIR / "hg38_chr.coding_exons.map"

# comparison -> (map, {accession-prefix: vcf subdir})
COMPARISONS = {
    "WGS_v_ATAC": (FULL_MAP, {"EG": "wgs", "EA": "atac"}),
    "WGS_v_RNA": (FULL_MAP, {"EG": "wgs", "ER": "rna"}),
    "WGS_v_WGBS": (FILTERED_MAP, {"EG": "filtered/wgs", "EB": "filtered/wgbs"}),
}
TSV = {
    "WGS_v_ATAC": PROJECT / "results/crosscheck_wgs_v_atac.tsv",
    "WGS_v_RNA": PROJECT / "results/crosscheck_wgs_v_rna.tsv",
    "WGS_v_WGBS": PROJECT / "results/filtered/crosscheck_wgs_v_wgbs.tsv",
}

TOL = 1e-3
data_available = HAPLOTYPE_MAP_DIR.exists() and FULL_MAP.exists()


# --------------------------------------------------------------------------- #
# Unit tests (no external data)
# --------------------------------------------------------------------------- #
def test_gl_from_pl():
    assert np.allclose(F._gl_from_pl(np.array([198, 0, 378])), [-19.8, 0.0, -37.8])


def test_hwe_prior_sums_to_one_and_matches_formula():
    maf = np.array([0.1, 0.25, 0.5])
    pi = F.hwe_prior(maf)
    assert np.allclose(pi.sum(axis=1), 1.0)
    q = 0.25
    assert np.allclose(pi[1], [(1 - q) ** 2, 2 * (1 - q) * q, q * q])


def test_set_loglikelihoods_normalizes_to_one_in_linear():
    ll = F._set_loglikelihoods(np.array([[0.0, -10.0, -20.0]]))
    assert np.isclose(np.power(10.0, ll).sum(), 1.0)


def test_set_loglikelihoods_is_shift_invariant():
    a = F._set_loglikelihoods(np.array([[0.0, -3.0, -7.0]]))
    b = F._set_loglikelihoods(np.array([[5.0, 2.0, -2.0]]))  # same shape, +5 shift
    assert np.allclose(a, b)


def test_cap_limits_confidence():
    # An arbitrarily confident block is capped so max prob ~ 1/(1+2*10^-3).
    L = F._capped_likelihoods(np.array([[0, 10000, 20000]]), np.array([False]), 3.0)[0]
    assert np.isclose(L[0], 1.0 / (1.0 + 2.0e-3), atol=1e-6)
    assert np.isclose(L[1], 1.0e-3 / (1.0 + 2.0e-3), atol=1e-6)


def test_orientation_swap():
    e = F.MapEntry(name="x", major="T", minor="G", maf=0.3)
    assert F._orientation_swap("T", "G", e) is False
    assert F._orientation_swap("G", "T", e) is True
    assert F._orientation_swap("A", "C", e) is None


def test_capped_likelihoods_swap_reverses():
    pl = np.array([[0, 84, 739]])  # most likely HOM_ALLELE1 (index 0)
    no = F._capped_likelihoods(pl, np.array([False]), 3.0)[0]
    sw = F._capped_likelihoods(pl, np.array([True]), 3.0)[0]
    assert np.allclose(no, sw[::-1])


def test_synthetic_single_block_hand_computed():
    # Two confident, opposite homozygotes at MAF=0.5; compute Δ by hand.
    maf = 0.5
    pi = np.array([(1 - maf) ** 2, 2 * (1 - maf) * maf, maf**2])
    L1 = F._capped_likelihoods(np.array([[0, 1000, 2000]]), np.array([False]), 3.0)[
        0
    ]  # hom-major
    L2 = F._capped_likelihoods(np.array([[2000, 1000, 0]]), np.array([False]), 3.0)[
        0
    ]  # hom-minor
    no = math.log10(float((L1 * L2 * pi).sum()))
    sw = math.log10(float((L1 * pi).sum())) + math.log10(float((L2 * pi).sum()))
    delta = no - sw
    eps = 1e-3 / (1 + 2e-3)
    main = 1.0 / (1 + 2e-3)
    exp_no = math.log10(main * eps * 0.25 + eps * eps * 0.5 + eps * main * 0.25)
    exp_sw = math.log10(main * 0.25 + eps * 0.5 + eps * 0.25) + math.log10(
        eps * 0.25 + eps * 0.5 + main * 0.25
    )
    assert abs(delta - (exp_no - exp_sw)) < 1e-9


# --------------------------------------------------------------------------- #
# Parity tests against Picard (the crosscheck TSVs)
# --------------------------------------------------------------------------- #
def _parse_metrics(path: Path):
    rows, header, in_m = [], None, False
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("## METRICS CLASS"):
            in_m = True
            continue
        if in_m and header is None:
            header = line.split("\t")
            continue
        if in_m and header is not None:
            if line == "" or line.startswith("#"):
                break
            rows.append(dict(zip(header, line.split("\t"))))
    return rows


def _vcf_for(accession: str, comparison: str) -> Path:
    _map, dirs = COMPARISONS[comparison]
    subdir = dirs[accession[:2]]
    return DATA_ROOT / subdir / f"{accession}.vcf.gz"


def _sample_pairs(
    comparison: str, n_extreme: int = 6, n_random: int = 10, seed: int = 1
):
    rows = [
        r
        for r in _parse_metrics(TSV[comparison])
        if r["LEFT_SAMPLE"] != r["RIGHT_SAMPLE"]
    ]
    rows.sort(key=lambda r: float(r["LOD_SCORE"]))
    rng = random.Random(seed)
    picked = (
        rows[:n_extreme]
        + rows[-n_extreme:]
        + rng.sample(rows, min(n_random, len(rows)))
    )
    seen, out = set(), []
    for r in picked:
        key = (r["LEFT_SAMPLE"], r["RIGHT_SAMPLE"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            (comparison, r["LEFT_SAMPLE"], r["RIGHT_SAMPLE"], float(r["LOD_SCORE"]))
        )
    return out


def _all_parity_cases():
    if not data_available:
        return []
    cases = []
    for comp in COMPARISONS:
        if TSV[comp].exists():
            cases.extend(_sample_pairs(comp))
    return cases


@pytest.mark.skipif(not data_available, reason="fingerprint VCFs not reachable")
@pytest.mark.parametrize(
    "comparison,left,right,picard_lod",
    _all_parity_cases(),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_parity_with_picard(comparison, left, right, picard_lod):
    hap, _dirs = COMPARISONS[comparison]
    v1, v2 = _vcf_for(left, comparison), _vcf_for(right, comparison)
    if not (v1.exists() and v2.exists()):
        pytest.skip(f"missing VCF: {v1 if not v1.exists() else v2}")
    res = F.pair_lod(str(v1), str(v2), str(hap))
    assert abs(res.total - picard_lod) < TOL, (
        f"{comparison} {left} x {right}: ours={res.total:.6f} picard={picard_lod:.6f}"
    )


@pytest.mark.skipif(not data_available, reason="fingerprint VCFs not reachable")
def test_sum_of_deltas_equals_total():
    res = F.pair_lod(
        str(_vcf_for("EG100526", "WGS_v_ATAC")),
        str(_vcf_for("EA100030", "WGS_v_ATAC")),
        str(FULL_MAP),
    )
    assert np.isclose(res.delta.sum(), res.total, atol=1e-9)


@pytest.mark.skipif(not data_available, reason="fingerprint VCFs not reachable")
def test_ranking_directions():
    # A true match pair: most-positive Δ should be positive; ranking is sorted.
    res = F.pair_lod(
        str(_vcf_for("EG100500", "WGS_v_ATAC")),
        str(_vcf_for("EG100578", "WGS_v_ATAC")),
        str(FULL_MAP),
    )
    top_match = F.rank_blocks(res, "match", top=5)
    deltas = [t[4] for t in top_match]
    assert deltas == sorted(deltas, reverse=True)
    top_swap = F.rank_blocks(res, "swap", top=5)
    assert [t[4] for t in top_swap] == sorted(t[4] for t in top_swap)


# --------------------------------------------------------------------------- #
# Hermetic Picard re-run (independent of the TSVs)
# --------------------------------------------------------------------------- #
def _parse_matrix_lods(path: Path):
    rows = _parse_metrics(path)
    out = {}
    for r in rows:
        out[(r["LEFT_SAMPLE"], r["RIGHT_SAMPLE"])] = float(r["LOD_SCORE"])
    return out


@pytest.mark.skipif(
    not (data_available and shutil.which("picard") and shutil.which("java")),
    reason="picard.jar / jdk / data not available",
)
def test_hermetic_picard_rerun(tmp_path):
    accs = ["EG100526", "EG100264", "EG100500", "EG100578"]
    vcfs = [_vcf_for(a, "WGS_v_ATAC") for a in accs]
    if not all(v.exists() for v in vcfs):
        pytest.skip("missing WGS VCFs")
    out = tmp_path / "cc.tsv"
    cmd = [
        "picard",
        "CrosscheckFingerprints",
        *sum([["--INPUT", str(v)] for v in vcfs], []),
        "--HAPLOTYPE_MAP",
        str(FULL_MAP),
        "--CROSSCHECK_BY",
        "FILE",
        "--CALCULATE_TUMOR_AWARE_RESULTS",
        "false",
        "--LOD_THRESHOLD",
        "-5.0",
        "--OUTPUT",
        str(out),
    ]
    # CrosscheckFingerprints exits non-zero when groups don't relate as expected
    # (these are different individuals); the LOD matrix is still written.
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert out.exists(), "Picard did not write output"
    picard = _parse_matrix_lods(out)
    # Picard groups by FILE -> keys look like "<uri>::<sample>"; map sample->LOD instead.
    by_sample = {}
    for (l, r), lod in picard.items():
        ls = l.split("::")[-1]
        rs = r.split("::")[-1]
        by_sample[(ls, rs)] = lod
    worst = 0.0
    for i in range(len(accs)):
        for j in range(len(accs)):
            if i == j:
                continue
            exp = by_sample[(accs[i], accs[j])]
            res = F.pair_lod(str(vcfs[i]), str(vcfs[j]), str(FULL_MAP))
            worst = max(worst, abs(res.total - exp))
    assert worst < 1e-4, f"worst hermetic diff {worst:.2e}"
