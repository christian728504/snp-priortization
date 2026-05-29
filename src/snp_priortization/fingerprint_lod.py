"""Exact-parity reimplementation of Picard CrosscheckFingerprints' LOD score.

The genome-wide LOD that Picard reports for a pair of samples is a sum of
per-haplotype-block contributions Δᵢ. Picard never writes those Δᵢ out; this
module recomputes them from two ExtractFingerprint VCFs so we can rank the
blocks (SNPs) that most drive a swap call.

Parity is verified against Picard 3.4.0 in test_fingerprint_lod.py
(|Σ Δᵢ − Picard LOD_SCORE| < 1e-3 across many real pairs).

Algorithm (per block; genotype order [HOM_MAJOR, HET, HOM_MINOR]; log10):
  1. gl   = -PL / 10                          (htsjdk GenotypeLikelihoods.getAsVector)
  2. swap gl[0]<->gl[2] if VCF (REF,ALT) is reversed vs map (MAJOR,MINOR)
  3. normalize  (HaplotypeProbabilitiesUsingLogLikelihoods.setLogLikelihoods)
  4. cap at MAX_EFFECT=3.0  (CappedHaplotypeProbabilities, applied to BOTH samples)
  5. L  = getLikelihoods()  (MathUtil.pNormalizeLogProbability, incl. clamp)
  6. π  = [(1-maf)², 2(1-maf)maf, maf²]   (HaplotypeBlock HWE frequencies)
  7. Δ  = log10(Σ L1·L2·π) − log10(Σ L1·π) − log10(Σ L2·π)
A block is included only if both samples have evidence (PL not flat), matching
FingerprintChecker.calculateMatchResults' `hasEvidence()` gate. Σ Δ = LOD.

Verbatim source basis (Picard master, src/main/java/picard/):
  fingerprint/FingerprintChecker.calculateMatchResults
  fingerprint/HaplotypeProbabilities.{shiftedLogEvidenceProbability,...}
  fingerprint/HaplotypeProbabilitiesUsingLogLikelihoods.{setLogLikelihoods,getLikelihoods0}
  fingerprint/CappedHaplotypeProbabilities
  fingerprint/HaplotypeBlock (HWE frequencies)
  util/MathUtil.{subtractMax,capFromBelow,pNormalizeLogProbability}
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pysam

# Picard defaults (CrosscheckFingerprints.java)
MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK = 3.0
# MathUtil.MAX_PROB_BELOW_ONE
MAX_PROB_BELOW_ONE = 0.9999999999999999


# --------------------------------------------------------------------------- #
# Haplotype map
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapEntry:
    name: str
    major: str  # MAJOR_ALLELE (allele1)
    minor: str  # MINOR_ALLELE (allele2)
    maf: float


@functools.lru_cache(maxsize=8)
def read_haplotype_map(path: str) -> dict[tuple[str, int], MapEntry]:
    """Parse a Picard haplotype map into {(chrom, pos): MapEntry}.

    Columns: CHROMOSOME POSITION NAME MAJOR_ALLELE MINOR_ALLELE MAF ANCHOR_SNP PANELS.
    SAM-style '@' lines and the '#'-prefixed column header are skipped.

    Crucially, SNPs are grouped into haplotype (LD) blocks exactly as Picard's
    HaplotypeMap.fromHaplotypeDatabase does: a row is an *anchor* (block start)
    if ANCHOR_SNP is empty or equals its own NAME; the block's MAF — used for the
    HWE prior π — is the anchor's MAF, shared by every SNP in the block. Each
    SNP keeps its OWN MAJOR/MINOR (used for the allele-orientation swap), but its
    own per-row MAF is ignored for the prior if it is a non-anchor.
    """
    rows: list[tuple[str, int, str, str, str, float, str]] = []
    anchor_maf: dict[str, float] = {}  # anchor NAME -> block MAF
    with open(path) as fh:
        for line in fh:
            if line.startswith("@") or line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            chrom, pos, name = f[0], int(f[1]), f[2]
            major, minor, maf = f[3], f[4], float(f[5])
            anchor = f[6].strip() if len(f) > 6 else ""
            rows.append((chrom, pos, name, major, minor, maf, anchor))
            if anchor == "" or anchor == name:  # this row is a block anchor
                anchor_maf[name] = maf

    out: dict[tuple[str, int], MapEntry] = {}
    for chrom, pos, name, major, minor, maf, anchor in rows:
        block_anchor = name if (anchor == "" or anchor == name) else anchor
        block_maf = anchor_maf.get(block_anchor, maf)
        out[(chrom, pos)] = MapEntry(name=name, major=major, minor=minor, maf=block_maf)
    return out


# --------------------------------------------------------------------------- #
# Fingerprint VCF
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VcfRecord:
    ref: str
    alt: str
    pl: tuple[int, int, int]


def read_fingerprint_vcf(path: str) -> dict[tuple[str, int], VcfRecord]:
    """Parse an ExtractFingerprint VCF into {(chrom, pos): VcfRecord}.

    One record per haplotype block (the map's representative SNP). Records
    without PL, or not biallelic SNPs, are skipped.
    """
    out: dict[tuple[str, int], VcfRecord] = {}
    vf = pysam.VariantFile(path)
    sample = list(vf.header.samples)[0]
    for rec in vf:
        if rec.alts is None or len(rec.alts) != 1:
            continue
        ref, alt = rec.ref, rec.alts[0]
        if len(ref) != 1 or len(alt) != 1:
            continue
        pl = rec.samples[sample].get("PL")
        if pl is None or len(pl) != 3:
            continue
        # pysam VariantRecord.pos is the 1-based VCF POS.
        out[(rec.chrom, rec.pos)] = VcfRecord(
            ref=ref, alt=alt, pl=(int(pl[0]), int(pl[1]), int(pl[2]))
        )
    vf.close()
    return out


# --------------------------------------------------------------------------- #
# MathUtil primitives (vectorized; rows = blocks, cols = 3 genotypes)
# --------------------------------------------------------------------------- #
def _set_loglikelihoods(ll: np.ndarray) -> np.ndarray:
    """HaplotypeProbabilitiesUsingLogLikelihoods.setLogLikelihoods (per row).

    subtractMax -> linear -> normalize so Σ 10^ll = 1 per row.
    """
    max_removed = ll - ll.max(axis=1, keepdims=True)
    s = np.power(10.0, max_removed).sum(axis=1, keepdims=True)
    return max_removed - np.log10(s)


def _cap(ll_norm: np.ndarray, cap: float) -> np.ndarray:
    """CappedHaplotypeProbabilities: subtractMax, floor at -cap, renormalize."""
    capped = np.maximum(ll_norm - ll_norm.max(axis=1, keepdims=True), -cap)
    return _set_loglikelihoods(capped)


def _p_normalize_log_probability(ll: np.ndarray) -> np.ndarray:
    """MathUtil.pNormalizeLogProbability (per row), incl. the MAX_PROB_BELOW_ONE clamp."""
    bump = 300.0 - ll.max(axis=1, keepdims=True)
    tmp = np.power(10.0, ll + bump)
    p = tmp / tmp.sum(axis=1, keepdims=True)
    n = ll.shape[1]
    max_p = MAX_PROB_BELOW_ONE
    min_p = (1.0 - MAX_PROB_BELOW_ONE) / (n - 1)
    p = np.where(p > max_p, max_p, p)
    p = np.where(p < min_p, min_p, p)
    return p


def _gl_from_pl(pl: np.ndarray) -> np.ndarray:
    """htsjdk GenotypeLikelihoods.fromPLs(...).getAsVector(): GL = -PL/10 (log10)."""
    return -pl.astype(np.float64) / 10.0


def _capped_likelihoods(pl: np.ndarray, swap: np.ndarray, cap: float) -> np.ndarray:
    """PL rows -> capped, normalized linear likelihoods L (rows x 3).

    `swap[i]` True flips genotype indices 0<->2 for row i (REF/ALT reversed
    vs map MAJOR/MINOR), mirroring addToLogLikelihoods.
    """
    gl = _gl_from_pl(pl)
    if swap.any():
        gl = gl.copy()
        gl[swap] = gl[swap][:, ::-1]
    ll = _set_loglikelihoods(gl)
    ll = _cap(ll, cap)
    return _p_normalize_log_probability(ll)


def hwe_prior(maf: np.ndarray) -> np.ndarray:
    """HaplotypeBlock HWE genotype frequencies: [(1-maf)², 2(1-maf)maf, maf²]."""
    p = 1.0 - maf
    return np.stack([p * p, 2.0 * p * maf, maf * maf], axis=1)


# --------------------------------------------------------------------------- #
# Pairwise LOD
# --------------------------------------------------------------------------- #
@dataclass
class PairLOD:
    total: float
    chrom: np.ndarray  # (N,)
    pos: np.ndarray  # (N,)
    name: np.ndarray  # (N,)
    maf: np.ndarray  # (N,)
    delta: np.ndarray  # (N,) per-block contribution; Σ == total

    def __len__(self) -> int:
        return len(self.delta)


def _orientation_swap(ref: str, alt: str, e: MapEntry) -> bool | None:
    """None => alleles don't match the map at all (Picard would throw)."""
    if ref == e.major and alt == e.minor:
        return False
    if ref == e.minor and alt == e.major:
        return True
    return None


def pair_lod(
    vcf1: str,
    vcf2: str,
    haplotype_map: str,
    cap: float = MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK,
) -> PairLOD:
    """Compute the CrosscheckFingerprints LOD and per-block Δ between two VCFs.

    Returns a PairLOD whose `.total` reproduces Picard's LOD_SCORE and whose
    `.delta` holds per-block contributions (Σ delta == total). Only blocks
    present in the map and in both VCFs, with evidence in both, are included.
    """
    hap = read_haplotype_map(haplotype_map)
    fp1 = read_fingerprint_vcf(vcf1)
    fp2 = read_fingerprint_vcf(vcf2)

    keys = [k for k in hap if k in fp1 and k in fp2]
    keys.sort()

    chroms, poss, names, mafs = [], [], [], []
    pl1, pl2, sw1, sw2 = [], [], [], []
    for k in keys:
        e = hap[k]
        r1, r2 = fp1[k], fp2[k]
        # evidence gate: a flat PL carries no information (hasEvidence == false)
        if len(set(r1.pl)) == 1 or len(set(r2.pl)) == 1:
            continue
        s1 = _orientation_swap(r1.ref, r1.alt, e)
        s2 = _orientation_swap(r2.ref, r2.alt, e)
        if s1 is None or s2 is None:
            # Alleles disagree with the map; Picard would throw. Skip + ignore.
            continue
        chroms.append(k[0])
        poss.append(k[1])
        names.append(e.name)
        mafs.append(e.maf)
        pl1.append(r1.pl)
        pl2.append(r2.pl)
        sw1.append(s1)
        sw2.append(s2)

    if not poss:
        return PairLOD(
            0.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        )

    pl1 = np.array(pl1, dtype=np.int64)
    pl2 = np.array(pl2, dtype=np.int64)
    sw1 = np.array(sw1, dtype=bool)
    sw2 = np.array(sw2, dtype=bool)
    maf = np.array(mafs, dtype=np.float64)

    L1 = _capped_likelihoods(pl1, sw1, cap)
    L2 = _capped_likelihoods(pl2, sw2, cap)
    pi = hwe_prior(maf)

    no_swap = np.log10((L1 * L2 * pi).sum(axis=1))
    swap = np.log10((L1 * pi).sum(axis=1)) + np.log10((L2 * pi).sum(axis=1))
    delta = no_swap - swap

    return PairLOD(
        total=float(delta.sum()),
        chrom=np.array(chroms),
        pos=np.array(poss),
        name=np.array(names),
        maf=maf,
        delta=delta,
    )


def rank_blocks(result: PairLOD, direction: str = "swap", top: int | None = None):
    """Rank blocks by evidential contribution.

    direction:
      'swap'      -> most-negative Δ first (strongest different-individual evidence)
      'match'     -> most-positive Δ first (strongest same-individual evidence)
      'magnitude' -> largest |Δ| first
    Returns a list of (chrom, pos, name, maf, delta) tuples.
    """
    if direction == "swap":
        order = np.argsort(result.delta)
    elif direction == "match":
        order = np.argsort(result.delta)[::-1]
    elif direction == "magnitude":
        order = np.argsort(np.abs(result.delta))[::-1]
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    if top is not None:
        order = order[:top]
    return [
        (
            result.chrom[i],
            int(result.pos[i]),
            result.name[i],
            float(result.maf[i]),
            float(result.delta[i]),
        )
        for i in order
    ]
