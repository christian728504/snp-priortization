"""Pairwise LOD: parse two fingerprint VCFs, build per-block arrays, score.

The genome-wide LOD that Picard CrosscheckFingerprints reports for a pair of
samples is a sum of per-haplotype-block contributions Δᵢ. Picard never writes
those out; ``pair_lod`` recomputes them so we can rank the blocks (SNPs) that most
drive a swap call. Parity with Picard 3.4.0 is verified in tests/.

The per-block math lives in :mod:`snp_prioritization.kernel` (numba); this module
handles I/O, the map intersection, the evidence/orientation gates, and assembly.
The block intersection is a columnar polars join (struct-of-arrays), so there is no
Python per-block loop on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .io import MapEntry, _map_frame, _vcf_frame
from .kernel import (
    MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK,
    block_deltas,
    block_deltas_parallel,
)


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
    parallel: bool = False,
) -> PairLOD:
    """Compute the CrosscheckFingerprints LOD and per-block Δ between two VCFs.

    Returns a PairLOD whose `.total` reproduces Picard's LOD_SCORE and whose
    `.delta` holds per-block contributions (Σ delta == total). Only blocks present
    in the map and in both VCFs, with evidence in both, are included. Set
    ``parallel=True`` to use the prange kernel (faster per call, but the block
    math is so cheap that this mainly helps very large maps).
    """
    empty = np.array([])

    # Intersect map ∩ vcf1 ∩ vcf2 on (chrom, pos) as a columnar join. Output order
    # follows the map (deterministic for fixed inputs); rank_blocks sorts anyway, so
    # we skip an explicit sort here.
    a = _vcf_frame(vcf1).rename({"ref": "ref1", "alt": "alt1", "pl": "pl1"})
    b = _vcf_frame(vcf2).rename({"ref": "ref2", "alt": "alt2", "pl": "pl2"})
    j = (
        _map_frame(haplotype_map)
        .join(a, on=["chrom", "pos"], how="inner")
        .join(b, on=["chrom", "pos"], how="inner")
    )

    # Orientation: swap genotype order if the VCF (REF, ALT) is reversed vs the map
    # (MAJOR, MINOR). Alleles matching neither orientation disagree with the map
    # (Picard would throw) and are dropped, mirroring `_orientation_swap is None`.
    fwd1 = (pl.col("ref1") == pl.col("major")) & (pl.col("alt1") == pl.col("minor"))
    rev1 = (pl.col("ref1") == pl.col("minor")) & (pl.col("alt1") == pl.col("major"))
    fwd2 = (pl.col("ref2") == pl.col("major")) & (pl.col("alt2") == pl.col("minor"))
    rev2 = (pl.col("ref2") == pl.col("minor")) & (pl.col("alt2") == pl.col("major"))
    j = j.with_columns(sw1=rev1, sw2=rev2, _ok1=fwd1 | rev1, _ok2=fwd2 | rev2).filter(
        pl.col("_ok1") & pl.col("_ok2")
    )
    if j.height == 0:
        return PairLOD(0.0, empty, empty, empty, empty, empty)

    pl1 = j["pl1"].to_numpy().astype(np.int64, copy=False)
    pl2 = j["pl2"].to_numpy().astype(np.int64, copy=False)
    # evidence gate: a flat PL carries no information (hasEvidence == false)
    flat1 = (pl1[:, 0] == pl1[:, 1]) & (pl1[:, 1] == pl1[:, 2])
    flat2 = (pl2[:, 0] == pl2[:, 1]) & (pl2[:, 1] == pl2[:, 2])
    keep = ~flat1 & ~flat2
    if not keep.any():
        return PairLOD(0.0, empty, empty, empty, empty, empty)

    pl1 = np.ascontiguousarray(pl1[keep])
    pl2 = np.ascontiguousarray(pl2[keep])
    sw1 = np.ascontiguousarray(j["sw1"].to_numpy()[keep])
    sw2 = np.ascontiguousarray(j["sw2"].to_numpy()[keep])
    maf = np.ascontiguousarray(j["maf"].to_numpy()[keep].astype(np.float64))

    kernel = block_deltas_parallel if parallel else block_deltas
    delta = kernel(pl1, pl2, sw1, sw2, maf, float(cap))

    return PairLOD(
        total=float(delta.sum()),
        chrom=j["chrom"].to_numpy()[keep],
        pos=j["pos"].to_numpy()[keep],
        name=j["name"].to_numpy()[keep],
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
