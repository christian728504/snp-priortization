"""Per-haplotype-block decomposition of Picard CrosscheckFingerprints' LOD score.

Pinpoint the SNPs (haplotype blocks) that drive a sample-swap call: ``pair_lod``
reproduces Picard's genome-wide LOD_SCORE and exposes the per-block contributions
Δᵢ (Σ Δᵢ == LOD); ``rank_blocks`` orders them so you can open the strongest in IGV.
"""

from __future__ import annotations

from .io import MapEntry, VcfRecord, read_fingerprint_vcf, read_haplotype_map
from .kernel import MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK
from .lod import PairLOD, pair_lod, rank_blocks

__all__ = [
    "MapEntry",
    "VcfRecord",
    "PairLOD",
    "pair_lod",
    "rank_blocks",
    "read_haplotype_map",
    "read_fingerprint_vcf",
    "MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK",
]
