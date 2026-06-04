"""Backward-compatibility facade.

The library used to be a single module ``fingerprint_lod``; downstream code still
does ``from snp_prioritization import fingerprint_lod as fplod`` and calls
``fplod.pair_lod`` / ``fplod.read_fingerprint_vcf`` / ``fplod.read_haplotype_map``.
The implementation now lives in :mod:`snp_prioritization.io`,
:mod:`snp_prioritization.kernel`, and :mod:`snp_prioritization.lod`; this module
just re-exports the public surface so those imports keep working.

``read_fingerprint_vcf`` is cached natively now, so the old monkeypatch wrapping it
in ``functools.lru_cache`` is no longer needed.
"""

from __future__ import annotations

from .io import MapEntry, VcfRecord, read_fingerprint_vcf, read_haplotype_map
from .kernel import MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK
from .lod import PairLOD, _orientation_swap, pair_lod, rank_blocks

__all__ = [
    "MapEntry",
    "VcfRecord",
    "PairLOD",
    "pair_lod",
    "rank_blocks",
    "read_haplotype_map",
    "read_fingerprint_vcf",
    "_orientation_swap",
    "MAX_EFFECT_OF_EACH_HAPLOTYPE_BLOCK",
]
