"""Parsing of the two inputs: Picard haplotype maps and ExtractFingerprint VCFs.

VCFs are read columnarly with `oxbow` (Rust/Arrow) into a polars frame — the hot
path that feeds `pair_lod`. The map is a Picard-specific text format oxbow does not
know, so it keeps a small text parser. All readers are `lru_cache`d on their path so
a file is parsed once even across the thousands of `pair_lod` calls in a large
cross-comparison; cache keys are paths, so an edited file in the same process keeps
the stale parse — fine for this read-only reference data.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import oxbow as ox
import polars as pl


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

    SNPs are grouped into haplotype (LD) blocks exactly as Picard's
    HaplotypeMap.fromHaplotypeDatabase does: a row is an *anchor* (block start) if
    ANCHOR_SNP is empty or equals its own NAME; the block's MAF — used for the HWE
    prior π — is the anchor's MAF, shared by every SNP in the block. Each SNP keeps
    its OWN MAJOR/MINOR (used for the allele-orientation swap), but its own per-row
    MAF is ignored for the prior if it is a non-anchor.
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


@functools.lru_cache(maxsize=8)
def _map_frame(path: str) -> pl.DataFrame:
    """Columnar view of the haplotype map for the join in `pair_lod`.

    Columns: chrom (str), pos (Int64), name (str), major (str), minor (str),
    maf (Float64). Built from the cached dict parser so the block/anchor MAF logic
    lives in exactly one place.
    """
    hap = read_haplotype_map(path)
    rows = [
        (chrom, pos, e.name, e.major, e.minor, e.maf)
        for (chrom, pos), e in hap.items()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "chrom": pl.String,
            "pos": pl.Int64,
            "name": pl.String,
            "major": pl.String,
            "minor": pl.String,
            "maf": pl.Float64,
        },
        orient="row",
    )


# --------------------------------------------------------------------------- #
# Fingerprint VCF
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VcfRecord:
    ref: str
    alt: str
    pl: tuple[int, int, int]


@functools.lru_cache(maxsize=None)
def _vcf_frame(path: str) -> pl.DataFrame:
    """Columnar ExtractFingerprint VCF reader (oxbow -> polars), the hot path.

    One row per biallelic SNP record with a 3-length PL. Columns: chrom (str),
    pos (Int64), ref (str), alt (str), pl (Array(Int32, 3)). The fixed-size Array
    dtype lets `pair_lod` pull PL straight to an (N, 3) numpy array with
    `.to_numpy()` — never via Python lists.
    """
    vf = ox.from_vcf(path, samples="*", genotype_fields=["PL"])
    sample = vf.samples[0]  # ExtractFingerprint VCFs are single-sample
    return (
        vf.to_polars()
        .select(
            pl.col("chrom").cast(pl.String).alias("chrom"),
            pl.col("pos").cast(pl.Int64).alias("pos"),
            pl.col("ref").alias("ref"),
            pl.col("alt").alias("alt"),
            pl.col(sample).struct.field("PL").alias("pl"),
        )
        .filter(
            (pl.col("alt").list.len() == 1)
            & (pl.col("ref").str.len_chars() == 1)
            & (pl.col("alt").list.first().str.len_chars() == 1)
            & (pl.col("pl").list.len() == 3)
        )
        .with_columns(
            pl.col("alt").list.first().alias("alt"),
            pl.col("pl").list.to_array(3).alias("pl"),
        )
    )


def read_fingerprint_vcf(path: str) -> dict[tuple[str, int], VcfRecord]:
    """Parse an ExtractFingerprint VCF into {(chrom, pos): VcfRecord}.

    Back-compat array-of-structs view built from the cached columnar frame. The
    hot path (`pair_lod`) uses `_vcf_frame` directly; this is for callers/tests that
    want the per-record dict.
    """
    df = _vcf_frame(path)
    out: dict[tuple[str, int], VcfRecord] = {}
    for chrom, pos, ref, alt, plv in zip(
        df["chrom"], df["pos"], df["ref"], df["alt"], df["pl"].to_numpy()
    ):
        out[(chrom, int(pos))] = VcfRecord(
            ref=ref, alt=alt, pl=(int(plv[0]), int(plv[1]), int(plv[2]))
        )
    return out
