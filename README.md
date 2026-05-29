# snp-priortization

Pinpoint the SNPs (haplotype blocks) that drive a sample-swap call.

Picard `CrosscheckFingerprints` reports a single genome-wide LOD score per pair
of samples, but never writes out the per-block contributions that sum to it.
This package reimplements that LOD with exact parity to Picard 3.4.0 and exposes
the per-haplotype-block deltas (Δᵢ), so you can rank the blocks that most drive a
match or swap.

## Install

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python
≥ 3.12.

```bash
# from the repo root
uv sync
```

This creates a virtual environment and installs the runtime dependencies
(`loguru`, `polars`, `pysam`). To include the dev tools (`pytest`, `marimo`,
`quak`) as well:

```bash
uv sync --all-groups
```

Run the test suite to confirm the install:

```bash
uv run pytest
```

The unit tests run without any external data. The Picard-parity tests are
skipped automatically unless the fingerprint VCFs and haplotype maps are
reachable.

## Usage

The core entry point is `pair_lod`, which takes two ExtractFingerprint VCFs and a
Picard haplotype map and returns a `PairLOD` whose `.total` reproduces Picard's
`LOD_SCORE` and whose `.delta` holds the per-block contributions (`Σ delta == total`).

```python
from snp_priortization import fingerprint_lod as F

result = F.pair_lod(
    "sampleA.vcf.gz",
    "sampleB.vcf.gz",
    "scratch/haplotype-maps/hg38_chr.map",
)

print(f"genome-wide LOD: {result.total:.4f}")   # matches Picard LOD_SCORE
print(f"blocks compared: {len(result)}")
```

Rank the blocks that most drive the call with `rank_blocks`:

```python
# strongest different-individual ("swap") evidence: most-negative Δ first
for chrom, pos, name, maf, delta in F.rank_blocks(result, "swap", top=10):
    print(f"{chrom}:{pos}\t{name}\tmaf={maf:.3f}\tΔ={delta:+.4f}")

# strongest same-individual ("match") evidence: most-positive Δ first
top_match = F.rank_blocks(result, "match", top=10)

# largest absolute contribution either way
top_magnitude = F.rank_blocks(result, "magnitude", top=10)
```

`direction` accepts `"swap"`, `"match"`, or `"magnitude"`; `top` limits the
number of blocks returned (omit it for all of them).

### Notes

- Inputs are Picard `ExtractFingerprint` VCFs (one biallelic SNP record per
  haplotype block, with a `PL` field) and a Picard haplotype map
  (`CHROMOSOME POSITION NAME MAJOR_ALLELE MINOR_ALLELE MAF ANCHOR_SNP PANELS`).
- Only blocks present in the map and in both VCFs, with genotype evidence in
  both (a non-flat `PL`), are included — matching Picard's `hasEvidence()` gate.
- See `src/snp_priortization/fingerprint_lod.py` for the algorithm and the
  verbatim Picard source basis it mirrors.
