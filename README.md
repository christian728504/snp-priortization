# snp-prioritization

Pinpoint the SNPs (haplotype blocks) that drive a sample-swap call.

Picard `CrosscheckFingerprints` reports a single genome-wide LOD score per pair
of samples, but never writes out the per-block contributions that sum to it.
This package reimplements that LOD with exact parity to Picard 3.4.0 and exposes
the per-haplotype-block deltas (Δᵢ), so you can rank the blocks that most drive a
match or swap. The per-block math runs in a [numba](https://numba.pydata.org/)
`@njit` kernel.

## Install

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python
≥ 3.12.

```bash
# from the repo root
uv sync
```

This creates a virtual environment and installs the runtime dependencies
(`numpy`, `numba`, `oxbow`, `polars`). To include the dev tools (`pytest`,
`marimo`, `quak`) as well:

```bash
uv sync --all-groups
```

Run the test suite to confirm the install:

```bash
uv run pytest
```

The unit tests and the **static** Picard-parity test (against committed synthetic
fixtures) run with no external data or tools. A second, **live** parity test
re-runs `picard CrosscheckFingerprints` itself and is skipped automatically unless
`picard`/`java` and the fingerprint VCFs/maps are reachable.

## Usage (library)

The core entry point is `pair_lod`, which takes two ExtractFingerprint VCFs and a
Picard haplotype map and returns a `PairLOD` whose `.total` reproduces Picard's
`LOD_SCORE` and whose `.delta` holds the per-block contributions (`Σ delta == total`).

```python
from snp_prioritization import pair_lod, rank_blocks

result = pair_lod(
    "sampleA.vcf.gz",
    "sampleB.vcf.gz",
    "scratch/haplotype-maps/hg38_chr.map",
)

print(f"genome-wide LOD: {result.total:.4f}")   # matches Picard LOD_SCORE
print(f"blocks compared: {len(result)}")

# strongest different-individual ("swap") evidence: most-negative Δ first
for chrom, pos, name, maf, delta in rank_blocks(result, "swap", top=10):
    print(f"{chrom}:{pos}\t{name}\tmaf={maf:.3f}\tΔ={delta:+.4f}")
```

`direction` accepts `"swap"` (most-negative Δ first), `"match"` (most-positive
first), or `"magnitude"` (largest |Δ|); `top` limits the number of blocks returned
(omit it for all). Pass `parallel=True` to `pair_lod` to use the `prange`
multithreaded kernel.

The legacy module path still works:
`from snp_prioritization import fingerprint_lod as fplod; fplod.pair_lod(...)`.

## Usage (CLI)

```bash
uv run snp-prioritization sampleA.vcf.gz sampleB.vcf.gz hg38_chr.map \
    --direction swap --top 20 --bed top_swap.bed
```

Prints the genome-wide LOD (stderr) and a ranked block table (stdout); with
`--bed`/`--tsv` it also writes a file ready for IGV (BED is 0-based half-open).

### Notes

- Inputs are Picard `ExtractFingerprint` VCFs (one biallelic SNP record per
  haplotype block, with a `PL` field) and a Picard haplotype map
  (`CHROMOSOME POSITION NAME MAJOR_ALLELE MINOR_ALLELE MAF ANCHOR_SNP PANELS`).
- Only blocks present in the map and in both VCFs, with genotype evidence in
  both (a non-flat `PL`), are included — matching Picard's `hasEvidence()` gate.
- Layout: `io.py` (columnar VCF parsing via `oxbow`, map parsing), `kernel.py` (the
  numba LOD math), `lod.py` (`pair_lod` — a polars join feeding the kernel, plus
  `rank_blocks`), `cli.py`. See the module docstrings and the verbatim Picard source
  basis they mirror.
