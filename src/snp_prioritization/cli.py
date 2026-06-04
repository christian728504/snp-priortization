"""Command-line entry point: rank the haplotype blocks driving a pair's LOD.

    snp-prioritization VCF1 VCF2 MAP [--direction swap|match|magnitude]
                       [--top N] [--bed OUT | --tsv OUT]

Prints the genome-wide LOD and the ranked blocks; with --bed/--tsv, also writes a
file ready for IGV (BED is 0-based half-open).
"""

from __future__ import annotations

import argparse
import sys

from .lod import pair_lod, rank_blocks


def _write_bed(path: str, ranked) -> None:
    with open(path, "w") as fh:
        for chrom, pos, name, maf, delta in ranked:
            label = f"{name}_MAF{maf:.3f}_D{delta:+.2f}"
            fh.write(f"{chrom}\t{pos - 1}\t{pos}\t{label}\n")


def _write_tsv(path: str, ranked) -> None:
    with open(path, "w") as fh:
        fh.write("chrom\tpos\tname\tmaf\tdelta\n")
        for chrom, pos, name, maf, delta in ranked:
            fh.write(f"{chrom}\t{pos}\t{name}\t{maf:.4f}\t{delta:+.6f}\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="snp-prioritization",
        description="Rank the haplotype blocks (SNPs) that drive a CrosscheckFingerprints LOD.",
    )
    p.add_argument("vcf1", help="first ExtractFingerprint VCF")
    p.add_argument("vcf2", help="second ExtractFingerprint VCF")
    p.add_argument("haplotype_map", help="Picard haplotype map")
    p.add_argument(
        "--direction",
        choices=("swap", "match", "magnitude"),
        default="swap",
        help="swap: most-negative Δ first; match: most-positive first; "
        "magnitude: largest |Δ| first (default: swap)",
    )
    p.add_argument("--top", type=int, default=20, help="show top N blocks (default: 20)")
    p.add_argument("--bed", metavar="OUT", help="write ranked blocks to a BED file")
    p.add_argument("--tsv", metavar="OUT", help="write ranked blocks to a TSV file")
    p.add_argument(
        "--parallel", action="store_true", help="use the prange (multithreaded) kernel"
    )
    args = p.parse_args(argv)

    result = pair_lod(
        args.vcf1, args.vcf2, args.haplotype_map, parallel=args.parallel
    )
    ranked = rank_blocks(result, args.direction, top=args.top)

    print(f"genome-wide LOD: {result.total:+.4f}", file=sys.stderr)
    print(f"blocks compared: {len(result)}", file=sys.stderr)
    print("chrom\tpos\tname\tmaf\tdelta")
    for chrom, pos, name, maf, delta in ranked:
        print(f"{chrom}\t{pos}\t{name}\t{maf:.4f}\t{delta:+.6f}")

    if args.bed:
        _write_bed(args.bed, ranked)
    if args.tsv:
        _write_tsv(args.tsv, ranked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
