"""Regenerate the synthetic Picard-parity fixture (run once; outputs are committed).

Produces, under tests/fixtures/:
  mini.map               - a Picard haplotype map, all single-SNP blocks
  {A,B,C}.vcf            - synthetic single-sample ExtractFingerprint-style VCFs
  expected_crosscheck.tsv - Picard 3.4.0 CrosscheckFingerprints output (ground truth)

The data is entirely synthetic (hand-chosen PLs) so no real donor genotypes are
committed. Blocks are engineered to span matches, mismatches, hets, the no-evidence
gate, REF/ALT-reversed orientation, and the ±MAX_EFFECT cap.

Usage:  uv run python tests/fixtures/_generate.py
Requires `picard` + `java` on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTIG_LEN = 1_000_000

# Each row: pos, major, minor, maf, ref, alt, plA, plB, plC
# ref/alt are the VCF alleles (ExtractFingerprint orients PL to REF/ALT); when
# ref==minor the library/Picard must swap genotype order 0<->2.
BLOCKS = [
    # matches, mismatches, hets at common MAF, REF==MAJOR (no swap)
    (1000, "A", "G", 0.30, "A", "G", (0, 50, 500), (0, 40, 450), (500, 50, 0)),
    (2000, "C", "T", 0.45, "C", "T", (60, 0, 60), (50, 0, 55), (0, 40, 400)),
    # REF==MINOR -> orientation swap
    (3000, "G", "A", 0.10, "A", "G", (0, 30, 300), (0, 30, 300), (300, 30, 0)),
    (3500, "T", "G", 0.22, "G", "T", (400, 35, 0), (380, 33, 0), (0, 35, 360)),
    # no-evidence gate: flat PL in A -> block dropped for any pair with A
    (4000, "T", "C", 0.20, "T", "C", (0, 0, 0), (0, 40, 400), (0, 40, 400)),
    # cap: arbitrarily confident opposite homozygotes
    (5000, "A", "T", 0.40, "A", "T", (0, 1000, 5000), (5000, 1000, 0), (0, 900, 4800)),
    # rare-allele concordant homozygotes -> strong same-individual evidence
    (6000, "G", "C", 0.05, "G", "C", (400, 40, 0), (350, 35, 0), (0, 40, 420)),
    (7000, "A", "C", 0.15, "A", "C", (0, 45, 480), (0, 48, 500), (0, 47, 470)),
    (8000, "T", "A", 0.48, "T", "A", (70, 0, 75), (0, 38, 380), (360, 36, 0)),
    (9000, "C", "G", 0.33, "C", "G", (0, 55, 520), (510, 52, 0), (0, 50, 500)),
    (10000, "G", "T", 0.27, "T", "G", (320, 30, 0), (0, 33, 340), (0, 31, 330)),
    (11000, "A", "G", 0.08, "A", "G", (0, 60, 600), (0, 58, 580), (590, 59, 0)),
    (12000, "C", "A", 0.42, "C", "A", (0, 0, 0), (45, 0, 48), (0, 44, 440)),
]

SAMPLES = ["A", "B", "C"]
PL_IDX = {"A": 6, "B": 7, "C": 8}


def write_map() -> Path:
    path = HERE / "mini.map"
    lines = [
        "@HD\tVN:1.5\tSO:coordinate",
        f"@SQ\tSN:chr1\tLN:{CONTIG_LEN}",
        "#CHROMOSOME\tPOSITION\tNAME\tMAJOR_ALLELE\tMINOR_ALLELE\tMAF\tANCHOR_SNP\tPANELS",
    ]
    for i, b in enumerate(BLOCKS):
        pos, major, minor, maf = b[0], b[1], b[2], b[3]
        # all single-SNP blocks: ANCHOR_SNP empty -> each SNP is its own anchor
        lines.append(f"chr1\t{pos}\tchr1:{i}\t{major}\t{minor}\t{maf:.6f}\t\t")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_vcf(sample: str) -> Path:
    path = HERE / f"{sample}.vcf"
    hdr = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth">',
        '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled likelihoods">',
        f"##contig=<ID=chr1,length={CONTIG_LEN}>",
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}",
    ]
    rows = []
    for b in BLOCKS:
        pos, ref, alt = b[0], b[4], b[5]
        pl = b[PL_IDX[sample]]
        plstr = ",".join(str(x) for x in pl)
        rows.append(
            f"chr1\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tAD:DP:PL\t10,10:20:{plstr}"
        )
    path.write_text("\n".join(hdr + rows) + "\n")
    return path


def run_picard(map_path: Path, vcfs: list[Path]) -> Path:
    out = HERE / "expected_crosscheck.tsv"
    cmd = [
        "picard",
        "CrosscheckFingerprints",
        *sum([["--INPUT", str(v)] for v in vcfs], []),
        "--HAPLOTYPE_MAP", str(map_path),
        "--CROSSCHECK_BY", "FILE",
        "--CALCULATE_TUMOR_AWARE_RESULTS", "false",
        "--LOD_THRESHOLD", "-1000000",
        "--OUTPUT", str(out),
    ]
    # exits non-zero on unexpected mismatch but still writes the matrix
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists():
        raise SystemExit(f"picard wrote no output:\n{r.stderr}")
    return out


def main() -> None:
    m = write_map()
    vcfs = [write_vcf(s) for s in SAMPLES]
    out = run_picard(m, vcfs)
    print(f"wrote {m.name}, {[v.name for v in vcfs]}, {out.name}")


if __name__ == "__main__":
    main()
