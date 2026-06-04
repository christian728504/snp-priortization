"""Parsing tests for snp_prioritization.io (no external data)."""

from __future__ import annotations

from snp_prioritization import io

FIX = __import__("pathlib").Path(__file__).resolve().parent / "fixtures"


def test_map_skips_headers_and_parses_columns():
    hap = io.read_haplotype_map(str(FIX / "mini.map"))
    # 13 single-SNP blocks in the fixture; @/# lines skipped.
    assert len(hap) == 13
    e = hap[("chr1", 1000)]
    assert (e.major, e.minor) == ("A", "G")
    assert abs(e.maf - 0.30) < 1e-9


def test_anchor_maf_is_shared_across_a_block(tmp_path):
    # Block where the ANCHOR is the higher-coordinate SNP and the representative
    # (lowest-coord) member has a DIFFERENT own-MAF: the prior must use the anchor's.
    m = tmp_path / "anchor.map"
    m.write_text(
        "@HD\tVN:1.5\n"
        "@SQ\tSN:chr1\tLN:1000000\n"
        "#CHROMOSOME\tPOSITION\tNAME\tMAJOR_ALLELE\tMINOR_ALLELE\tMAF\tANCHOR_SNP\tPANELS\n"
        # member (lower coord) points at the anchor; its own MAF 0.1136 is ignored
        "chr1\t100\tchr1:b\tA\tG\t0.113600\tchr1:a\t\n"
        # anchor (higher coord), empty ANCHOR_SNP, block MAF 0.129400
        "chr1\t200\tchr1:a\tT\tC\t0.129400\t\t\n"
    )
    hap = io.read_haplotype_map(str(m))
    member, anchor = hap[("chr1", 100)], hap[("chr1", 200)]
    # both SNPs carry the anchor's MAF...
    assert abs(member.maf - 0.1294) < 1e-9
    assert abs(anchor.maf - 0.1294) < 1e-9
    # ...but each keeps its own alleles (used for the orientation swap)
    assert (member.major, member.minor) == ("A", "G")
    assert (anchor.major, anchor.minor) == ("T", "C")


def test_vcf_parses_pl_and_keys_by_chrom_pos():
    rec = io.read_fingerprint_vcf(str(FIX / "A.vcf"))
    assert rec[("chr1", 1000)].pl == (0, 50, 500)
    assert rec[("chr1", 1000)].ref == "A" and rec[("chr1", 1000)].alt == "G"
    # one record per block position
    assert len(rec) == 13


def test_columnar_readers_are_cached():
    # lru_cache lives on the columnar readers; same path returns the identical frame.
    assert io.read_haplotype_map(str(FIX / "mini.map")) is io.read_haplotype_map(
        str(FIX / "mini.map")
    )
    assert io._map_frame(str(FIX / "mini.map")) is io._map_frame(str(FIX / "mini.map"))
    assert io._vcf_frame(str(FIX / "A.vcf")) is io._vcf_frame(str(FIX / "A.vcf"))
    # read_fingerprint_vcf builds a fresh dict from the cached frame: equal, not identical
    assert io.read_fingerprint_vcf(str(FIX / "A.vcf")) == io.read_fingerprint_vcf(
        str(FIX / "A.vcf")
    )


def test_vcf_frame_schema_and_biallelic_filter():
    df = io._vcf_frame(str(FIX / "A.vcf"))
    assert df.columns == ["chrom", "pos", "ref", "alt", "pl"]
    # PL is a fixed-size length-3 array, ready for a zero-copy (N, 3) extraction
    assert df["pl"].to_numpy().shape == (13, 3)
    assert df["pl"].to_numpy()[0].tolist() == [0, 50, 500]
    # map frame carries the block MAF + alleles for the join
    mf = io._map_frame(str(FIX / "mini.map"))
    assert set(mf.columns) == {"chrom", "pos", "name", "major", "minor", "maf"}
