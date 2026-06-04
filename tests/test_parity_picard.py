"""Live Picard-parity test against the linked real fingerprint data.

Opt-in: skipped unless `picard` + `java` are on PATH and the linked `fingerprints/`
data + `scratch/haplotype-maps/` are present. When it runs, it invokes the real
`picard CrosscheckFingerprints` and asserts our LOD matches Picard's LOD_SCORE to
within 1e-3 across several real cross-modality / relative pairs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from snp_prioritization import pair_lod

PROJECT = Path(__file__).resolve().parent.parent
FP = PROJECT / "fingerprints"
HM = PROJECT / "scratch" / "haplotype-maps"

FULL = HM / "hg38_chr.map"
FILTERED = HM / "hg38_chr.filtered.map"
CODING = HM / "hg38_chr.coding_exons.map"

# (label, map, [vcf paths sharing that map]) — picard scores every pair within a group.
CONTEXTS = [
    ("full", FULL, [FP / "wgs/MOHD_EG100578.vcf.gz",      # father WGS
                    FP / "wgs/MOHD_EG100500.vcf.gz",      # son WGS (relatives)
                    FP / "atac/MOHD_EA100479.vcf.gz"]),   # father ATAC (cross-modality)
    ("filtered", FILTERED, [FP / "wgs/MOHD_EG100578.vcf.gz",
                            FP / "wgbs/MOHD_EB100578.vcf.gz"]),  # father WGS vs WGBS
    ("coding", CODING, [FP / "wgs/MOHD_EG100578.vcf.gz",
                        FP / "rna/MOHD_ER100509.vcf.gz"]),       # WGS vs son RNA
]

TOL = 1e-3
_have_tools = bool(shutil.which("picard") and shutil.which("java"))
_have_data = FP.exists() and FULL.exists()

pytestmark = pytest.mark.skipif(
    not (_have_tools and _have_data), reason="picard/java or linked fingerprint data not available"
)


def _picard_lods(map_path: Path, vcfs: list[Path], tmp: Path) -> dict[tuple[str, str], float]:
    out = tmp / "cc.tsv"
    cmd = [
        "picard", "CrosscheckFingerprints",
        *sum([["--INPUT", str(v)] for v in vcfs], []),
        "--HAPLOTYPE_MAP", str(map_path),
        "--CROSSCHECK_BY", "FILE",
        "--CALCULATE_TUMOR_AWARE_RESULTS", "false",
        "--LOD_THRESHOLD", "-1000000",
        "--OUTPUT", str(out),
    ]
    # exits non-zero on mismatches but still writes the matrix
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert out.exists(), "Picard wrote no output"
    rows, header, in_m = {}, None, False
    for line in out.read_text().splitlines():
        if line.startswith("## METRICS CLASS"):
            in_m = True
            continue
        if in_m and header is None:
            header = line.split("\t")
            continue
        if in_m and header is not None:
            if line == "" or line.startswith("#"):
                break
            r = dict(zip(header, line.split("\t")))
            # group-by-FILE keys are "<uri>::<sample>"; key on the sample name
            ls, rs = r["LEFT_SAMPLE"], r["RIGHT_SAMPLE"]
            rows[(ls, rs)] = float(r["LOD_SCORE"])
    return rows


@pytest.mark.parametrize("ctx", CONTEXTS, ids=lambda c: c[0])
def test_parity_with_picard(ctx, tmp_path):
    label, map_path, vcfs = ctx
    vcfs = [v for v in vcfs if v.exists()]
    if len(vcfs) < 2:
        pytest.skip(f"{label}: fewer than 2 VCFs present")
    picard = _picard_lods(map_path, vcfs, tmp_path)
    sample = {v: v.name.split(".")[0] for v in vcfs}  # MOHD_EG100578
    worst = 0.0
    for i, vi in enumerate(vcfs):
        for vj in vcfs[i + 1 :]:
            key = (sample[vi], sample[vj])
            exp = picard.get(key) or picard.get((sample[vj], sample[vi]))
            assert exp is not None, f"no Picard LOD for {key}"
            ours = pair_lod(str(vi), str(vj), str(map_path)).total
            worst = max(worst, abs(ours - exp))
    assert worst < TOL, f"{label}: worst |diff| {worst:.2e}"
