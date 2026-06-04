"""Static Picard-parity test: our LOD == committed Picard LOD on synthetic data.

Runs everywhere (no picard/java/external data needed) — it compares against the
checked-in `expected_crosscheck.tsv`, which was produced once by Picard 3.4.0
(regenerate with `uv run python tests/fixtures/_generate.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snp_prioritization import pair_lod

FIX = Path(__file__).resolve().parent / "fixtures"
MAP = str(FIX / "mini.map")
TOL = 1e-3


def _parse_expected(path: Path):
    """{(LEFT_SAMPLE, RIGHT_SAMPLE): LOD_SCORE} from a CrosscheckMetric TSV."""
    rows, header, in_m = {}, None, False
    for line in path.read_text().splitlines():
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
            rows[(r["LEFT_SAMPLE"], r["RIGHT_SAMPLE"])] = float(r["LOD_SCORE"])
    return rows


EXPECTED = _parse_expected(FIX / "expected_crosscheck.tsv")


@pytest.mark.parametrize("pair", sorted(EXPECTED), ids=lambda p: f"{p[0]}x{p[1]}")
def test_lod_matches_picard(pair):
    left, right = pair
    ours = pair_lod(str(FIX / f"{left}.vcf"), str(FIX / f"{right}.vcf"), MAP).total
    picard = EXPECTED[pair]
    assert abs(ours - picard) < TOL, f"{left}x{right}: ours={ours} picard={picard}"
