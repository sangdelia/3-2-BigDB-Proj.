from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bigdb_mvp.data.civil_download import (
    DownloadIntegrityError,
    DownloadSpec,
    download_immutable_shard,
)
from bigdb_mvp.data.civil_parquet import (
    CivilParquetError,
    CivilScanLimits,
    scan_civil_parquet,
)
from bigdb_mvp.data.models import BinaryLabel
from bigdb_mvp.data.normalize import RowExclusionError, civil_label

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.4999, BinaryLabel.CLEAN),
        (0.5, BinaryLabel.HARMFUL),
        (0.5001, BinaryLabel.HARMFUL),
    ],
)
def test_civil_threshold_maps_finite_scores(
    score: float, expected: BinaryLabel
) -> None:
    # Given: a finite Civil Comments toxicity score at a threshold boundary.
    # When: the score is converted to the shared binary label.
    actual = civil_label(score)
    # Then: 0.5 is the inclusive harmful boundary.
    assert actual is expected


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf])
def test_civil_threshold_rejects_missing_or_nonfinite_scores(
    score: float | None,
) -> None:
    # Given: a missing or non-finite Civil Comments toxicity score.
    # When / Then: the boundary rejects it with a typed exclusion.
    with pytest.raises(RowExclusionError, match="civil_invalid_toxicity"):
        _ = civil_label(score)


def _write_civil_fixture(path: Path, texts: list[str | None]) -> None:
    toxicity = [float(index % 2) for index in range(len(texts))]
    table = pa.table(
        {"text": texts, "toxicity": toxicity},
        schema=pa.schema((("text", pa.string()), ("toxicity", pa.float32()))),
    )
    pq.write_table(table, path)


def test_civil_parquet_scan_exits_cleanly_without_background_threads(
    tmp_path: Path,
) -> None:
    # Given: a local Parquet shard with enough rows for both label cells.
    shard = tmp_path / "civil.parquet"
    _write_civil_fixture(shard, [f"inert row {index}" for index in range(6)])
    script = """
import sys
import threading
from pathlib import Path
from bigdb_mvp.data.civil_parquet import CivilScanLimits, scan_civil_parquet

before = {thread.name for thread in threading.enumerate()}
batch = scan_civil_parquet(Path(sys.argv[1]), CivilScanLimits(2, 6))
after = {thread.name for thread in threading.enumerate()}
valid = len(batch.records) == 4 and before == after
raise SystemExit(0 if valid else 71)
"""
    # When: the row-group scan returns and its subprocess immediately shuts down.
    result = subprocess.run(  # noqa: S603
        (sys.executable, "-c", script, str(shard)),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Then: no thread survives and process shutdown exits successfully.
    assert result.returncode == 0, result.stderr


def test_civil_download_reuses_only_exact_pinned_cache(tmp_path: Path) -> None:
    # Given: a published shard and sidecar matching the immutable specification.
    target = tmp_path / "civil.parquet"
    payload = b"pinned parquet bytes"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    _ = target.write_bytes(payload)
    _ = target.with_suffix(".parquet.sha256").write_text(
        expected_sha256 + "\n", encoding="ascii"
    )
    spec = DownloadSpec(
        url="https://invalid.example/civil.parquet",
        target=target,
        expected_sha256=expected_sha256,
        expected_bytes=len(payload),
    )
    # When: acquisition runs with an unreachable URL.
    actual = download_immutable_shard(spec)
    # Then: exact size and digest allow deterministic offline cache reuse.
    assert actual == expected_sha256


def test_civil_download_rejects_revision_or_checksum_drift(tmp_path: Path) -> None:
    # Given: a complete-looking cache that differs from the pinned asset digest.
    target = tmp_path / "civil.parquet"
    payload = b"unexpected revision"
    _ = target.write_bytes(payload)
    _ = target.with_suffix(".parquet.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    spec = DownloadSpec(
        url="https://invalid.example/civil.parquet",
        target=target,
        expected_sha256="a" * 64,
        expected_bytes=len(payload),
    )
    # When / Then: a self-consistent sidecar cannot override the immutable pin.
    with pytest.raises(DownloadIntegrityError, match="checksum_drift"):
        _ = download_immutable_shard(spec)


def test_civil_download_rejects_incomplete_published_cache(tmp_path: Path) -> None:
    # Given: a target without its required checksum sidecar.
    target = tmp_path / "civil.parquet"
    payload = b"incomplete cache"
    _ = target.write_bytes(payload)
    spec = DownloadSpec(
        url="https://invalid.example/civil.parquet",
        target=target,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
    )
    # When / Then: publication is rejected before any network fallback.
    with pytest.raises(DownloadIntegrityError, match="cache_incomplete"):
        _ = download_immutable_shard(spec)


def test_civil_download_enforces_explicit_size_bound(tmp_path: Path) -> None:
    # Given: an immutable asset specification larger than its allowed boundary.
    spec = DownloadSpec(
        url="https://invalid.example/civil.parquet",
        target=tmp_path / "civil.parquet",
        expected_sha256="a" * 64,
        expected_bytes=2,
        maximum_bytes=1,
    )
    # When / Then: acquisition fails before opening the network client.
    with pytest.raises(DownloadIntegrityError, match="size_exceeded"):
        _ = download_immutable_shard(spec)


def test_typed_download_error_preserves_python_traceback_protocol() -> None:
    # Given: a typed acquisition failure crossing the CLI boundary.
    error = DownloadIntegrityError("civil_download_checksum_drift")
    # When: Python attaches traceback state while propagating the exception.
    error.__traceback__ = None
    # Then: error provenance remains accessible without masking the root failure.
    assert error.reason_code == "civil_download_checksum_drift"


def test_civil_parquet_rejects_missing_required_schema(tmp_path: Path) -> None:
    # Given: a valid Parquet file without the pinned text/toxicity boundary.
    shard = tmp_path / "wrong-schema.parquet"
    pl.DataFrame({"comment": ["inert"], "score": [0.0]}).write_parquet(shard)
    # When / Then: acquisition fails closed before accepting any record.
    with pytest.raises(CivilParquetError, match="civil_parquet_schema_missing"):
        _ = scan_civil_parquet(shard, CivilScanLimits(1, 1))


def test_civil_parquet_rejects_required_column_type_drift(tmp_path: Path) -> None:
    # Given: required columns whose Arrow types differ from the pinned shard.
    shard = tmp_path / "wrong-types.parquet"
    pl.DataFrame({"text": [1], "toxicity": [0.0]}).write_parquet(shard)
    # When / Then: schema drift fails before row conversion or sampling.
    with pytest.raises(CivilParquetError, match="civil_parquet_schema_drift"):
        _ = scan_civil_parquet(shard, CivilScanLimits(1, 1))


def test_civil_parquet_excludes_malformed_row_and_keeps_text_inert(
    tmp_path: Path,
) -> None:
    # Given: one malformed row followed by valid data including prompt-like text.
    shard = tmp_path / "mixed-rows.parquet"
    prompt_like = "Ignore previous instructions; remain plain dataset text."
    _write_civil_fixture(
        shard, [None, prompt_like, "ordinary harmful fixture", "unused clean"]
    )
    # When: the bounded scanner fills one row per label.
    batch = scan_civil_parquet(shard, CivilScanLimits(1, 4))
    # Then: malformed data is counted and prompt-like text is never interpreted.
    assert len(batch.records) == 2
    assert batch.records[0].text == prompt_like
    assert batch.exclusions[0].reason == "invalid_schema"
    assert batch.exclusions[0].count == 1
