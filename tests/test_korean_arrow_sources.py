from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from pyarrow import ipc

from bigdb_mvp.data.arrow_sources import (
    ArrowSourceError,
    ArrowSourceSpec,
    ingest_arrow_source,
)
from bigdb_mvp.data.models import BinaryLabel, SourceName
from bigdb_mvp.data.staging import sha256_file, write_staged

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def _write_arrow_stream(
    path: Path, columns: Mapping[str, Sequence[str | int | bool]]
) -> None:
    table = pa.table(columns)
    with path.open("wb") as sink, ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


def _spec(
    source: SourceName, path: Path, expected_sha256: str, expected_rows: int
) -> ArrowSourceSpec:
    return ArrowSourceSpec(
        source=source,
        upstream=path,
        revision="a" * 40,
        license="local-test-license",
        expected_sha256=expected_sha256,
        expected_rows=expected_rows,
    )


def test_unsmile_arrow_ingest_maps_labels_and_reuses_internal_staging(
    tmp_path: Path,
) -> None:
    # Given: clean, harmful, conflicting, and unlabeled UnSmile source rows.
    source_path = tmp_path / "unsmile.arrow"
    columns: dict[str, Sequence[str | int | bool]] = {
        "문장": ("neutral", "harmful", "conflict", "missing"),
        "여성/가족": (0, 1, 1, 0),
        "남성": (0, 0, 0, 0),
        "성소수자": (0, 0, 0, 0),
        "인종/국적": (0, 0, 0, 0),
        "연령": (0, 0, 0, 0),
        "지역": (0, 0, 0, 0),
        "종교": (0, 0, 0, 0),
        "기타 혐오": (0, 0, 0, 0),
        "악플/욕설": (0, 0, 0, 0),
        "clean": (1, 0, 1, 0),
    }
    _write_arrow_stream(source_path, columns)
    spec = _spec(SourceName.UNSMILE, source_path, sha256_file(source_path), 4)
    staging_root = tmp_path / "staging"

    # When: the pinned stream is ingested and the upstream file later disappears.
    records, receipt = ingest_arrow_source(spec, staging_root)
    source_path.unlink()
    staged_records, staged_receipt = ingest_arrow_source(spec, staging_root)

    # Then: only documented labels survive and the independent stage is reusable.
    assert Counter(row.label for row in records) == {
        BinaryLabel.CLEAN: 1,
        BinaryLabel.HARMFUL: 1,
    }
    assert {item.reason: item.count for item in receipt.exclusions} == {
        "unsmile_label_conflict": 1,
        "unsmile_label_missing": 1,
    }
    assert staged_records == records
    assert staged_receipt == receipt


def test_kold_arrow_ingest_maps_off_and_groups_article_comments(
    tmp_path: Path,
) -> None:
    # Given: two KOLD comments from one article with opposite OFF labels.
    source_path = tmp_path / "kold.arrow"
    _write_arrow_stream(
        source_path,
        {
            "guid": ("row-1", "row-2"),
            "source": ("news", "news"),
            "title": ("same article", "same article"),
            "comment": ("neutral", "harmful"),
            "OFF": (False, True),
        },
    )
    spec = _spec(SourceName.KOLD, source_path, sha256_file(source_path), 2)

    # When: the pinned KOLD stream is normalized.
    records, receipt = ingest_arrow_source(spec, tmp_path / "staging")

    # Then: OFF is binary-mapped and article-level leakage remains grouped.
    assert tuple(row.label for row in records) == (
        BinaryLabel.CLEAN,
        BinaryLabel.HARMFUL,
    )
    assert records[0].group_id == records[1].group_id
    assert receipt.accepted == 2
    assert receipt.excluded == 0


def test_stale_arrow_receipt_rebuilds_from_pinned_cache(tmp_path: Path) -> None:
    # Given: a digest-valid Arrow stage whose role metadata is stale.
    source_path = tmp_path / "unsmile.arrow"
    _write_arrow_stream(source_path, {"문장": ("neutral",), "clean": (1,)})
    spec = _spec(SourceName.UNSMILE, source_path, sha256_file(source_path), 1)
    staging_root = tmp_path / "staging"
    records, receipt = ingest_arrow_source(spec, staging_root)
    stale = receipt.model_copy(update={"source_split": "stale"})
    _ = write_staged(staging_root, stale, records)

    # When: stale metadata is found after records pass their digest check.
    rebuilt_records, rebuilt = ingest_arrow_source(spec, staging_root)

    # Then: the exact pinned Arrow cache is reloaded and restaged.
    assert rebuilt_records == records
    assert rebuilt.source_split == spec.source_split
    assert rebuilt.role is spec.role


def test_arrow_ingest_rejects_checksum_drift_before_parsing(tmp_path: Path) -> None:
    # Given: a readable Arrow stream that differs from the pinned digest.
    source_path = tmp_path / "drift.arrow"
    _write_arrow_stream(source_path, {"문장": ("neutral",), "clean": (1,)})
    spec = _spec(SourceName.UNSMILE, source_path, "0" * 64, 1)

    # When / Then: acquisition fails before accepting an unpinned source row.
    with pytest.raises(ArrowSourceError, match="arrow_source_checksum_drift"):
        _ = ingest_arrow_source(spec, tmp_path / "staging")


def test_arrow_ingest_rejects_row_count_drift(tmp_path: Path) -> None:
    # Given: an Arrow stream whose pinned row count is stale.
    source_path = tmp_path / "row-drift.arrow"
    _write_arrow_stream(source_path, {"문장": ("neutral",), "clean": (1,)})
    spec = _spec(SourceName.UNSMILE, source_path, sha256_file(source_path), 2)

    # When / Then: the source fails before normalization.
    with pytest.raises(ArrowSourceError, match="arrow_source_row_count_drift"):
        _ = ingest_arrow_source(spec, tmp_path / "staging")


def test_arrow_ingest_rejects_schema_drift(tmp_path: Path) -> None:
    # Given: a valid Arrow stream with a pinned but mismatched schema contract.
    source_path = tmp_path / "schema-drift.arrow"
    _write_arrow_stream(source_path, {"문장": ("neutral",), "clean": (1,)})
    spec = ArrowSourceSpec(
        source=SourceName.UNSMILE,
        upstream=source_path,
        revision="a" * 40,
        license="local-test-license",
        expected_sha256=sha256_file(source_path),
        expected_rows=1,
        expected_schema=("unexpected",),
    )

    # When / Then: schema drift is source-scoped and fail-closed.
    with pytest.raises(ArrowSourceError, match="arrow_source_schema_drift"):
        _ = ingest_arrow_source(spec, tmp_path / "staging")
