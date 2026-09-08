"""Bounded local row-group scan for the pinned Civil Comments shard."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from pydantic import ValidationError
from typing_extensions import override

from bigdb_mvp.data.boundaries import CivilInput
from bigdb_mvp.data.manifest import ExclusionCount
from bigdb_mvp.data.models import (
    BinaryLabel,
    Language,
    RecordDraft,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.normalize import RowExclusionError, build_record, civil_label

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class CivilBatch:
    """Bounded accepted records and aggregate acquisition provenance."""

    records: tuple[SupervisedRecord, ...]
    clean_count: int
    harmful_count: int
    exclusions: tuple[ExclusionCount, ...]
    scanned: int


@dataclass(frozen=True, slots=True)
class CivilScanLimits:
    """Bounded acquisition limits for one local Parquet scan."""

    per_label_target: int
    max_scanned_rows: int


class CivilParquetError(Exception):
    """Report a pinned shard missing its required typed columns."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize the traceback-compatible typed failure."""
        self.reason_code = reason_code
        super().__init__(reason_code)

    @override
    def __str__(self) -> str:
        """Return the stable Parquet boundary code."""
        return self.reason_code


def _validate_schema(parquet_file: pq.ParquetFile) -> None:
    required = {"text", "toxicity"}
    if not required.issubset(parquet_file.schema_arrow.names):
        reason = "civil_parquet_schema_missing"
        raise CivilParquetError(reason)
    actual_types = {
        name: str(parquet_file.schema_arrow.field(name).type) for name in required
    }
    if actual_types != {"text": "string", "toxicity": "float"}:
        reason = "civil_parquet_schema_drift"
        raise CivilParquetError(reason)


def _quotas_full(counts: Counter[BinaryLabel], target: int) -> bool:
    return all(counts[label] >= target for label in BinaryLabel)


def scan_civil_parquet(path: Path, limits: CivilScanLimits) -> CivilBatch:
    """Scan only enough local row groups to fill both label cells."""
    records: list[SupervisedRecord] = []
    label_counts: Counter[BinaryLabel] = Counter()
    exclusions: Counter[str] = Counter()
    scanned = 0
    with pq.ParquetFile(path) as parquet_file:
        _validate_schema(parquet_file)
        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=["text", "toxicity"],
        ):
            for raw_row in batch.to_pylist():
                if scanned >= limits.max_scanned_rows:
                    break
                scanned += 1
                try:
                    row = CivilInput.model_validate(raw_row)
                    label = civil_label(row.toxicity)
                    if label_counts[label] >= limits.per_label_target:
                        continue
                    records.append(
                        build_record(
                            RecordDraft(
                                text=row.text,
                                label=label,
                                language=Language.EN,
                                source=SourceName.CIVIL_COMMENTS,
                                source_split="train",
                            )
                        )
                    )
                    label_counts[label] += 1
                except ValidationError:
                    exclusions["invalid_schema"] += 1
                except RowExclusionError as error:
                    exclusions[error.reason_code] += 1
                if _quotas_full(label_counts, limits.per_label_target):
                    break
            if scanned >= limits.max_scanned_rows or _quotas_full(
                label_counts, limits.per_label_target
            ):
                break
    return CivilBatch(
        records=tuple(records),
        clean_count=label_counts[BinaryLabel.CLEAN],
        harmful_count=label_counts[BinaryLabel.HARMFUL],
        exclusions=tuple(
            ExclusionCount(reason=reason, count=count)
            for reason, count in sorted(exclusions.items())
        ),
        scanned=scanned,
    )
