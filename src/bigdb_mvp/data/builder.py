"""Deterministic balanced sample builder and safe manifest writer."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Final

import polars as pl

from bigdb_mvp.data.civil_source import ingest_civil_comments
from bigdb_mvp.data.local_sources import LOCAL_SOURCES, ingest_local_source
from bigdb_mvp.data.manifest import (
    DataCount,
    Manifest,
    SourceReceipt,
    SplitIntersection,
)
from bigdb_mvp.data.models import (
    BinaryLabel,
    Language,
    SourceName,
    SplitName,
    SupervisedRecord,
)
from bigdb_mvp.data.split import (
    assign_group_splits,
    deduplicate_exact,
    validate_minimum_quotas,
)
from bigdb_mvp.data.staging import sha256_file

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
DATASET_PATH: Final = PROJECT_ROOT / "data/processed/bilingual_supervised.parquet"
MANIFEST_PATH: Final = PROJECT_ROOT / "data/processed/manifest.json"
EVIDENCE_PATH: Final = PROJECT_ROOT / ".omo/evidence/mvp/task-2-data.json"
PER_LABEL_MAX: Final = 2_500


class DatasetWriteError(Exception):
    """Report an impossible missing split at the output boundary."""

    text_sha256: str

    def __init__(self, text_sha256: str) -> None:
        """Initialize a digest-only output failure."""
        self.text_sha256 = text_sha256
        super().__init__(f"output_split_missing:{text_sha256}")


def _sample_key(record: SupervisedRecord, seed: int) -> bytes:
    value = f"{seed}:{record.group_id}:{record.text_sha256}"
    return hashlib.sha256(value.encode()).digest()


def balanced_sample(
    records: tuple[SupervisedRecord, ...], *, seed: int
) -> tuple[SupervisedRecord, ...]:
    """Select at most 2,500 deterministic rows per language-label cell."""
    selected: list[SupervisedRecord] = []
    for language in Language:
        for label in BinaryLabel:
            cell = sorted(
                (
                    row
                    for row in records
                    if row.language is language and row.label is label
                ),
                key=lambda row: _sample_key(row, seed),
            )
            selected.extend(cell[:PER_LABEL_MAX])
    return tuple(selected)


def _split_text(record: SupervisedRecord) -> str:
    if record.split is None:
        raise DatasetWriteError(record.text_sha256)
    return record.split.value


def _logical_digest(records: tuple[SupervisedRecord, ...]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(row.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _data_counts(records: tuple[SupervisedRecord, ...]) -> tuple[DataCount, ...]:
    counts: Counter[tuple[Language, BinaryLabel, SplitName, SourceName]] = Counter()
    for row in records:
        if row.split is None:
            raise DatasetWriteError(row.text_sha256)
        counts[(row.language, row.label, row.split, row.source)] += 1
    return tuple(
        DataCount(
            language=language,
            label=label,
            split=split,
            source=source,
            count=count,
        )
        for (language, label, split, source), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1].value,
                item[0][2].value,
                item[0][3].value,
            ),
        )
    )


def _intersections(
    records: tuple[SupervisedRecord, ...],
) -> tuple[SplitIntersection, ...]:
    groups = {
        split: {row.group_id for row in records if row.split is split}
        for split in SplitName
    }
    pairs = (
        (SplitName.TRAIN, SplitName.CALIBRATION),
        (SplitName.TRAIN, SplitName.TEST),
        (SplitName.CALIBRATION, SplitName.TEST),
    )
    return tuple(
        SplitIntersection(
            left=left,
            right=right,
            count=len(groups[left] & groups[right]),
        )
        for left, right in pairs
    )


def _write_dataset(records: tuple[SupervisedRecord, ...]) -> str:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATASET_PATH.with_suffix(".parquet.tmp")
    frame = pl.DataFrame(
        {
            "text": [row.text for row in records],
            "label": [row.label.value for row in records],
            "language": [row.language.value for row in records],
            "source": [row.source.value for row in records],
            "source_split": [row.source_split for row in records],
            "group_id": [row.group_id for row in records],
            "text_sha256": [row.text_sha256 for row in records],
            "split": [_split_text(row) for row in records],
        }
    )
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        _ = temporary.replace(DATASET_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(DATASET_PATH)


def build_dataset(*, seed: int) -> Manifest:
    """Build the full sample from verified internal staging and publish evidence."""
    records: list[SupervisedRecord] = []
    receipts: list[SourceReceipt] = []
    for spec in LOCAL_SOURCES:
        source_records, receipt = ingest_local_source(spec)
        records.extend(source_records)
        receipts.append(receipt)
    civil_records, civil_receipt = ingest_civil_comments()
    records.extend(civil_records)
    receipts.append(civil_receipt)
    unique, removed = deduplicate_exact(tuple(records))
    sampled = balanced_sample(unique, seed=seed)
    assigned = assign_group_splits(sampled, seed=seed)
    validate_minimum_quotas(assigned)
    dataset_sha = _write_dataset(assigned)
    manifest = Manifest(
        schema_version=1,
        seed=seed,
        sources=tuple(receipts),
        dataset_file="data/processed/bilingual_supervised.parquet",
        dataset_sha256=dataset_sha,
        logical_digest=_logical_digest(assigned),
        record_count=len(assigned),
        counts=_data_counts(assigned),
        exact_duplicates_removed=removed,
        sampled_non_train_count=sum(row.source_split != "train" for row in assigned),
        split_intersections=_intersections(assigned),
    )
    payload = manifest.model_dump_json(indent=2)
    _ = MANIFEST_PATH.write_text(payload, encoding="utf-8")
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ = EVIDENCE_PATH.write_text(payload, encoding="utf-8")
    return manifest
