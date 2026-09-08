"""Independent aggregate audit for the processed supervised sample."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import polars as pl
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.data.builder import DATASET_PATH, MANIFEST_PATH
from bigdb_mvp.data.manifest import Manifest
from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_COLUMNS: Final = (
    "text",
    "label",
    "language",
    "source",
    "source_split",
    "group_id",
    "text_sha256",
    "split",
)


class AuditReport(BaseModel):
    """Machine-readable dataset-integrity audit result."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    reason_codes: tuple[str, ...]
    record_count: int
    dataset_sha256: str
    logical_digest: str
    split_intersections: int
    sampled_non_train_count: int
    raw_text_fields: int
    thesis_path_references: int


def audit_dataset(
    dataset_path: Path = DATASET_PATH, manifest_path: Path = MANIFEST_PATH
) -> AuditReport:
    """Verify file digest, schema, row count, split roles, and group isolation."""
    manifest = Manifest.model_validate_json(manifest_path.read_bytes())
    actual_sha = sha256_file(dataset_path)
    lazy = pl.scan_parquet(dataset_path)
    frame = lazy.collect()
    reasons: list[str] = []
    if actual_sha != manifest.dataset_sha256:
        reasons.append("dataset_checksum_drift")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        reasons.append("schema_mismatch")
    if frame.height != manifest.record_count:
        reasons.append("record_count_mismatch")
    non_train = frame.filter(pl.col("source_split") != "train").height
    if non_train:
        reasons.append("source_role_leakage")
    duplicate_rows = frame.height - frame["text_sha256"].n_unique()
    if duplicate_rows:
        reasons.append("exact_duplicate_leakage")
    split_intersections = (
        frame.group_by("group_id")
        .agg(pl.col("split").n_unique().alias("split_count"))
        .filter(pl.col("split_count") > 1)
        .height
    )
    if split_intersections:
        reasons.append("group_split_leakage")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    thesis_references = int(
        "hate-speech-grad-thesis" in manifest_text
        or "/home/kyu/Coulmn" in manifest_text
    )
    if thesis_references:
        reasons.append("thesis_path_leakage")
    return AuditReport(
        valid=not reasons,
        reason_codes=tuple(reasons),
        record_count=frame.height,
        dataset_sha256=actual_sha,
        logical_digest=manifest.logical_digest,
        split_intersections=split_intersections,
        sampled_non_train_count=non_train,
        raw_text_fields=manifest.raw_text_fields,
        thesis_path_references=thesis_references,
    )
