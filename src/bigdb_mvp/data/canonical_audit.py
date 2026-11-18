"""Fail-closed aggregate audit for the published full-canonical generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bigdb_mvp.data.canonical_pointer_audit import (
    PublicationBindingError,
    PublicationSelection,
    load_publication,
)
from bigdb_mvp.data.canonical_report import counts, group_intersections, parquet_facts
from bigdb_mvp.data.dedup_audit import audit_canonical_duplicates
from bigdb_mvp.data.manifest import (
    CanonicalCount,
    CanonicalIntersection,
    CanonicalManifest,
    CanonicalRoleCount,
    ExpectedSource,
    ExpectedSourceContract,
    SourceReceipt,
)
from bigdb_mvp.data.models import CanonicalRecord, CanonicalRole, SourceName
from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_COLUMNS: Final = (
    "text",
    "label",
    "language",
    "source",
    "source_split",
    "role",
    "group_id",
    "text_sha256",
)


class FullCanonicalAuditReport(BaseModel):
    """Digest-only result of the full-canonical quality gates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    reason_codes: tuple[str, ...]
    record_count: int = Field(ge=0)
    dataset_sha256: str = ""
    manifest_sha256: str = ""
    logical_digest: str = ""
    role_counts: tuple[CanonicalRoleCount, ...] = ()
    source_counts: tuple[CanonicalCount, ...] = ()
    language_counts: tuple[CanonicalCount, ...] = ()
    label_counts: tuple[CanonicalCount, ...] = ()
    split_counts: tuple[CanonicalCount, ...] = ()
    row_group_count: int = Field(default=0, ge=0)
    row_group_rows: tuple[int, ...] = ()
    compression: tuple[str, ...] = ()
    source_receipts: tuple[SourceReceipt, ...] = ()
    group_role_intersections: tuple[CanonicalIntersection, ...] = ()
    group_split_intersections: tuple[CanonicalIntersection, ...] = ()
    schema_fingerprint: str = ""
    file_size_bytes: int = Field(default=0, ge=0)
    raw_bytes: int | None = Field(default=None, ge=0)
    canonical_size_bytes: int = Field(default=0, ge=0)
    missing_sources: tuple[str, ...] = ()


def _failure(code: str) -> FullCanonicalAuditReport:
    return FullCanonicalAuditReport(valid=False, reason_codes=(code,), record_count=0)


def _source_key(
    item: ExpectedSource | SourceReceipt,
) -> tuple[SourceName, str, CanonicalRole]:
    return item.source, item.source_split, item.role


def _check_source_contract(
    manifest: CanonicalManifest,
    records: tuple[CanonicalRecord, ...],
    contract: ExpectedSourceContract,
) -> tuple[str, ...]:
    expected = {_source_key(item): item for item in contract.sources}
    receipts = {_source_key(item): item for item in manifest.source_receipts}
    reasons: list[str] = []
    if set(expected) != set(receipts):
        return ("source_provenance_source_mismatch",)
    for source_key, trusted in expected.items():
        receipt = receipts[source_key]
        if receipt.input_sha256 != trusted.input_sha256:
            reasons.append("source_provenance_digest_mismatch")
        if (
            receipt.asset_sha256 != trusted.asset_sha256
            or receipt.staging_sha256 != trusted.staging_sha256
            or receipt.schema_sha256 != trusted.schema_sha256
            or receipt.raw_bytes != trusted.raw_bytes
        ):
            reasons.append("source_provenance_digest_mismatch")
        if (
            receipt.source_split != trusted.source_split
            or receipt.role is not trusted.role
        ):
            reasons.append("source_provenance_source_mismatch")
        if (
            receipt.scanned != trusted.scanned
            or receipt.accepted != trusted.accepted
            or receipt.excluded != trusted.excluded
        ):
            reasons.append("source_receipt_mismatch")
        actual_count = sum(
            1
            for row in records
            if (row.source, row.source_split, row.role) == source_key
        )
        if actual_count > trusted.accepted:
            reasons.append("source_provenance_source_mismatch")
    return tuple(dict.fromkeys(reasons))


def _manifest(
    root: Path,
) -> PublicationSelection | FullCanonicalAuditReport:
    try:
        accepted = load_publication(root)
    except PublicationBindingError as error:
        return _failure(error.reason_code)
    return accepted


def audit_full_canonical(  # noqa: C901, PLR0912
    output_root: Path,
    *,
    expected_sources: ExpectedSourceContract | None = None,
) -> FullCanonicalAuditReport:
    """Audit the pointer, generation, Parquet facts, receipts, and leakage gates."""
    accepted = _manifest(output_root)
    if isinstance(accepted, FullCanonicalAuditReport):
        return accepted
    pointer = accepted.pointer
    manifest = accepted.manifest
    dataset_path = accepted.dataset_path
    manifest_path = accepted.manifest_path
    try:
        facts = parquet_facts(dataset_path)
        frame = pl.read_parquet(dataset_path)
    except (OSError, ValueError, RuntimeError):
        return _failure("parquet_invalid")
    reasons: list[str] = []
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        reasons.append("schema_mismatch")
    if facts.schema_fingerprint != manifest.schema_fingerprint:
        reasons.append("schema_fingerprint_mismatch")
    if manifest.artifact is not None and frame.height != manifest.artifact.row_count:
        reasons.append("row_count_mismatch")
    try:
        records = tuple(CanonicalRecord.model_validate(row) for row in frame.to_dicts())
    except ValidationError:
        return FullCanonicalAuditReport(
            valid=False,
            reason_codes=("role_label_invariant_mismatch",),
            record_count=frame.height,
            dataset_sha256=sha256_file(dataset_path),
            manifest_sha256=sha256_file(manifest_path),
            logical_digest=pointer.logical_digest,
        )
    actual = counts(records)
    actual_roles = tuple(
        CanonicalRoleCount(
            role=role, count=sum(1 for row in records if row.role.value == role.value)
        )
        for role in CanonicalRole
        if any(row.role.value == role.value for row in records)
    )
    if frame.height != manifest.record_count:
        reasons.append("row_count_mismatch")
    if actual_roles != manifest.role_counts:
        reasons.append("role_count_mismatch")
    if any(
        actual[name] != getattr(manifest, f"{name}_counts")
        for name in ("source", "language", "label", "split")
    ):
        reasons.append("dimension_count_mismatch")
    if (
        facts.file_size_bytes != manifest.file_size_bytes
        or facts.row_group_rows != manifest.row_group_rows
        or facts.compression != manifest.compression
    ):
        reasons.append("parquet_metadata_mismatch")
    if facts.compression != ("ZSTD",):
        reasons.append("compression_policy_violation")
    if manifest.missing_sources:
        reasons.append("missing_source")
    accepted_total = sum(receipt.accepted for receipt in manifest.source_receipts)
    excluded_total = sum(receipt.excluded for receipt in manifest.source_receipts)
    scanned_total = sum(receipt.scanned for receipt in manifest.source_receipts)
    if (
        scanned_total != accepted_total + excluded_total
        or accepted_total - manifest.exact_duplicates_removed != frame.height
    ):
        reasons.append("source_receipt_mismatch")
    if expected_sources is None:
        reasons.append("source_contract_missing")
    else:
        reasons.extend(_check_source_contract(manifest, records, expected_sources))
    duplicate = audit_canonical_duplicates(records)
    if (
        duplicate.normalized_collision_count != manifest.normalized_collision_count
        or duplicate.group_collision_count != manifest.group_collision_count
        or len(duplicate.near_duplicate_clusters)
        != manifest.near_duplicate_cluster_count
        or duplicate.near_audited_records != manifest.near_covered_record_count
        or duplicate.near_candidate_comparisons
        != manifest.near_candidate_comparison_count
    ):
        reasons.append("leakage_stats_mismatch")
    if duplicate.publication_blocked:
        reasons.append("leakage_gate_failed")
    return FullCanonicalAuditReport(
        valid=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        record_count=frame.height,
        dataset_sha256=sha256_file(dataset_path),
        manifest_sha256=sha256_file(manifest_path),
        logical_digest=pointer.logical_digest,
        role_counts=actual_roles,
        source_counts=actual["source"],
        language_counts=actual["language"],
        label_counts=actual["label"],
        split_counts=actual["split"],
        row_group_count=facts.row_group_count,
        row_group_rows=facts.row_group_rows,
        compression=facts.compression,
        source_receipts=manifest.source_receipts,
        group_role_intersections=group_intersections(records, "role"),
        group_split_intersections=group_intersections(records, "split"),
        schema_fingerprint=facts.schema_fingerprint,
        file_size_bytes=facts.file_size_bytes,
        raw_bytes=manifest.raw_bytes,
        canonical_size_bytes=facts.file_size_bytes,
        missing_sources=manifest.missing_sources,
    )
