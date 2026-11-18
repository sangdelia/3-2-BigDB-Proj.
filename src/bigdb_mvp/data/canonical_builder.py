"""Cap-free canonical corpus construction and pair publication."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bigdb_mvp.data.arrow_sources import ARROW_SOURCES, ingest_arrow_source
from bigdb_mvp.data.canonical_path_safety import (
    ensure_publication_directories,
    validate_file,
)
from bigdb_mvp.data.canonical_path_safety import (
    replace_path as safe_replace_path,
)
from bigdb_mvp.data.canonical_publication import (
    DEFAULT_DATASET_NAME,
    DEFAULT_GENERATIONS_DIR,
    CanonicalPointer,
    PublicationIntegrityError,
    PublishedCanonical,
    load_published_canonical,
    publish_generation,
    write_canonical_dataset,
)
from bigdb_mvp.data.canonical_report import counts, group_intersections, parquet_facts
from bigdb_mvp.data.civil_source import ingest_civil_comments_full
from bigdb_mvp.data.dedup_audit import (
    PublicationLeakageError,
    audit_canonical_duplicates,
    deduplicate_canonical_exact,
    normalized_text,
)
from bigdb_mvp.data.local_sources import (
    LOCAL_SOURCES,
    ingest_local_source,
    ingest_nikl_dev,
)
from bigdb_mvp.data.manifest import (
    CanonicalArtifact,
    CanonicalManifest,
    CanonicalRoleCount,
    ExclusionCount,
    ExpectedSourceContract,
    SourceReceipt,
)
from bigdb_mvp.data.models import (
    CanonicalRecord,
    CanonicalRole,
    SupervisedRecord,
)
from bigdb_mvp.data.remote_registry import REMOTE_SOURCES
from bigdb_mvp.data.remote_sources import ingest_remote_source
from bigdb_mvp.data.source_inventory import blocked_inventory_models
from bigdb_mvp.data.staging import sha256_file

__all__ = (
    "CanonicalInput",
    "CanonicalPointer",
    "PublicationIntegrityError",
    "PublishedCanonical",
    "build_full_canonical",
    "build_full_canonical_from_adapters",
    "expected_source_contract",
    "ingest_accepted_sources",
    "load_published_canonical",
    "replace_path",
)

_ReplacePath = Callable[[Path, Path], None]


@dataclass(frozen=True, slots=True)
class CanonicalInput:
    """Accepted records and the role-aware receipt that authorized them."""

    records: tuple[SupervisedRecord, ...]
    receipt: SourceReceipt


def expected_source_contract(
    inputs: tuple[CanonicalInput, ...],
) -> ExpectedSourceContract:
    """Freeze source contracts from trusted adapter/build inputs."""
    return ExpectedSourceContract.from_receipts(tuple(item.receipt for item in inputs))


def replace_path(source: Path, target: Path) -> None:
    """Replace one final path with a prepared same-filesystem path."""
    safe_replace_path(source, target)


def ingest_accepted_sources() -> tuple[CanonicalInput, ...]:
    """Ingest accepted records from every existing source adapter."""
    batches: list[CanonicalInput] = []
    for spec in LOCAL_SOURCES:
        records, receipt = ingest_local_source(spec)
        batches.append(CanonicalInput(records=records, receipt=receipt))
    dev_records, dev_receipt = ingest_nikl_dev()
    train_records = tuple(record for item in batches for record in item.records)
    train_texts = {normalized_text(record.text) for record in train_records}
    train_groups = {record.group_id for record in train_records}
    retained_dev = tuple(
        record
        for record in dev_records
        if normalized_text(record.text) not in train_texts
        and record.group_id not in train_groups
    )
    removed_dev = len(dev_records) - len(retained_dev)
    if removed_dev:
        dev_receipt = dev_receipt.model_copy(
            update={
                "accepted": len(retained_dev),
                "excluded": dev_receipt.excluded + removed_dev,
                "exclusions": (
                    *dev_receipt.exclusions,
                    ExclusionCount(reason="query_leakage_excluded", count=removed_dev),
                ),
            }
        )
    batches.append(CanonicalInput(records=retained_dev, receipt=dev_receipt))
    for spec in REMOTE_SOURCES:
        records, receipt = ingest_remote_source(spec)
        batches.append(CanonicalInput(records=records, receipt=receipt))
    for spec in ARROW_SOURCES:
        records, receipt = ingest_arrow_source(spec)
        batches.append(CanonicalInput(records=records, receipt=receipt))
    records, receipt = ingest_civil_comments_full()
    batches.append(CanonicalInput(records=records, receipt=receipt))
    return tuple(batches)


def _canonicalize(
    inputs: tuple[CanonicalInput, ...], profile: Literal["research", "production"]
) -> tuple[CanonicalRecord, ...]:
    rows: list[CanonicalRecord] = []
    for item in inputs:
        role = item.receipt.role if profile == "research" else CanonicalRole.PRODUCTION
        if role is None:
            reason = "source_role_missing"
            raise ValueError(reason)
        for record in item.records:
            if record.source is not item.receipt.source:
                reason = "source_provenance_mismatch"
                raise ValueError(reason)
            if record.source_split != item.receipt.source_split:
                reason = "source_split_provenance_mismatch"
                raise ValueError(reason)
            rows.append(
                CanonicalRecord(
                    text=record.text,
                    label=record.label,
                    language=record.language,
                    source=record.source,
                    source_split=record.source_split,
                    role=role,
                    group_id=record.group_id,
                    text_sha256=record.text_sha256,
                )
            )
    return tuple(sorted(rows, key=CanonicalRecord.stable_sort_key))


def _logical_digest(rows: tuple[CanonicalRecord, ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _role_counts(rows: tuple[CanonicalRecord, ...]) -> tuple[CanonicalRoleCount, ...]:
    counts = Counter(row.role for row in rows)
    return tuple(
        CanonicalRoleCount(role=role, count=counts[role])
        for role in CanonicalRole
        if counts[role]
    )


def build_full_canonical(
    inputs: tuple[CanonicalInput, ...],
    *,
    output_root: Path,
    seed: int,
    profile: Literal["research", "production"] = "research",
) -> CanonicalManifest:
    """Build and atomically publish the cap-free canonical artifact pair."""
    effective_inputs = tuple(
        CanonicalInput(
            records=item.records,
            receipt=item.receipt.model_copy(
                update={
                    "role": item.receipt.role
                    if profile == "research"
                    else CanonicalRole.PRODUCTION
                }
            ),
        )
        for item in inputs
    )
    rows = _canonicalize(effective_inputs, profile)
    audit = audit_canonical_duplicates(rows)
    if audit.publication_blocked:
        raise PublicationLeakageError(audit.blocking_reasons)
    rows, exact_duplicates_removed = deduplicate_canonical_exact(rows)
    published_audit = audit_canonical_duplicates(rows)
    ensure_publication_directories(output_root)
    dataset_tmp = output_root / f".{DEFAULT_DATASET_NAME}.tmp"
    try:
        validate_file(dataset_tmp, allow_missing=True)
        write_canonical_dataset(rows, dataset_tmp)
        dataset_sha = sha256_file(dataset_tmp)
        logical_digest = _logical_digest(rows)
        facts = parquet_facts(dataset_tmp)
        dimensions = counts(rows)
        manifest = CanonicalManifest(
            seed=seed,
            profile=profile,
            dataset_sha256=dataset_sha,
            logical_digest=logical_digest,
            record_count=len(rows),
            role_counts=_role_counts(rows),
            source_receipts=tuple(
                sorted(
                    (item.receipt for item in effective_inputs),
                    key=lambda receipt: (receipt.source.value, receipt.role.value),
                )
            ),
            artifact=CanonicalArtifact(
                path=(
                    f"{DEFAULT_GENERATIONS_DIR}/{logical_digest}/{DEFAULT_DATASET_NAME}"
                ),
                row_count=len(rows),
                sha256=dataset_sha,
            ),
            source_counts=dimensions["source"],
            language_counts=dimensions["language"],
            label_counts=dimensions["label"],
            split_counts=dimensions["split"],
            group_role_intersections=group_intersections(rows, "role"),
            group_split_intersections=group_intersections(rows, "split"),
            schema_fingerprint=facts.schema_fingerprint,
            file_size_bytes=facts.file_size_bytes,
            row_group_count=facts.row_group_count,
            row_group_rows=facts.row_group_rows,
            compression=facts.compression,
            raw_bytes=sum(item.receipt.raw_bytes for item in effective_inputs),
            canonical_size_bytes=facts.file_size_bytes,
            exact_duplicates_removed=exact_duplicates_removed,
            normalized_collision_count=published_audit.normalized_collision_count,
            group_collision_count=published_audit.group_collision_count,
            near_duplicate_cluster_count=len(published_audit.near_duplicate_clusters),
            near_covered_record_count=published_audit.near_audited_records,
            near_candidate_comparison_count=published_audit.near_candidate_comparisons,
            blocked_inventory=blocked_inventory_models(),
        )
        publish_generation(
            rows,
            manifest,
            output_root=output_root,
            replace=replace_path,
            trusted_contract=expected_source_contract(effective_inputs),
        )
    finally:
        _ = dataset_tmp.unlink(missing_ok=True)
    return manifest


def build_full_canonical_from_adapters(
    *,
    output_root: Path,
    seed: int,
    profile: Literal["research", "production"] = "research",
) -> CanonicalManifest:
    """Build the full canonical artifact from the configured adapters."""
    return build_full_canonical(
        ingest_accepted_sources(), output_root=output_root, seed=seed, profile=profile
    )
