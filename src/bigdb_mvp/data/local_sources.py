"""One-time read-only ingest for approved Korean train sources."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from pydantic import TypeAdapter

from bigdb_mvp.data.boundaries import NiklInput
from bigdb_mvp.data.manifest import ExclusionCount, SourceReceipt, schema_digest
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    RecordDraft,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.normalize import RowExclusionError, build_record
from bigdb_mvp.data.staging import read_staged, write_staged

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGING_ROOT = PROJECT_ROOT / "data" / "raw" / "staging"
NIKL_LOCAL_DIGEST: Final = (
    "73d9558e06a057f11d2910dfc13b57e287128c6b3aeba649b28a560c79fd5b8f"
)
BINARY_LABELS: Final = {
    "appropriate": BinaryLabel.CLEAN,
    "inappropriate": BinaryLabel.HARMFUL,
}


@dataclass(frozen=True, slots=True)
class LocalSourceSpec:
    """Immutable provenance needed for a one-time local ingest."""

    source: SourceName
    upstream: Path
    license: str
    revision: str = ""
    expected_sha256: str = ""
    expected_rows: int | None = None
    source_split: str = "train"
    role: CanonicalRole = CanonicalRole.RESEARCH_INDEX


LOCAL_SOURCES = (
    LocalSourceSpec(
        SourceName.NIKL,
        Path("/home/kyu/Coulmn/hate-speech-sib/experiments/data/nikl_train.json"),
        "restricted_local_review_required",
        revision=NIKL_LOCAL_DIGEST,
        expected_sha256=NIKL_LOCAL_DIGEST,
        expected_rows=2_109,
    ),
)


class LocalSourceError(Exception):
    """Report a source-scoped local acquisition failure."""

    reason_code: str
    source: SourceName

    def __init__(self, reason_code: str, source: SourceName) -> None:
        """Initialize a source-scoped local failure."""
        self.reason_code = reason_code
        self.source = source
        super().__init__(f"{reason_code}:{source.value}")


class LocalSourceMissingError(LocalSourceError):
    """Report an unavailable authorized local input before staging exists."""

    source: SourceName

    def __init__(self, source: SourceName) -> None:
        """Initialize a source-specific missing-input failure."""
        super().__init__("local_source_missing", source)


def _binary_cached_label(raw: str | None) -> BinaryLabel:
    if raw is None:
        reason = "unlabeled_row"
        raise RowExclusionError(reason)
    label = BINARY_LABELS.get(raw.strip().lower())
    if label is None:
        reason = "unknown_binary_label"
        raise RowExclusionError(reason)
    return label


def _exclusion_tuple(counts: Counter[str]) -> tuple[ExclusionCount, ...]:
    return tuple(
        ExclusionCount(reason=reason, count=count)
        for reason, count in sorted(counts.items())
    )


def _ingest_nikl(
    spec: LocalSourceSpec, payload: bytes
) -> tuple[tuple[RecordDraft, ...], Counter[str], int]:
    rows = TypeAdapter(tuple[NiklInput, ...]).validate_json(payload)
    drafts: list[RecordDraft] = []
    exclusions: Counter[str] = Counter()
    for row in rows:
        try:
            drafts.append(
                RecordDraft(
                    text=row.target_form,
                    label=_binary_cached_label(row.label),
                    language=Language.KO,
                    source=spec.source,
                    source_split=spec.source_split,
                    preferred_group_id=row.doc_id,
                )
            )
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
    return tuple(drafts), exclusions, len(rows)


def ingest_local_source(
    spec: LocalSourceSpec,
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Reuse verified staging or ingest one authorized local source once."""
    staged = read_staged(STAGING_ROOT, spec.source.value)
    if staged is not None:
        records, receipt = staged
        if not (
            receipt.revision != spec.revision
            or receipt.license != spec.license
            or receipt.input_sha256 != spec.expected_sha256
            or receipt.asset_sha256 != spec.expected_sha256
            or receipt.asset != spec.upstream.name
            or receipt.source_split != spec.source_split
            or receipt.role is not spec.role
            or (
                spec.expected_rows is not None
                and receipt.expected_rows != spec.expected_rows
            )
        ):
            return records, receipt
    if not spec.upstream.is_file():
        raise LocalSourceMissingError(spec.source)
    payload = spec.upstream.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if spec.expected_sha256 and digest != spec.expected_sha256:
        reason = "local_source_checksum_drift"
        raise LocalSourceError(reason, spec.source)
    match spec.source:
        case SourceName.NIKL:
            drafts, exclusions, scanned = _ingest_nikl(spec, payload)
        case (
            SourceName.BEEP
            | SourceName.KMHAS
            | SourceName.KHATERS
            | SourceName.UNSMILE
            | SourceName.KOLD
            | SourceName.CIVIL_COMMENTS
        ):
            reason = "source_not_local"
            raise LocalSourceError(reason, spec.source)
        case _ as unreachable:
            assert_never(unreachable)
    records = tuple(build_record(draft) for draft in drafts)
    receipt = SourceReceipt(
        source=spec.source,
        revision=spec.revision or digest,
        license=spec.license,
        input_sha256=digest,
        asset_sha256=digest,
        asset=spec.upstream.name,
        source_split=spec.source_split,
        role=spec.role,
        expected_rows=spec.expected_rows,
        scanned=scanned,
        accepted=len(records),
        excluded=sum(exclusions.values()),
        exclusions=_exclusion_tuple(exclusions),
        schema_sha256=schema_digest(
            tuple((name, "json") for name in NiklInput.model_fields)
        ),
        raw_bytes=len(payload),
        acquisition_mode="full",
    )
    if spec.role is CanonicalRole.UNLABELED_INVENTORY:
        exclusions["unlabeled_inventory_label"] += len(records)
        records = ()
        receipt = receipt.model_copy(
            update={"accepted": 0, "excluded": sum(exclusions.values())}
        )
    complete = write_staged(STAGING_ROOT, receipt, records)
    return records, complete


def ingest_nikl_dev() -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Ingest the authorized, independently hashed NIKL development asset."""
    path = Path("/home/kyu/Coulmn/hate-speech-sib/experiments/data/nikl_dev.json")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    drafts, exclusions, scanned = _ingest_nikl(
        LocalSourceSpec(
            source=SourceName.NIKL,
            upstream=path,
            license="restricted_local_review_required",
            source_split="train",
            role=CanonicalRole.CALIBRATION_QUERY,
        ),
        payload,
    )
    records = tuple(
        build_record(draft).model_copy(update={"source_split": "dev"})
        for draft in drafts
    )
    receipt = SourceReceipt(
        source=SourceName.NIKL,
        revision=digest,
        license="restricted_local_review_required",
        input_sha256=digest,
        asset_sha256=digest,
        asset=path.name,
        staging_sha256=digest,
        source_split="dev",
        role=CanonicalRole.CALIBRATION_QUERY,
        expected_rows=scanned,
        schema_sha256=schema_digest(
            tuple((name, "json") for name in NiklInput.model_fields)
        ),
        scanned=scanned,
        accepted=len(records),
        excluded=sum(exclusions.values()),
        raw_bytes=len(payload),
        exclusions=_exclusion_tuple(exclusions),
        acquisition_mode="full",
    )
    return records, receipt
