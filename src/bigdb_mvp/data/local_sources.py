"""One-time read-only ingest for approved Korean train sources."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from pydantic import TypeAdapter, ValidationError

from bigdb_mvp.data.boundaries import ExternalInput, NiklInput
from bigdb_mvp.data.manifest import ExclusionCount, SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    Language,
    RecordDraft,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.normalize import RowExclusionError, build_record, khaters_label
from bigdb_mvp.data.staging import read_staged, write_staged

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGING_ROOT = PROJECT_ROOT / "data" / "raw" / "staging"
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


LOCAL_SOURCES = (
    LocalSourceSpec(
        SourceName.NIKL,
        Path("/home/kyu/Coulmn/hate-speech-sib/experiments/data/nikl_train.json"),
        "restricted_local_review_required",
    ),
    LocalSourceSpec(
        SourceName.BEEP,
        Path("/home/kyu/Coulmn")
        / "hate-speech-grad-thesis/experiments/data/external/beep_kocohub.jsonl",
        "CC-BY-SA-4.0",
    ),
    LocalSourceSpec(
        SourceName.KMHAS,
        Path("/home/kyu/Coulmn")
        / "hate-speech-grad-thesis/experiments/data/external/kmhas.jsonl",
        "CC-BY-SA-4.0",
    ),
    LocalSourceSpec(
        SourceName.KHATERS,
        Path("/home/kyu/Coulmn")
        / "hate-speech-grad-thesis/experiments/data/external/khaters.jsonl",
        "CC-BY-4.0",
    ),
)


class LocalSourceMissingError(Exception):
    """Report an unavailable authorized local input before staging exists."""

    source: SourceName

    def __init__(self, source: SourceName) -> None:
        """Initialize a source-specific missing-input failure."""
        self.source = source
        super().__init__(f"local_source_missing:{source.value}")


def _binary_cached_label(raw: str) -> BinaryLabel:
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
                    source_split="train",
                    preferred_group_id=row.doc_id,
                )
            )
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
    return tuple(drafts), exclusions, len(rows)


def _external_draft(spec: LocalSourceSpec, row: ExternalInput) -> RecordDraft:
    metadata = row.metadata
    if metadata is None:
        reason = (
            "khaters_original_label_missing"
            if spec.source is SourceName.KHATERS
            else "metadata_missing"
        )
        raise RowExclusionError(reason)
    if metadata.split.strip().lower() != "train":
        reason = "non_train_split"
        raise RowExclusionError(reason)
    match spec.source:
        case SourceName.BEEP:
            label = _binary_cached_label(row.label)
            preferred_id = metadata.title
        case SourceName.KMHAS:
            label = _binary_cached_label(row.label)
            preferred_id = None
        case SourceName.KHATERS:
            original = metadata.original_label
            match original:
                case str() as original_text:
                    label = khaters_label(original_text)
                case tuple() | None:
                    label = khaters_label(None)
                case _ as unreachable:
                    assert_never(unreachable)
            preferred_id = None
        case SourceName.NIKL | SourceName.CIVIL_COMMENTS:
            reason = "source_not_external"
            raise RowExclusionError(reason)
        case _ as unreachable:
            assert_never(unreachable)
    return RecordDraft(
        text=row.target_form,
        label=label,
        language=Language.KO,
        source=spec.source,
        source_split="train",
        preferred_group_id=preferred_id,
    )


def _ingest_external(
    spec: LocalSourceSpec, payload: bytes
) -> tuple[tuple[RecordDraft, ...], Counter[str], int]:
    drafts: list[RecordDraft] = []
    exclusions: Counter[str] = Counter()
    lines = tuple(line for line in payload.splitlines() if line)
    for line in lines:
        try:
            row = ExternalInput.model_validate_json(line)
            drafts.append(_external_draft(spec, row))
        except ValidationError:
            exclusions["invalid_schema"] += 1
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
    return tuple(drafts), exclusions, len(lines)


def ingest_local_source(
    spec: LocalSourceSpec,
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Reuse verified staging or ingest one authorized local source once."""
    staged = read_staged(STAGING_ROOT, spec.source.value)
    if staged is not None:
        return staged
    if not spec.upstream.is_file():
        raise LocalSourceMissingError(spec.source)
    payload = spec.upstream.read_bytes()
    match spec.source:
        case SourceName.NIKL:
            drafts, exclusions, scanned = _ingest_nikl(spec, payload)
        case SourceName.BEEP | SourceName.KMHAS | SourceName.KHATERS:
            drafts, exclusions, scanned = _ingest_external(spec, payload)
        case SourceName.CIVIL_COMMENTS:
            reason = "source_not_local"
            raise RowExclusionError(reason)
        case _ as unreachable:
            assert_never(unreachable)
    records = tuple(build_record(draft) for draft in drafts)
    receipt = SourceReceipt(
        source=spec.source,
        revision=hashlib.sha256(payload).hexdigest(),
        license=spec.license,
        input_sha256=hashlib.sha256(payload).hexdigest(),
        scanned=scanned,
        accepted=len(records),
        excluded=sum(exclusions.values()),
        exclusions=_exclusion_tuple(exclusions),
    )
    complete = write_staged(STAGING_ROOT, receipt, records)
    return records, complete
