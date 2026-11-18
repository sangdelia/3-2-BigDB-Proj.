"""Pinned local Arrow-cache adapters for UnSmile and KOLD."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from pyarrow import ArrowInvalid, Table, ipc
from pydantic import ValidationError
from typing_extensions import override

from bigdb_mvp.data.boundaries import KoldInput, UnSmileInput
from bigdb_mvp.data.local_sources import STAGING_ROOT
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
from bigdb_mvp.data.staging import read_staged, sha256_file, write_staged


@dataclass(frozen=True, slots=True)
class ArrowSourceSpec:
    """Integrity and provenance contract for one cached Arrow stream."""

    source: SourceName
    upstream: Path
    revision: str
    license: str
    expected_sha256: str
    expected_rows: int
    source_split: str = "train"
    role: CanonicalRole = CanonicalRole.RESEARCH_INDEX
    expected_schema: tuple[str, ...] = ()


UNSMILE_ARROW_SCHEMA: Final = (
    "문장",
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
    "기타 혐오",
    "악플/욕설",
    "clean",
    "개인지칭",
    "labels",
)
KOLD_ARROW_SCHEMA: Final = (
    "guid",
    "source",
    "date",
    "title",
    "comment",
    "OFF",
    "TGT",
    "GRP",
    "OFF_span",
    "TGT_span",
    "raw_labels",
)


ARROW_SOURCES = (
    ArrowSourceSpec(
        source=SourceName.UNSMILE,
        upstream=Path("/home/kyu/.cache/huggingface/datasets")
        / "smilegate-ai___kor_unsmile/default/0.0.0"
        / "6ef17c27881aa44b0d74e9fe29e8d031b40ba07f"
        / "kor_unsmile-train.arrow",
        revision="6ef17c27881aa44b0d74e9fe29e8d031b40ba07f",
        license="CC-BY-NC-ND-4.0",
        expected_sha256=(
            "75f67b6c1c5bf8758df9f7d442f3acc188dc6a2c7ab13577e3b9ea88ad301866"
        ),
        expected_rows=15_005,
        expected_schema=UNSMILE_ARROW_SCHEMA,
    ),
    ArrowSourceSpec(
        source=SourceName.KOLD,
        upstream=Path("/home/kyu/.cache/huggingface/datasets")
        / "nayohan___kold/default/0.0.0"
        / "2e7aa9d2e202db21eb15c3ab10baff60cf7b8a98"
        / "kold-train.arrow",
        revision="2e7aa9d2e202db21eb15c3ab10baff60cf7b8a98",
        license="license-not-declared-local-analysis-only",
        expected_sha256=(
            "f1873f3fa3b8d6a738fcb709538a0c6aa9c058a9921019f6b5002748d8e04b2b"
        ),
        expected_rows=40_429,
        expected_schema=KOLD_ARROW_SCHEMA,
    ),
)


class ArrowSourceError(Exception):
    """Reject unavailable, drifted, or unsupported cached source data."""

    reason_code: str
    source: SourceName

    def __init__(self, reason_code: str, source: SourceName) -> None:
        """Initialize a traceback-compatible source error."""
        self.reason_code = reason_code
        self.source = source
        super().__init__(f"{reason_code}:{source.value}")

    @override
    def __str__(self) -> str:
        """Return a source-scoped stable failure code."""
        return f"{self.reason_code}:{self.source.value}"


def _unsmile_label(row: UnSmileInput) -> BinaryLabel:
    harmful = any(flag == 1 for flag in row.harmful_flags)
    if row.clean == 1 and not harmful:
        return BinaryLabel.CLEAN
    if row.clean == 0 and harmful:
        return BinaryLabel.HARMFUL
    reason = "unsmile_label_conflict" if harmful else "unsmile_label_missing"
    raise RowExclusionError(reason_code=reason)


def _unsmile_records(
    table: Table,
) -> tuple[tuple[SupervisedRecord, ...], Counter[str]]:
    records: list[SupervisedRecord] = []
    exclusions: Counter[str] = Counter()
    for raw_row in table.to_pylist():
        try:
            row = UnSmileInput.model_validate(raw_row)
            records.append(
                build_record(
                    RecordDraft(
                        text=row.text,
                        label=_unsmile_label(row),
                        language=Language.KO,
                        source=SourceName.UNSMILE,
                        source_split="train",
                    )
                )
            )
        except ValidationError:
            exclusions["invalid_schema"] += 1
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
    return tuple(records), exclusions


def _kold_records(
    table: Table,
) -> tuple[tuple[SupervisedRecord, ...], Counter[str]]:
    records: list[SupervisedRecord] = []
    exclusions: Counter[str] = Counter()
    for raw_row in table.to_pylist():
        try:
            row = KoldInput.model_validate(raw_row)
            records.append(
                build_record(
                    RecordDraft(
                        text=row.comment,
                        label=(BinaryLabel.HARMFUL if row.off else BinaryLabel.CLEAN),
                        language=Language.KO,
                        source=SourceName.KOLD,
                        source_split="train",
                        preferred_group_id=f"{row.source}:{row.title}",
                    )
                )
            )
        except ValidationError:
            exclusions["invalid_schema"] += 1
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
    return tuple(records), exclusions


def _load_table(spec: ArrowSourceSpec) -> Table:
    if not spec.upstream.is_file():
        reason = "arrow_source_missing"
        raise ArrowSourceError(reason, spec.source)
    if sha256_file(spec.upstream) != spec.expected_sha256:
        reason = "arrow_source_checksum_drift"
        raise ArrowSourceError(reason, spec.source)
    try:
        table = ipc.open_stream(spec.upstream).read_all()
    except (ArrowInvalid, OSError) as error:
        reason = "arrow_source_invalid"
        raise ArrowSourceError(reason, spec.source) from error
    if table.num_rows != spec.expected_rows:
        reason = "arrow_source_row_count_drift"
        raise ArrowSourceError(reason, spec.source)
    if spec.expected_schema and tuple(table.schema.names) != spec.expected_schema:
        reason = "arrow_source_schema_drift"
        raise ArrowSourceError(reason, spec.source)
    return table


def _normalize(
    spec: ArrowSourceSpec, table: Table
) -> tuple[tuple[SupervisedRecord, ...], Counter[str]]:
    match spec.source:
        case SourceName.UNSMILE:
            return _unsmile_records(table)
        case SourceName.KOLD:
            return _kold_records(table)
        case (
            SourceName.NIKL
            | SourceName.BEEP
            | SourceName.KMHAS
            | SourceName.KHATERS
            | SourceName.CIVIL_COMMENTS
        ):
            reason = "arrow_source_unsupported"
            raise ArrowSourceError(reason, spec.source)
        case unreachable:
            assert_never(unreachable)


def ingest_arrow_source(
    spec: ArrowSourceSpec, staging_root: Path = STAGING_ROOT
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Ingest one verified Arrow stream into independent BigDB staging."""
    staged = read_staged(staging_root, spec.source.value)
    if staged is not None:
        records, receipt = staged
        if (
            receipt.revision == spec.revision
            and receipt.license == spec.license
            and receipt.input_sha256 == spec.expected_sha256
            and receipt.source_split == spec.source_split
            and receipt.role is spec.role
            and receipt.expected_rows == spec.expected_rows
        ):
            return records, receipt
    table = _load_table(spec)
    records, exclusions = _normalize(spec, table)
    if spec.role is CanonicalRole.UNLABELED_INVENTORY:
        exclusions["unlabeled_inventory_label"] += len(records)
        records = ()
    receipt = SourceReceipt(
        source=spec.source,
        revision=spec.revision,
        license=spec.license,
        input_sha256=spec.expected_sha256,
        asset_sha256=spec.expected_sha256,
        asset=spec.upstream.name,
        source_split=spec.source_split,
        role=spec.role,
        expected_rows=spec.expected_rows,
        scanned=spec.expected_rows,
        accepted=len(records),
        excluded=sum(exclusions.values()),
        exclusions=tuple(
            ExclusionCount(reason=reason, count=count)
            for reason, count in sorted(exclusions.items())
        ),
        schema_sha256=schema_digest(
            tuple((name, "arrow") for name in table.schema.names)
        ),
        raw_bytes=spec.upstream.stat().st_size,
        acquisition_mode="full",
    )
    complete = write_staged(staging_root, receipt, records)
    return records, complete
