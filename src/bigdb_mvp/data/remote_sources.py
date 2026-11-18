"""Pinned official train-source acquisition and normalization."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final, assert_never

from pydantic import ValidationError

from bigdb_mvp.data.boundaries import BeepInput, KhatersInput, KmhasInput
from bigdb_mvp.data.manifest import ExclusionCount, SourceReceipt, schema_digest
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    RecordDraft,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.normalize import RowExclusionError, build_record, khaters_label
from bigdb_mvp.data.remote_integrity import (
    RemoteSourceError,
    download_remote_assets,
    read_matching_stage,
    validate_remote_schema,
)
from bigdb_mvp.data.staging import write_staged

if TYPE_CHECKING:
    from bigdb_mvp.data.remote_registry import RemoteSourceSpec

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
STAGING_ROOT: Final = PROJECT_ROOT / "data/raw/staging"
BEEP_LABELS: Final = {
    "none": BinaryLabel.CLEAN,
    "offensive": BinaryLabel.HARMFUL,
    "hate": BinaryLabel.HARMFUL,
}
BEEP_ASSET_COUNT: Final = 2


def _source_digest(spec: RemoteSourceSpec) -> str:
    if len(spec.assets) == 1:
        return spec.assets[0].expected_sha256
    digest = hashlib.sha256()
    for asset in spec.assets:
        digest.update(asset.expected_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _label_kmhas(raw: str) -> BinaryLabel:
    labels = {item.strip() for item in raw.split(",") if item.strip()}
    harmful = {str(index) for index in range(8)}
    if labels == {"8"}:
        return BinaryLabel.CLEAN
    if "8" in labels and labels & harmful:
        reason = "kmhas_label_conflict"
        raise RowExclusionError(reason)
    if labels and labels <= harmful:
        return BinaryLabel.HARMFUL
    reason = "kmhas_label_unknown"
    raise RowExclusionError(reason)


def _label_beep(raw: str) -> BinaryLabel:
    label = BEEP_LABELS.get(raw.strip().lower())
    if label is None:
        reason = "beep_label_unknown"
        raise RowExclusionError(reason)
    return label


def _require_beep_title(title: str) -> str:
    if not title.strip():
        reason = "beep_article_title_missing"
        raise RowExclusionError(reason)
    return title


def _append_record(
    records: list[SupervisedRecord],
    exclusions: Counter[str],
    draft: RecordDraft,
) -> None:
    try:
        records.append(build_record(draft))
    except RowExclusionError as error:
        exclusions[error.reason_code] += 1


def _ingest_kmhas(
    path: Path, source_split: str
) -> tuple[list[SupervisedRecord], Counter[str], int]:
    records: list[SupervisedRecord] = []
    exclusions: Counter[str] = Counter()
    scanned = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for payload in csv.DictReader(handle, delimiter="\t"):
            scanned += 1
            try:
                row = KmhasInput.model_validate(payload)
                label = _label_kmhas(row.label)
            except ValidationError:
                exclusions["invalid_schema"] += 1
                continue
            except RowExclusionError as error:
                exclusions[error.reason_code] += 1
                continue
            _append_record(
                records,
                exclusions,
                RecordDraft(
                    row.document,
                    label,
                    Language.KO,
                    SourceName.KMHAS,
                    source_split,
                ),
            )
    return records, exclusions, scanned


def _ingest_beep(
    comments_path: Path, titles_path: Path, source_split: str
) -> tuple[list[SupervisedRecord], Counter[str], int]:
    with comments_path.open(encoding="utf-8", newline="") as handle:
        payloads = list(csv.DictReader(handle, delimiter="\t"))
    titles = titles_path.read_text(encoding="utf-8").splitlines()
    if len(payloads) != len(titles):
        reason = "beep_asset_row_mismatch"
        raise RemoteSourceError(reason, SourceName.BEEP)
    records: list[SupervisedRecord] = []
    exclusions: Counter[str] = Counter()
    for payload, title in zip(payloads, titles, strict=True):
        try:
            row = BeepInput.model_validate(payload)
            label = _label_beep(row.hate)
            verified_title = _require_beep_title(title)
        except ValidationError:
            exclusions["invalid_schema"] += 1
            continue
        except RowExclusionError as error:
            exclusions[error.reason_code] += 1
            continue
        _append_record(
            records,
            exclusions,
            RecordDraft(
                row.comments,
                label,
                Language.KO,
                SourceName.BEEP,
                source_split,
                verified_title,
            ),
        )
    return records, exclusions, len(payloads)


def _ingest_khaters(
    path: Path, source_split: str
) -> tuple[list[SupervisedRecord], Counter[str], int]:
    records: list[SupervisedRecord] = []
    exclusions: Counter[str] = Counter()
    scanned = 0
    with path.open(encoding="utf-8") as handle:
        for payload in handle:
            if not payload.strip():
                continue
            scanned += 1
            try:
                row = KhatersInput.model_validate_json(payload)
                label = khaters_label(row.label)
            except ValidationError:
                exclusions["invalid_schema"] += 1
                continue
            except RowExclusionError as error:
                exclusions[error.reason_code] += 1
                continue
            _append_record(
                records,
                exclusions,
                RecordDraft(
                    row.text, label, Language.KO, SourceName.KHATERS, source_split
                ),
            )
    return records, exclusions, scanned


def ingest_remote_source(
    spec: RemoteSourceSpec, staging_root: Path = STAGING_ROOT
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Reuse matching staging or acquire and normalize one pinned source."""
    expected_digest = _source_digest(spec)
    staged = read_matching_stage(spec, expected_digest, staging_root)
    if staged is not None:
        return staged
    paths = download_remote_assets(spec)
    records, exclusions, scanned = _normalize_remote(spec, paths)
    if spec.expected_rows is not None and scanned != spec.expected_rows:
        reason = "remote_source_row_count_drift"
        raise RemoteSourceError(reason, spec.source)
    if spec.role is CanonicalRole.UNLABELED_INVENTORY:
        exclusions["unlabeled_inventory_label"] += len(records)
        records = []
    receipt = SourceReceipt(
        source=spec.source,
        revision=spec.revision,
        license=spec.license,
        input_sha256=expected_digest,
        asset_sha256=expected_digest,
        asset=spec.repository,
        source_split=spec.source_split,
        role=spec.role,
        expected_rows=spec.expected_rows,
        scanned=scanned,
        accepted=len(records),
        excluded=sum(exclusions.values()),
        exclusions=tuple(
            ExclusionCount(reason=reason, count=count)
            for reason, count in sorted(exclusions.items())
        ),
        schema_sha256=schema_digest(
            tuple((name, "string") for name in spec.expected_schema)
        ),
        raw_bytes=sum(path.stat().st_size for path in paths),
        acquisition_mode="full",
    )
    complete = write_staged(staging_root, receipt, tuple(records))
    return tuple(records), complete


def _normalize_remote(
    spec: RemoteSourceSpec, paths: tuple[Path, ...]
) -> tuple[list[SupervisedRecord], Counter[str], int]:
    validate_remote_schema(spec, paths)
    match spec.source:
        case SourceName.BEEP:
            if len(paths) != BEEP_ASSET_COUNT:
                reason = "beep_asset_count_drift"
                raise RemoteSourceError(reason, spec.source)
            return _ingest_beep(paths[0], paths[1], spec.source_split)
        case SourceName.KMHAS:
            if len(paths) != 1:
                reason = "kmhas_asset_count_drift"
                raise RemoteSourceError(reason, spec.source)
            return _ingest_kmhas(paths[0], spec.source_split)
        case SourceName.KHATERS:
            if len(paths) != 1:
                reason = "khaters_asset_count_drift"
                raise RemoteSourceError(reason, spec.source)
            return _ingest_khaters(paths[0], spec.source_split)
        case (
            SourceName.NIKL
            | SourceName.UNSMILE
            | SourceName.KOLD
            | SourceName.CIVIL_COMMENTS
        ):
            reason = f"source_not_remote:{spec.source.value}"
            raise RemoteSourceError(reason, spec.source)
        case _ as unreachable:
            assert_never(unreachable)
