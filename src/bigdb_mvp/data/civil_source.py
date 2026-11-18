"""Pinned immutable Parquet acquisition for Civil Comments train data."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from bigdb_mvp.data.civil_download import (
    DownloadArea,
    DownloadSpec,
    download_immutable_shard,
)
from bigdb_mvp.data.civil_parquet import CivilScanLimits, scan_civil_parquet
from bigdb_mvp.data.manifest import ExclusionCount, SourceReceipt, schema_digest
from bigdb_mvp.data.models import CanonicalRole, SourceName, SupervisedRecord
from bigdb_mvp.data.staging import read_staged, write_staged

CIVIL_REPOSITORY: Final = "google/civil_comments"
CIVIL_REVISION: Final = "f2970eb3a55777454c94069077cc8d9b5866312d"
CIVIL_LICENSE: Final = "CC0-1.0"
CIVIL_SHARD_NAME: Final = "train-00000-of-00002.parquet"
CIVIL_SHARD_SHA256: Final = (
    "c20f01c3aecbdd942886cacb0ee67e995df33bc712a622fde75927ed3d6ccefe"
)
CIVIL_SHARD_BYTES: Final = 193_527_147
CIVIL_SHARD_URL: Final = (
    "https://huggingface.co/datasets/google/civil_comments/resolve/"
    f"{CIVIL_REVISION}/data/{CIVIL_SHARD_NAME}"
)
MAX_SCANNED_ROWS: Final = 100_000
PER_LABEL_TARGET: Final = 2_500
PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
STAGING_ROOT: Final = PROJECT_ROOT / "data/raw/staging"
SHARD_PATH: Final = STAGING_ROOT / f"civil-comments-{CIVIL_REVISION}.parquet"


class CivilAcquisitionError(Exception):
    """Report a bounded immutable shard that cannot fill required label cells."""

    reason_code: str
    scanned: int

    def __init__(self, reason_code: str, scanned: int) -> None:
        """Initialize a bounded acquisition failure."""
        self.reason_code = reason_code
        self.scanned = scanned
        super().__init__(f"{reason_code}:scanned={scanned}")


class CivilSourceError(Exception):
    """Report stale staged metadata for the pinned Civil source."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize a typed Civil source boundary failure."""
        self.reason_code = reason_code
        super().__init__(reason_code)


def ingest_civil_comments() -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Reuse verified sampled rows or acquire the sampled immutable shard."""
    return _ingest_civil(
        per_label_target=PER_LABEL_TARGET,
        max_scanned_rows=MAX_SCANNED_ROWS,
        mode="sampled",
    )


def ingest_civil_comments_full() -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Acquire every valid row from the approved Civil shard without a quota."""
    return _ingest_civil(per_label_target=None, max_scanned_rows=None, mode="full")


def _ingest_civil(
    *,
    per_label_target: int | None,
    max_scanned_rows: int | None,
    mode: Literal["sampled", "full"],
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    staged = read_staged(STAGING_ROOT, SourceName.CIVIL_COMMENTS.value)
    if staged is not None:
        _, receipt = staged
        if (
            receipt.revision != CIVIL_REVISION
            or receipt.license != CIVIL_LICENSE
            or receipt.input_sha256 != CIVIL_SHARD_SHA256
            or receipt.asset_sha256 != CIVIL_SHARD_SHA256
            or receipt.asset != CIVIL_SHARD_URL
            or receipt.source_split != "train"
            or receipt.role is not CanonicalRole.RESEARCH_INDEX
            or receipt.expected_rows is None
            or receipt.expected_rows != receipt.scanned
            or receipt.scanned != receipt.accepted + receipt.excluded
            or receipt.acquisition_mode != mode
        ):
            staged = None
        else:
            return staged
    shard_sha = download_immutable_shard(
        DownloadSpec(
            url=CIVIL_SHARD_URL,
            target=SHARD_PATH,
            expected_sha256=CIVIL_SHARD_SHA256,
            expected_bytes=CIVIL_SHARD_BYTES,
            area=DownloadArea.STAGING,
        )
    )
    batch = scan_civil_parquet(
        SHARD_PATH,
        CivilScanLimits(
            per_label_target=per_label_target,
            max_scanned_rows=max_scanned_rows,
        ),
    )
    if per_label_target is not None and (
        batch.clean_count < per_label_target or batch.harmful_count < per_label_target
    ):
        reason = "civil_bounded_quota_unmet"
        raise CivilAcquisitionError(reason, batch.scanned)
    invalid_excluded = sum(item.count for item in batch.exclusions)
    quota_excluded = (
        batch.scanned - len(batch.records) - invalid_excluded
        if per_label_target is not None
        else 0
    )
    exclusions = (
        (
            *batch.exclusions,
            ExclusionCount(reason="civil_quota_capped", count=quota_excluded),
        )
        if quota_excluded
        else batch.exclusions
    )
    receipt = SourceReceipt(
        source=SourceName.CIVIL_COMMENTS,
        revision=CIVIL_REVISION,
        license=CIVIL_LICENSE,
        input_sha256=shard_sha,
        asset_sha256=shard_sha,
        asset=CIVIL_SHARD_URL,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=batch.scanned,
        scanned=batch.scanned,
        accepted=len(batch.records),
        excluded=sum(item.count for item in exclusions),
        exclusions=exclusions,
        schema_sha256=schema_digest((("text", "string"), ("toxicity", "float"))),
        raw_bytes=SHARD_PATH.stat().st_size,
        acquisition_mode=mode,
    )
    complete = write_staged(STAGING_ROOT, receipt, batch.records)
    return batch.records, complete
