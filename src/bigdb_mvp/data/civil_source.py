"""Pinned immutable Parquet acquisition for Civil Comments train data."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from bigdb_mvp.data.civil_download import DownloadSpec, download_immutable_shard
from bigdb_mvp.data.civil_parquet import CivilScanLimits, scan_civil_parquet
from bigdb_mvp.data.manifest import SourceReceipt
from bigdb_mvp.data.models import SourceName, SupervisedRecord
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


def ingest_civil_comments() -> tuple[tuple[SupervisedRecord, ...], SourceReceipt]:
    """Reuse verified rows or download and locally scan one immutable shard."""
    staged = read_staged(STAGING_ROOT, SourceName.CIVIL_COMMENTS.value)
    if staged is not None:
        return staged
    shard_sha = download_immutable_shard(
        DownloadSpec(
            url=CIVIL_SHARD_URL,
            target=SHARD_PATH,
            expected_sha256=CIVIL_SHARD_SHA256,
            expected_bytes=CIVIL_SHARD_BYTES,
        )
    )
    batch = scan_civil_parquet(
        SHARD_PATH,
        CivilScanLimits(
            per_label_target=PER_LABEL_TARGET,
            max_scanned_rows=MAX_SCANNED_ROWS,
        ),
    )
    if batch.clean_count < PER_LABEL_TARGET or batch.harmful_count < PER_LABEL_TARGET:
        reason = "civil_bounded_quota_unmet"
        raise CivilAcquisitionError(reason, batch.scanned)
    receipt = SourceReceipt(
        source=SourceName.CIVIL_COMMENTS,
        revision=CIVIL_REVISION,
        license=CIVIL_LICENSE,
        input_sha256=shard_sha,
        asset=CIVIL_SHARD_URL,
        scanned=batch.scanned,
        accepted=len(batch.records),
        excluded=sum(item.count for item in batch.exclusions),
        exclusions=batch.exclusions,
    )
    complete = write_staged(STAGING_ROOT, receipt, batch.records)
    return batch.records, complete
