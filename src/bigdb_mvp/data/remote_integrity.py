"""Integrity and staging boundaries for remote source adapters."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, assert_never

from bigdb_mvp.data.boundaries import KhatersInput
from bigdb_mvp.data.civil_download import (
    DownloadIntegrityError,
    download_immutable_shard,
)
from bigdb_mvp.data.models import SourceName
from bigdb_mvp.data.staging import read_staged

if TYPE_CHECKING:
    from pathlib import Path

    from bigdb_mvp.data.manifest import SourceReceipt
    from bigdb_mvp.data.models import SupervisedRecord
    from bigdb_mvp.data.remote_registry import RemoteSourceSpec


class RemoteSourceError(Exception):
    """Report a source-scoped remote acquisition failure."""

    reason_code: str
    source: SourceName

    def __init__(self, reason_code: str, source: SourceName) -> None:
        """Initialize a source-scoped remote failure."""
        self.reason_code = reason_code
        self.source = source
        super().__init__(f"{reason_code}:{source.value}")


def read_matching_stage(
    spec: RemoteSourceSpec, expected_digest: str, staging_root: Path
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt] | None:
    """Return a stage only when every role-aware pin matches."""
    staged = read_staged(staging_root, spec.source.value)
    if staged is None:
        return None
    records, receipt = staged
    if (
        receipt.revision == spec.revision
        and receipt.input_sha256 == expected_digest
        and receipt.license == spec.license
        and receipt.asset == spec.repository
        and receipt.source_split == spec.source_split
        and receipt.role is spec.role
        and (spec.expected_rows is None or receipt.expected_rows == spec.expected_rows)
    ):
        return records, receipt
    return None


def download_remote_assets(spec: RemoteSourceSpec) -> tuple[Path, ...]:
    """Acquire all remote assets only after each cache passes its pin."""
    for asset in spec.assets:
        try:
            _ = download_immutable_shard(asset)
        except DownloadIntegrityError as error:
            raise RemoteSourceError(error.reason_code, spec.source) from error
    return tuple(asset.target for asset in spec.assets)


def validate_remote_schema(spec: RemoteSourceSpec, paths: tuple[Path, ...]) -> None:
    """Reject a remote asset whose pinned header schema has drifted."""
    if not spec.expected_schema:
        return
    if not paths:
        reason = "remote_source_asset_missing"
        raise RemoteSourceError(reason, spec.source)
    match spec.source:
        case SourceName.BEEP | SourceName.KMHAS:
            with paths[0].open(encoding="utf-8", newline="") as handle:
                actual = tuple(next(csv.reader(handle, delimiter="\t"), ()))
        case SourceName.KHATERS:
            actual = tuple(KhatersInput.model_fields)
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
    if actual != spec.expected_schema:
        reason = "remote_source_schema_drift"
        raise RemoteSourceError(reason, spec.source)
