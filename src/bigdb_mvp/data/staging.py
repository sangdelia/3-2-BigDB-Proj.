"""Integrity-checked internal staging storage."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from bigdb_mvp.data.manifest import SourceReceipt
from bigdb_mvp.data.models import SupervisedRecord


class StagingIntegrityError(Exception):
    """Report incomplete or checksum-drifted staging state."""

    reason_code: str
    source: str

    def __init__(self, reason_code: str, source: str) -> None:
        """Initialize a source-specific staging-integrity failure."""
        self.reason_code = reason_code
        self.source = source
        super().__init__(f"{reason_code}:{source}")


def sha256_file(path: Path) -> str:
    """Hash a file without loading a large artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_paths(root: Path, source: str) -> tuple[Path, Path]:
    """Resolve paired records and receipt paths for one source."""
    return root / f"{source}.jsonl", root / f"{source}.receipt.json"


def read_staged(
    root: Path, source: str
) -> tuple[tuple[SupervisedRecord, ...], SourceReceipt] | None:
    """Load a complete staging pair after checking its records digest."""
    records_path, receipt_path = stage_paths(root, source)
    if not records_path.exists() and not receipt_path.exists():
        return None
    if not records_path.exists() or not receipt_path.exists():
        reason = "staging_incomplete"
        raise StagingIntegrityError(reason, source)
    receipt = SourceReceipt.model_validate_json(receipt_path.read_bytes())
    actual_sha = sha256_file(records_path)
    if actual_sha != receipt.staging_sha256:
        reason = "staging_checksum_drift"
        raise StagingIntegrityError(reason, source)
    records = tuple(
        SupervisedRecord.model_validate_json(line)
        for line in records_path.read_bytes().splitlines()
        if line
    )
    return records, receipt


def write_staged(
    root: Path,
    receipt: SourceReceipt,
    records: tuple[SupervisedRecord, ...],
) -> SourceReceipt:
    """Atomically publish internal rows and their checksum receipt."""
    root.mkdir(parents=True, exist_ok=True)
    records_path, receipt_path = stage_paths(root, receipt.source.value)
    records_tmp = records_path.with_suffix(".jsonl.tmp")
    receipt_tmp = receipt_path.with_suffix(".json.tmp")
    try:
        payload = b"\n".join(row.model_dump_json().encode() for row in records) + b"\n"
        _ = records_tmp.write_bytes(payload)
        complete_receipt = receipt.model_copy(
            update={"staging_sha256": hashlib.sha256(payload).hexdigest()}
        )
        _ = receipt_tmp.write_text(
            complete_receipt.model_dump_json(indent=2), encoding="utf-8"
        )
        _ = records_tmp.replace(records_path)
        _ = receipt_tmp.replace(receipt_path)
    finally:
        records_tmp.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
    return complete_receipt
