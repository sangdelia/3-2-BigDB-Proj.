"""Deterministic group splitting and quota enforcement."""

from __future__ import annotations

import hashlib
from typing import Final

from bigdb_mvp.data.models import BinaryLabel, Language, SplitName, SupervisedRecord

TRAIN_CUTOFF: Final = 80
CALIBRATION_CUTOFF: Final = 90


class MinimumQuotaError(Exception):
    """Report a missing language-label-split quota."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize a quota failure safe for CLI propagation."""
        self.reason_code = reason_code
        super().__init__(reason_code)


def deduplicate_exact(
    records: tuple[SupervisedRecord, ...],
) -> tuple[tuple[SupervisedRecord, ...], int]:
    """Keep the first representative for each exact text digest."""
    unique: dict[str, SupervisedRecord] = {}
    for record in records:
        if record.text_sha256 not in unique:
            unique[record.text_sha256] = record
    ordered = tuple(unique.values())
    return ordered, len(records) - len(ordered)


def assign_group_splits(
    records: tuple[SupervisedRecord, ...], *, seed: int
) -> tuple[SupervisedRecord, ...]:
    """Assign whole groups to deterministic 80/10/10 hash buckets."""
    assignments: dict[str, SplitName] = {}
    assigned: list[SupervisedRecord] = []
    for record in records:
        split = assignments.get(record.group_id)
        if split is None:
            bucket = (
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{record.group_id}".encode()).digest()[:8],
                    byteorder="big",
                )
                % 100
            )
            if bucket < TRAIN_CUTOFF:
                split = SplitName.TRAIN
            elif bucket < CALIBRATION_CUTOFF:
                split = SplitName.CALIBRATION
            else:
                split = SplitName.TEST
            assignments[record.group_id] = split
        assigned.append(record.with_split(split))
    return tuple(assigned)


def validate_minimum_quotas(records: tuple[SupervisedRecord, ...]) -> None:
    """Block publication when any language-label-split cell is undersized."""
    required = {
        SplitName.TRAIN: 400,
        SplitName.CALIBRATION: 50,
        SplitName.TEST: 50,
    }
    for language in Language:
        for label in BinaryLabel:
            for split, minimum in required.items():
                count = sum(
                    row.language is language
                    and row.label is label
                    and row.split is split
                    for row in records
                )
                if count < minimum:
                    code = (
                        f"minimum_quota:{language.value}:{label.value}:"
                        f"{split.value}:{count}<{minimum}"
                    )
                    raise MinimumQuotaError(reason_code=code)
