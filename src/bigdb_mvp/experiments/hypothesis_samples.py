"""Deterministic source/label sampling preserves existing corpus partitions."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bigdb_mvp.experiments.classification_contracts import Row


def sample(rows: tuple[Row, ...], split: str, cap: int) -> tuple[Row, ...]:
    """Rank within each source/label by seeded text digest, never model scores."""
    selected: list[Row] = []
    for source in sorted({row.source for row in rows}):
        for label in ("clean", "harmful"):
            candidates = (
                row
                for row in rows
                if row.split == split and row.source == source and row.label == label
            )
            selected.extend(
                sorted(
                    candidates,
                    key=lambda row: hashlib.sha256(
                        f"20261014|{row.text_sha256}".encode()
                    ).hexdigest(),
                )[:cap]
            )
    return tuple(selected)
