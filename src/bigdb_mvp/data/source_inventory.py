"""Approved but currently blocked source-split inventory."""

from __future__ import annotations

from typing import Final

from bigdb_mvp.data.manifest import BlockedSourceInventory
from bigdb_mvp.data.models import CanonicalRole, SourceName

BLOCKED_INVENTORY: Final = (
    ("nikl", "test", "test_query", "nikl_test.json", "hidden_labels_unverified"),
    (
        "unsmile",
        "valid",
        "test_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "kmhas",
        "validation",
        "test_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "kmhas",
        "test",
        "test_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "khaters",
        "validation",
        "test_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "khaters",
        "test",
        "test_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "beep",
        "dev",
        "calibration_query",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
    (
        "beep",
        "test",
        "unlabeled_inventory",
        "official_revision_unverified",
        "asset_pin_unavailable",
    ),
)


def blocked_inventory_models() -> tuple[BlockedSourceInventory, ...]:
    """Return typed safe entries for every unpinned approved split."""
    return tuple(
        BlockedSourceInventory(
            source=SourceName(source),
            source_split=split,
            role=CanonicalRole(role),
            asset=asset,
            reason_code=reason,
        )
        for source, split, role, asset, reason in BLOCKED_INVENTORY
    )
