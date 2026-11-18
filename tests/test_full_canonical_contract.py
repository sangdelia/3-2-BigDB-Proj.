from __future__ import annotations

import pytest
from pydantic import ValidationError

from bigdb_mvp.data.manifest import (
    CANONICAL_ARTIFACT_LAYOUT,
    CanonicalManifest,
    SourceReceipt,
)
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRecord,
    CanonicalRole,
    Language,
    SourceName,
)


def test_canonical_roles_and_sort_key_are_explicit() -> None:
    # Given: a synthetic labeled row in the research index role.
    row = CanonicalRecord(
        text="synthetic canonical row",
        label=BinaryLabel.CLEAN,
        language=Language.EN,
        source=SourceName.CIVIL_COMMENTS,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        group_id="group-1",
        text_sha256="a" * 64,
    )

    # When: the stable key is requested.
    actual = row.stable_sort_key()

    # Then: the key is digest-only and has a documented field order.
    assert actual == (
        "research_index",
        "en",
        "clean",
        "civil_comments",
        "train",
        "group-1",
        "a" * 64,
    )


def test_unlabeled_inventory_rejects_a_supervised_label() -> None:
    # Given: an inventory row incorrectly carrying a supervised label.
    # When / Then: the role contract rejects the mixed representation.
    with pytest.raises(ValidationError, match="unlabeled_inventory_label"):
        _ = CanonicalRecord(
            text="synthetic inventory row",
            label=BinaryLabel.CLEAN,
            language=Language.EN,
            source=SourceName.CIVIL_COMMENTS,
            source_split="inventory",
            role=CanonicalRole.UNLABELED_INVENTORY,
            group_id="group-1",
            text_sha256="b" * 64,
        )


def test_supervised_role_requires_a_label() -> None:
    # Given: a query row without a label.
    # When / Then: supervised canonical roles fail closed.
    with pytest.raises(ValidationError, match="supervised_role_label"):
        _ = CanonicalRecord(
            text="synthetic query row",
            label=None,
            language=Language.EN,
            source=SourceName.CIVIL_COMMENTS,
            source_split="test",
            role=CanonicalRole.TEST_QUERY,
            group_id="group-1",
            text_sha256="c" * 64,
        )


def test_canonical_manifest_requires_digests_and_uses_full_layout() -> None:
    # Given: a complete synthetic canonical manifest.
    manifest = CanonicalManifest(
        seed=42,
        dataset_sha256="d" * 64,
        logical_digest="e" * 64,
        record_count=1,
        role_counts=(),
        source_receipts=(),
    )

    # When: its safe JSON representation and layout are inspected.
    payload = manifest.model_dump_json()

    # Then: paths and metadata contain no raw canonical text.
    assert manifest.dataset_file == CANONICAL_ARTIFACT_LAYOUT.local_dataset
    assert manifest.manifest_file == CANONICAL_ARTIFACT_LAYOUT.local_manifest
    assert "synthetic canonical row" not in payload


@pytest.mark.parametrize("field", ["dataset_sha256", "logical_digest"])
def test_canonical_manifest_rejects_missing_digest(field: str) -> None:
    # Given: a manifest with one required digest omitted.
    values = {
        "seed": 42,
        "dataset_sha256": "f" * 64,
        "logical_digest": "f" * 64,
        "record_count": 0,
        "role_counts": (),
        "source_receipts": (),
    }
    values[field] = ""

    # When / Then: publication metadata fails closed at the typed boundary.
    with pytest.raises(ValidationError, match=field):
        _ = CanonicalManifest.model_validate(values)


@pytest.mark.parametrize("value", ["", "not-a-sha", "A" * 64])
def test_source_receipt_rejects_malformed_trusted_digest(value: str) -> None:
    # Given: a receipt crossing the trusted provenance boundary.
    values = {
        "source": SourceName.CIVIL_COMMENTS,
        "revision": "fixture",
        "license": "test",
        "input_sha256": value,
        "asset_sha256": "a" * 64,
        "staging_sha256": "b" * 64,
        "schema_sha256": "c" * 64,
        "raw_bytes": 1,
        "accepted": 0,
        "excluded": 0,
    }

    # When / Then: malformed or empty digests fail closed.
    with pytest.raises(ValidationError):
        _ = SourceReceipt.model_validate(values)
