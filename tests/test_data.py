from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bigdb_mvp.data.manifest import Manifest, SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    Language,
    RecordDraft,
    SourceName,
    SplitName,
)
from bigdb_mvp.data.normalize import (
    RowExclusionError,
    build_record,
    group_id_for,
    khaters_label,
    require_train_split,
)
from bigdb_mvp.data.split import (
    MinimumQuotaError,
    assign_group_splits,
    deduplicate_exact,
    validate_minimum_quotas,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Normal ", BinaryLabel.CLEAN),
        ("OFFENSIVE", BinaryLabel.HARMFUL),
        (" L1_HATE", BinaryLabel.HARMFUL),
        ("l2_hate ", BinaryLabel.HARMFUL),
    ],
)
def test_khaters_remaps_only_normalized_original_label(
    raw: str, expected: BinaryLabel
) -> None:
    # Given: an original K-HATERS label with arbitrary casing and whitespace.
    # When: the original label is remapped.
    actual = khaters_label(raw)
    # Then: only the documented four-value mapping is used.
    assert actual is expected


@pytest.mark.parametrize("raw", [None, "hate", "", "unknown"])
def test_khaters_rejects_missing_or_unknown_original_label(
    raw: str | None,
) -> None:
    # Given: missing or undocumented K-HATERS metadata.original_label.
    # When / Then: no cached binary label is used as a fallback.
    with pytest.raises(RowExclusionError, match="khaters_original_label"):
        _ = khaters_label(raw)


@pytest.mark.parametrize("split", ["dev", "validation", "test", ""])
def test_non_train_source_splits_are_rejected(split: str) -> None:
    # Given: a record that is not explicitly from a train split.
    # When / Then: role leakage is rejected at the ingest boundary.
    with pytest.raises(RowExclusionError, match="non_train_split"):
        require_train_split(split)


def test_group_id_prefers_article_or_conversation_identifier() -> None:
    # Given: two texts attached to the same source article identifier.
    first = group_id_for(SourceName.NIKL, "first", preferred_id="article-7")
    second = group_id_for(SourceName.NIKL, "second", preferred_id="article-7")

    # When / Then: the preferred source identifier groups both records together.
    assert first == second


def test_group_id_falls_back_to_normalized_text_hash() -> None:
    # Given: semantically identical spacing/case/Unicode variants without a source ID.
    composed = group_id_for(SourceName.CIVIL_COMMENTS, "  Café\tTEST ", None)
    decomposed = group_id_for(
        SourceName.CIVIL_COMMENTS,
        "cafe\N{COMBINING ACUTE ACCENT} test",
        None,
    )

    # When / Then: NFKC, lowercase, and whitespace normalization produce one group.
    assert composed == decomposed


def test_exact_duplicates_are_removed_before_splitting() -> None:
    # Given: the same exact text appears in two source records.
    first = build_record(
        RecordDraft(
            text="same text",
            label=BinaryLabel.CLEAN,
            language=Language.EN,
            source=SourceName.CIVIL_COMMENTS,
            source_split="train",
        )
    )
    duplicate = build_record(
        RecordDraft(
            text="same text",
            label=BinaryLabel.CLEAN,
            language=Language.EN,
            source=SourceName.CIVIL_COMMENTS,
            source_split="train",
        )
    )

    # When: exact duplicates are removed.
    unique, removed = deduplicate_exact((first, duplicate))

    # Then: only one deterministic representative remains.
    assert unique == (first,)
    assert removed == 1


def test_seeded_group_split_is_deterministic_and_disjoint() -> None:
    # Given: records from independently identifiable source groups.
    records = tuple(
        build_record(
            RecordDraft(
                text=f"synthetic text {index}",
                label=(BinaryLabel.CLEAN if index % 2 == 0 else BinaryLabel.HARMFUL),
                language=Language.EN,
                source=SourceName.CIVIL_COMMENTS,
                source_split="train",
                preferred_group_id=f"group-{index}",
            )
        )
        for index in range(200)
    )

    # When: the same seed is applied twice.
    first = assign_group_splits(records, seed=42)
    second = assign_group_splits(records, seed=42)

    # Then: assignments are byte-order deterministic and group-disjoint.
    assert first == second
    split_groups = {
        split: {row.group_id for row in first if row.split is split}
        for split in SplitName
    }
    assert split_groups[SplitName.TRAIN].isdisjoint(split_groups[SplitName.CALIBRATION])
    assert split_groups[SplitName.TRAIN].isdisjoint(split_groups[SplitName.TEST])
    assert split_groups[SplitName.CALIBRATION].isdisjoint(split_groups[SplitName.TEST])


def test_minimum_quota_failure_is_explicit() -> None:
    # Given: a dataset far below the per-language/label split minimums.
    row = build_record(
        RecordDraft(
            text="only row",
            label=BinaryLabel.CLEAN,
            language=Language.KO,
            source=SourceName.NIKL,
            source_split="train",
        )
    ).with_split(SplitName.TRAIN)

    # When / Then: publication is blocked with a typed quota error.
    with pytest.raises(MinimumQuotaError, match="minimum_quota"):
        validate_minimum_quotas((row,))


def test_runtime_manifest_never_contains_thesis_paths(tmp_path: Path) -> None:
    # Given: a safe runtime manifest produced from internal data only.
    manifest = Manifest.empty(seed=42).with_source(
        SourceReceipt(
            source=SourceName.NIKL,
            revision="local-sha256",
            license="restricted_local_review_required",
            input_sha256="a" * 64,
            accepted=1,
            excluded=0,
        )
    )
    target = tmp_path / "manifest.json"

    # When: the manifest is persisted.
    _ = target.write_text(manifest.model_dump_json(), encoding="utf-8")

    # Then: neither thesis repository name nor an absolute source path is retained.
    contents = target.read_text(encoding="utf-8")
    assert "hate-speech-grad-thesis" not in contents
    assert "/home/kyu/Coulmn" not in contents
