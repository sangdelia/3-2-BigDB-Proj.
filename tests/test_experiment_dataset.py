"""Regression tests for lossless labels and pre-dedup grouping."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from bigdb_mvp.data.models import SourceName
from bigdb_mvp.experiments.dataset import deduplicate, group_rows, validate_splits
from bigdb_mvp.experiments.dataset_sources import parse_row, verify_asset
from bigdb_mvp.experiments.dataset_types import Annotation, DatasetError, RawRow


def test_multilabel_annotations_survive_dedup() -> None:
    raw = RawRow(
        {
            "text": "합성 문장",
            "label": "L2_hate",
            "target_label": ["a", "b"],
            "target_rationale": [[1, 2]],
        }
    )
    first = parse_row(SourceName.KHATERS, raw, 0)
    second = parse_row(
        SourceName.KMHAS, RawRow({"document": "합성  문장", "label": "0,2"}), 1
    )
    rows, counts = deduplicate(group_rows((first, second)))
    assert len(rows) == 1
    assert set(rows[0].target_labels) == {
        "khaters:a",
        "khaters:b",
        "kmhas:0",
        "kmhas:2",
    }
    annotations = TypeAdapter(list[Annotation]).validate_json(
        rows[0].source_labels_json
    )
    assert annotations[0].raw == raw
    assert len(annotations) == 2
    assert counts["consistent_duplicate_rows"] == 1


def test_conflicting_sets_are_entirely_excluded() -> None:
    first = parse_row(SourceName.KMHAS, RawRow({"document": "합성", "label": "8"}), 0)
    second = parse_row(SourceName.KMHAS, RawRow({"document": "합성", "label": "0"}), 1)
    rows, counts = deduplicate(group_rows((first, second)))
    assert rows == ()
    assert counts["conflicting_binary_rows"] == 2
    assert counts["conflicting_binary_sets"] == 1


def test_transitive_groups_survive_conflict_removal() -> None:
    rows = tuple(
        parse_row(
            SourceName.NIKL,
            RawRow({"target_form": text, "label": label, "doc_id": doc, "context": ""}),
            index,
        )
        for index, (text, doc, label) in enumerate(
            (
                ("문장 하나", "a", "appropriate"),
                ("공통 문장", "a", "appropriate"),
                ("공통  문장", "b", "inappropriate"),
                ("문장 둘", "b", "appropriate"),
            )
        )
    )
    grouped = group_rows(rows)
    assert len({row.group_id for row in grouped}) == 1
    assert len({row.split for row in grouped}) == 1
    kept, _ = deduplicate(grouped)
    assert len(kept) == 2
    validate_splits(kept)
    assert group_rows(tuple(reversed(rows)))[0].group_id == grouped[0].group_id


def test_shared_article_title_groups_sources() -> None:
    a = parse_row(
        SourceName.BEEP,
        RawRow(
            {
                "comments": "첫 문장",
                "hate": "none",
                "bias": "none",
                "title": "합성 기사 제목 입니다",
            }
        ),
        0,
    )
    b = parse_row(
        SourceName.KOLD,
        RawRow(
            {
                "comment": "다른 문장",
                "OFF": False,
                "TGT": "",
                "GRP": "",
                "title": "합성 기사 제목 입니다",
            }
        ),
        0,
    )
    grouped = group_rows((a, b))
    assert grouped[0].group_id == grouped[1].group_id
    assert grouped[0].context_text == a.context_text


def test_checksum_rejection(tmp_path: Path) -> None:
    asset = tmp_path / "source.json"
    _ = asset.write_text("changed", encoding="utf-8")
    with pytest.raises(DatasetError, match="checksum_mismatch"):
        verify_asset(asset, "0" * 64)


def test_kold_absent_targets_retain_clean_row() -> None:
    raw = RawRow(
        {
            "comment": "합성 중립 문장",
            "OFF": False,
            "TGT": None,
            "GRP": None,
            "title": "",
        }
    )
    row = parse_row(SourceName.KOLD, raw, 0)
    assert row.label == "clean"
    assert row.target_labels == []
    assert row.context_kind == "unavailable"
    annotations = TypeAdapter(list[Annotation]).validate_json(row.source_labels_json)
    assert annotations[0].raw.root["TGT"] is None


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_kold_rejects_non_boolean_off(value: str | int | None) -> None:
    raw = RawRow({"comment": "합성 문장", "OFF": value})
    with pytest.raises(DatasetError, match="kold_invalid_off"):
        _ = parse_row(SourceName.KOLD, raw, 0)


def test_kold_rejects_missing_off() -> None:
    with pytest.raises(DatasetError, match="kold_invalid_off"):
        _ = parse_row(SourceName.KOLD, RawRow({"comment": "합성 문장"}), 0)


def test_missing_asset_has_typed_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"asset_missing:absent\.json"):
        verify_asset(tmp_path / "absent.json", "0" * 64)
