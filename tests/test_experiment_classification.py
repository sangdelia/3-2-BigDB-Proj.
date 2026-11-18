"""Real experiment boundaries and held-out-safe feature construction."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest
from pydantic import ValidationError

from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification import run
from bigdb_mvp.experiments.classification_contracts import (
    ExperimentError,
    Row,
    feature_text,
    load_rows,
)
from bigdb_mvp.experiments.classification_metrics import (
    metrics,
    threshold_on_validation,
)
from bigdb_mvp.experiments.lexicon import learn_lexicon

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    split: Literal["train", "validation", "test"],
    label: Literal["harmful", "clean"],
    index: int = 0,
) -> Row:
    word = "공격적인 바보 멍청이" if label == "harmful" else "감사합니다 좋은 하루 행복"
    text = f"{word} {split} {index}"
    return Row(
        text=text,
        label=label,
        language="ko",
        source="fixture",
        group_id=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        split=split,
        context_text="실제 기사 제목",
        context_kind="title",
    )


def _rows() -> tuple[Row, ...]:
    return tuple(
        _row(split, label, index)
        for split in ("train", "validation", "test")
        for label in ("clean", "harmful")
        for index in range(6)
    )


def _write(root: Path, rows: tuple[Row, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([row.model_dump() for row in rows]).write_parquet(
        root / "corpus.parquet"
    )
    _ = (root / "manifest.json").write_text(
        json.dumps({"dataset_sha256": sha256_file(root / "corpus.parquet")}),
        encoding="utf-8",
    )


def test_gold_metadata_cannot_enter_features() -> None:
    raw = _row("train", "clean").model_dump()
    raw.update(
        {
            "target_labels": ["LEAK_TARGET"],
            "expression_labels": ["LEAK_EXPRESSION"],
            "source_labels_json": "LEAK_LABEL",
        }
    )
    row = Row.model_validate(raw)
    assert feature_text(row) == row.text
    assert (
        feature_text(row, context=True) == f"제목: {row.context_text}\n댓글: {row.text}"
    )
    assert "LEAK" not in feature_text(row, context=True)


def test_dictionary_excludes_test_and_validation_words() -> None:
    train = _rows()
    before = learn_lexicon(train, min_count=1)
    injected = _row("test", "harmful").model_copy(
        update={"text": "heldoutsecret heldoutsecret"}
    )
    assert learn_lexicon((*train, injected), min_count=1) == before
    assert "heldoutsecret" not in (*before.positive, *before.negative)
    assert "멍청이" in before.positive


@pytest.mark.parametrize("field", ["group_id", "text_sha256"])
def test_reject_cross_split_leakage(tmp_path: Path, field: str) -> None:
    rows = list(_rows())
    replacement = {field: getattr(rows[0], field)}
    if field == "text_sha256":
        replacement["text"] = rows[0].text
    rows[-1] = rows[-1].model_copy(update=replacement)
    _write(tmp_path, tuple(rows))
    with pytest.raises(ExperimentError, match="cross_split_leakage"):
        _ = load_rows(tmp_path)


def test_reject_digest_drift(tmp_path: Path) -> None:
    _write(tmp_path, _rows())
    with (tmp_path / "corpus.parquet").open("ab") as handle:
        _ = handle.write(b"drift")
    with pytest.raises(ExperimentError, match="dataset_digest_mismatch"):
        _ = load_rows(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"), [("language", "en"), ("split", "calibration"), ("group_id", "")]
)
def test_reject_invalid_row(field: str, value: str) -> None:
    raw = _row("train", "clean").model_dump()
    raw[field] = value
    with pytest.raises(ValidationError):
        _ = Row.model_validate(raw)


def test_all_methods_real_fit_and_safe_reports(tmp_path: Path) -> None:
    dataset, artifacts, reports = (
        tmp_path / name for name in ("dataset", "artifacts", "reports")
    )
    rows = _rows()
    without_context = tuple(
        _row(split, label, 99).model_copy(update={"context_text": ""})
        for split in ("train", "validation", "test")
        for label in ("clean", "harmful")
    )
    _write(dataset, (*rows, *without_context))
    results = run(dataset, artifacts, reports, max_features=200)
    assert len(results) == 4
    assert results[0].train_count == 14
    assert results[2].train_count == results[3].train_count == 12
    assert results[2].test.count == results[3].test.count == 12
    assert all(
        result.test.tp + result.test.tn + result.test.fp + result.test.fn
        == result.test.count
        for result in results
    )
    assert len(list(artifacts.glob("*.joblib"))) == 4
    assert all(result.test_by_source["fixture"] == result.test for result in results)
    content = (reports / "metrics.json").read_text()
    assert "공격적인" not in content
    assert "train_lexicon" not in content
    assert (reports / "comparison.csv").stat().st_size > 0


def test_digest_uses_shared_whitespace_normalization(tmp_path: Path) -> None:
    rows = list(_rows())
    rows[0] = rows[0].model_copy(update={"text": "  " + rows[0].text + "\n"})
    _write(tmp_path, tuple(rows))
    loaded, _ = load_rows(tmp_path)
    assert loaded[0].text.startswith("  ")


def test_binary_metrics_positive_class_and_false_positive_rate() -> None:
    result = metrics((0, 0, 1, 1), (0.1, 0.9, 0.8, 0.2), 0.5)
    assert (result.tn, result.fp, result.fn, result.tp) == (1, 1, 1, 1)
    assert result.false_positive_rate == result.macro_f1 == 0.5
    assert result.harmful_precision == result.harmful_recall == 0.5


def test_validation_threshold_selects_known_macro_f1_optimum() -> None:
    threshold = threshold_on_validation((0, 0, 1, 1), (0.1, 0.2, 0.3, 0.4))
    assert 0.2 < threshold <= 0.3
    assert metrics((0, 0, 1, 1), (0.1, 0.2, 0.3, 0.4), threshold).macro_f1 == 1.0
