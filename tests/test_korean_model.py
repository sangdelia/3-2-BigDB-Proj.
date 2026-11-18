"""Real fitted-model inference, offsets, evidence and artifact integrity."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import polars as pl
import pytest
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification import run
from bigdb_mvp.experiments.classification_contracts import ExperimentError, Row
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.korean_model import (
    KoreanModelService,
    KoreanRequest,
    load_korean_model,
)

if TYPE_CHECKING:
    from pathlib import Path


def _dataset(root: Path) -> None:
    messages: tuple[tuple[str, Literal["clean", "harmful"]], ...] = (
        ("공격적인 바보 멍청이", "harmful"),
        ("감사합니다 좋은 하루", "clean"),
    )
    rows = tuple(
        Row(
            text=(text := f"{message} {split} {i}"),
            label=label,
            language="ko",
            source="fixture",
            group_id=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            split=split,
            context_text="기사 제목",
            context_kind="title",
        )
        for split in ("train", "validation", "test")
        for message, label in messages
        for i in range(3)
    )
    root.mkdir()
    pl.DataFrame([row.model_dump() for row in rows]).write_parquet(
        root / "corpus.parquet"
    )
    _ = (root / "manifest.json").write_text(
        json.dumps({"dataset_sha256": sha256_file(root / "corpus.parquet")})
    )


@pytest.fixture
def service() -> KoreanModelService:
    texts = (
        "너는 정말 멍청한 바보야",
        "멍청한 바보 바보야",
        "정말 멍청하고 바보 같아",
        "고맙습니다 좋은 하루 보내세요",
        "정말 감사합니다 행복하세요",
        "좋은 하루 행복한 하루",
    )
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, solver="liblinear"
    )
    _ = classifier.fit(matrix, (1, 1, 1, 0, 0, 0))
    return KoreanModelService(
        Model(vectorizer, classifier, None, 0.44, context=False), "a" * 64, "b" * 64
    )


@pytest.mark.parametrize("text", ["", "  \n", "only english", "가" * 1001])
def test_request_rejects_invalid_text(text: str) -> None:
    with pytest.raises(ValidationError):
        _ = KoreanRequest(text=text)


@pytest.mark.parametrize(
    "text",
    [
        "너는 정말 멍청한 바보야",
        "🙂 너는\t\n정말  멍청한 바보야",
        "멍청한 바보야 멍청한 바보야",
    ],
)
def test_real_scores_offsets_and_removal_evidence(
    service: KoreanModelService, text: str
) -> None:
    result = service.predict(KoreanRequest(text=text))
    direct = (
        float(
            service.model.classifier.predict_proba(
                service.model.vectorizer.transform((text,))
            )[:, 1][0]
        )
        * 100
    )
    assert result.harmful_score == pytest.approx(direct)
    assert result.label == ("harmful" if direct >= 44 else "clean")
    assert result.threshold_score == 44
    assert result.candidates
    for candidate in result.candidates:
        assert text[candidate.start : candidate.end] == candidate.text
        removed = text[: candidate.start] + text[candidate.end :]
        without = (
            float(
                service.model.classifier.predict_proba(
                    service.model.vectorizer.transform((removed,))
                )[:, 1][0]
            )
            * 100
        )
        retained = int(service.model.vectorizer.transform((removed,)).getnnz(axis=1)[0])
        assert candidate.recognized_features_without == retained
        if retained:
            assert candidate.score_without == pytest.approx(without)
            assert candidate.score_drop_points == pytest.approx(direct - without)
        else:
            assert candidate.score_without is None
            assert candidate.score_drop_points is None
        assert candidate.highlight_as_harmful_candidate == (
            result.label == "harmful" and retained > 0 and direct > without
        )
        assert candidate.positive_contribution == pytest.approx(
            sum(item.occurrence_contribution for item in candidate.evidence)
        )
    positions = sorted((item.start, item.end) for item in result.candidates)
    assert all(left[1] <= right[0] for left, right in pairwise(positions))


def test_repeated_features_do_not_multiply_contribution(
    service: KoreanModelService,
) -> None:
    text = "멍청한 바보야 멍청한 바보야"
    result = service.predict(KoreanRequest(text=text))
    totals: dict[str, float] = {}
    contributions: dict[str, float] = {}
    for candidate in result.candidates:
        for evidence in candidate.evidence:
            totals[evidence.ngram] = (
                totals.get(evidence.ngram, 0) + evidence.occurrence_contribution
            )
            contributions[evidence.ngram] = evidence.contribution
    assert totals
    for ngram, total in totals.items():
        assert total == pytest.approx(contributions[ngram])


def test_unknown_features_report_insufficient(service: KoreanModelService) -> None:
    result = service.predict(KoreanRequest(text="힣힣힣"))
    assert result.candidates == ()
    assert result.explanation_status == "insufficient"
    assert result.label == "review"
    assert result.recognized_feature_count == 0
    assert result.review_reason == "no_recognized_features"


def test_request_and_prediction_keep_original_text(service: KoreanModelService) -> None:
    text = "🙂 İ   멍청한 바보야"
    request = KoreanRequest(text=text)
    assert request.text == text
    result = service.predict(request)
    assert result.candidates
    assert len(result.candidates) == 1
    assert result.candidates[0].start == 3
    assert result.candidates[0].end == len(text)
    assert result.candidates[0].text == "   멍청한 바보야"


def test_clean_sentence_never_marks_a_harmful_span(service: KoreanModelService) -> None:
    result = service.predict(KoreanRequest(text="고맙습니다 좋은 하루 보내세요"))
    assert result.label == "clean"
    assert all(not item.highlight_as_harmful_candidate for item in result.candidates)


def test_load_real_saved_model_then_reject_digest_drift(tmp_path: Path) -> None:
    dataset, artifacts, reports = (
        tmp_path / name for name in ("dataset", "artifacts", "reports")
    )
    _dataset(dataset)
    _ = run(dataset, artifacts, reports, max_features=200)
    service = load_korean_model(artifacts, reports / "metrics.json", dataset)
    before = {
        path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()
    }
    result = service.predict(KoreanRequest(text="공격적인 바보 멍청이"))
    assert result.label == "harmful"
    assert result.model_sha256 == service.model_sha256
    after = {
        path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()
    }
    assert before == after
    report_path = reports / "metrics.json"
    original_report = report_path.read_text()
    changed = original_report.replace('"threshold": 0.44', '"threshold": 0.99')
    if changed == original_report:
        threshold = service.model.threshold
        changed = original_report.replace(
            f'"threshold": {threshold}', '"threshold": 0.99'
        )
    assert changed != original_report
    _ = report_path.write_text(changed)
    with pytest.raises(ExperimentError, match="model_configuration_mismatch"):
        _ = load_korean_model(artifacts, report_path, dataset)
    _ = report_path.write_text(original_report)
    with (artifacts / "char_tfidf.joblib").open("ab") as handle:
        _ = handle.write(b"drift")
    with pytest.raises(ExperimentError, match="model_digest_mismatch"):
        _ = load_korean_model(artifacts, reports / "metrics.json", dataset)


def test_missing_artifact_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_korean_model(tmp_path, tmp_path / "metrics.json", tmp_path)


def test_dataset_mismatch_rejected_before_model_load(tmp_path: Path) -> None:
    dataset, artifacts, reports = (
        tmp_path / name for name in ("dataset", "artifacts", "reports")
    )
    _dataset(dataset)
    _ = run(dataset, artifacts, reports, max_features=200)
    path = dataset / "manifest.json"
    _ = path.write_text(json.dumps({"dataset_sha256": "f" * 64}))
    with pytest.raises(ExperimentError, match="dataset_digest_mismatch"):
        _ = load_korean_model(artifacts, reports / "metrics.json", dataset)
