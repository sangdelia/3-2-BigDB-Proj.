from __future__ import annotations

import math

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.korean_model import KoreanModelService, KoreanRequest


@pytest.fixture
def predictor() -> KoreanModelService:
    texts = (
        "너는 바보야",
        "정말 바보야",
        "바보가 아니야",
        "멍청하지 아니야",
        "감사합니다",
    )
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, solver="liblinear"
    )
    _ = classifier.fit(matrix, (1, 1, 0, 0, 0))
    return KoreanModelService(
        Model(vectorizer, classifier, None, 0.43, context=False), "a" * 64, "b" * 64
    )


def test_signed_evidence_reconstructs_score(predictor: KoreanModelService) -> None:
    request = KoreanRequest(text="너는 바보가 아니야")
    result = predictor.predict(request)
    evidence = result.decision_evidence
    assert evidence is not None
    reconstructed = 100 / (
        1
        + math.exp(
            -(evidence.intercept + evidence.positive_sum + evidence.negative_sum)
        )
    )
    assert reconstructed == pytest.approx(result.harmful_score)
    assert evidence.margin_points == pytest.approx(result.harmful_score - 43)
    assert evidence.negative_sum < 0 < evidence.positive_sum


def test_negative_features_are_exposed(predictor: KoreanModelService) -> None:
    request = KoreanRequest(text="너는 바보가 아니야")
    result = predictor.predict(request)
    evidence = result.decision_evidence
    assert evidence is not None
    assert any("아니" in item.ngram for item in evidence.negative_features)
    assert all(item.contribution < 0 for item in evidence.negative_features)
    assert all(item.contribution > 0 for item in evidence.positive_features)


def test_featureless_removal_has_no_comparison(predictor: KoreanModelService) -> None:
    request = KoreanRequest(text="바보")
    result = predictor.predict(request)
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.recognized_features_without == 0
        assert candidate.score_without is None
        assert candidate.score_drop_points is None
        assert not candidate.highlight_as_harmful_candidate


def test_unknown_input_has_no_equation(predictor: KoreanModelService) -> None:
    request = KoreanRequest(text="힣힣힣")
    result = predictor.predict(request)
    assert result.label == "review"
    assert result.decision_evidence is None
