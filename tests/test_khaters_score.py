from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import pytest
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.experiments.khaters_ratings import (
    REVISION,
    Ratings,
    load_dataset,
    text_key,
)
from bigdb_mvp.khaters_score import Head, Probabilities, ScoreModel, predict
from bigdb_mvp.khaters_score_artifact import load_score_model
from bigdb_mvp.korean_model import KoreanPrediction, load_korean_model
from bigdb_mvp.korean_web import create_korean_app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ordinary() -> Ratings:
    return Ratings(
        text="감사합니다",
        gender=0,
        age=0,
        disability=0,
        job=0,
        race=0,
        religion=0,
        politics=0,
        individuals=0,
        others=0,
        insult=0,
        swear=0,
        threat=0,
        obscenity=0,
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((1.0, 0.0, 0.0), 0),
        ((0.0, 1.0, 0.0), 50),
        ((0.0, 0.0, 1.0), 100),
        ((0.2, 0.3, 0.5), 65),
    ],
)
def test_score_is_expected_grade(
    values: tuple[float, float, float], expected: int
) -> None:
    assert Probabilities(zero=values[0], one=values[1], two=values[2]).score == expected


def test_individual_insult_and_swear_are_not_group_hate(ordinary: Ratings) -> None:
    row = ordinary.model_copy(update={"individuals": 1, "insult": 2, "swear": 2})
    assert (row.grade("hate"), row.grade("abuse"), row.grade("swear")) == (0, 2, 2)


def test_protected_group_rating_is_hate_without_swear(ordinary: Ratings) -> None:
    row = ordinary.model_copy(update={"religion": 1, "gender": 2})
    assert (row.grade("hate"), row.grade("abuse"), row.grade("swear")) == (2, 2, 0)


def test_missing_and_invalid_ratings_are_not_zero_filled(ordinary: Ratings) -> None:
    data = ordinary.model_dump()
    del data["swear"]
    with pytest.raises(ValidationError):
        _ = Ratings.model_validate(data)
    data["swear"] = 3
    with pytest.raises(ValidationError):
        _ = Ratings.model_validate(data)


def test_dedup_excludes_cross_split_text_and_rating_conflicts(
    tmp_path: Path,
    ordinary: Ratings,
) -> None:
    crossing = ordinary.model_copy(update={"text": "같은 문장"})
    conflict = ordinary.model_copy(update={"text": "충돌 문장"})
    changed = conflict.model_copy(update={"swear": 2})
    val = ordinary.model_copy(update={"text": "검증 문장"})
    test = ordinary.model_copy(update={"text": "평가 문장"})
    originals = (
        (ordinary, ordinary, crossing, conflict, changed),
        (crossing, val),
        (test, ordinary.model_copy(update={"text": "  "})),
    )
    for name, rows in zip(("train", "val", "test"), originals, strict=True):
        _ = (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(row.model_dump_json() for row in rows), encoding="utf-8"
        )
    dataset = load_dataset(tmp_path)
    assert dataset.original_counts == (5, 2, 2)
    assert dataset.excluded_rows == (4, 1, 1)
    assert dataset.empty_rows == (0, 0, 1)
    assert dataset.cross_split_keys == dataset.conflicting_keys == 1
    assert tuple(map(len, dataset.splits)) == (1, 1, 1)
    assert text_key("같은\n 문장") == text_key("같은 문장")


@pytest.fixture
def tiny_model() -> ScoreModel:
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), sublinear_tf=True, max_features=100
    )
    matrix = vectorizer.fit_transform(("감사합니다", "나쁜 사람", "심한 표현"))
    classifier = LogisticRegression(
        class_weight=None, solver="lbfgs", max_iter=1000, random_state=42
    )
    _ = classifier.fit(matrix, (0, 1, 2))
    heads = (
        Head("hate", classifier, 1),
        Head("abuse", classifier, 1),
        Head("swear", classifier, 1),
    )
    return ScoreModel(vectorizer, heads, "fixture", REVISION)


def test_unrecognized_input_has_no_scores(tiny_model: ScoreModel) -> None:
    result = predict(tiny_model, "ㅅㅂ")
    assert result.recognized_features == 0
    assert (result.hate_score, result.abuse_score, result.swear_score) == (
        None,
        None,
        None,
    )
    assert result.grades == {}


@pytest.mark.parametrize("version", ["khaters_score_v1", "khaters_score_v2"])
def test_artifact_roundtrip_and_tampering(
    tmp_path: Path,
    tiny_model: ScoreModel,
    version: str,
) -> None:
    path = tmp_path / "artifacts" / version / "model.joblib"
    report = tmp_path / "reports" / version / "metrics.json"
    path.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    _ = joblib.dump(tiny_model, path)
    _ = report.write_text(
        json.dumps(
            {
                "model_sha256": sha256_file(path),
                "dataset_digest": "fixture",
                "revision": REVISION,
            }
        ),
        encoding="utf-8",
    )
    if version.endswith("v2"):
        _ = (tmp_path / "reports/khaters_score_selection.json").write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
    loaded = load_score_model(tmp_path)
    assert predict(loaded, "감사합니다") == predict(tiny_model, "감사합니다")
    _ = path.write_bytes(b"changed artifact")
    with pytest.raises(ExperimentError, match="score_model_checksum_mismatch"):
        _ = load_score_model(tmp_path)


def test_rating_api_matches_frozen_head(tiny_model: ScoreModel) -> None:
    client = create_korean_app(
        load_korean_model(), score_model=tiny_model
    ).test_client()
    response = client.post("/api/korean/predict", json={"text": "감사합니다"})
    assert response.status_code == 200
    result = KoreanPrediction.model_validate_json(response.text)
    assert result.rating_scores == predict(tiny_model, "감사합니다")
    assert response.headers["Cache-Control"] == "no-store"
