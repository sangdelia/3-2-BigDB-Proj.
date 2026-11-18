"""Predict expected original K-HATERS grades on a documented 0-100 scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.experiments.khaters_ratings import Rating, ScoreKind

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import FeatureMatrix, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression


class Probabilities(BaseModel):
    """Keep grade probabilities distinct from the expected ordinal score."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True)
    zero: float = Field(ge=0, le=1)
    one: float = Field(ge=0, le=1)
    two: float = Field(ge=0, le=1)

    @property
    def score(self) -> float:
        """Return 50 times the expected grade, not a hate classification chance."""
        return 50 * (self.one + 2 * self.two)


@dataclass(frozen=True, slots=True)
class Head:
    """Persist the classifier and validation-only probability temperature."""

    kind: ScoreKind
    classifier: LogisticRegression
    temperature: float


@dataclass(frozen=True, slots=True)
class ScoreModel:
    """Use train-only character features and separately aggregated rating heads."""

    vectorizer: TfidfVectorizer
    heads: tuple[Head, ...]
    dataset_digest: str
    revision: str


def probabilities(head: Head, matrix: FeatureMatrix) -> tuple[Probabilities, ...]:
    """Align estimator classes explicitly before applying temperature scaling."""
    result = head.classifier.predict_proba(matrix)
    columns: tuple[tuple[float, ...], ...] = tuple(
        tuple(float(p) ** (1 / head.temperature) for p in result[:, i])
        for i in range(len(head.classifier.classes_))
    )
    classes = tuple(int(label) for label in head.classifier.classes_)
    rows: list[Probabilities] = []
    for values in zip(*columns, strict=True):
        aligned: dict[int, float] = dict(zip(classes, values, strict=True))
        total = sum(values)
        rows.append(
            Probabilities(
                zero=aligned.get(0, 0.0) / total,
                one=aligned.get(1, 0.0) / total,
                two=aligned.get(2, 0.0) / total,
            )
        )
    return tuple(rows)


class Prediction(BaseModel):
    """Withhold all rating numbers when the feature vocabulary cannot see input."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    recognized_features: int
    hate_score: float | None
    abuse_score: float | None
    swear_score: float | None
    grades: dict[ScoreKind, Probabilities]
    dataset_digest: str
    revision: str


def predict(model: ScoreModel, text: str) -> Prediction:
    """Score a real input using frozen heads; absence of features means review."""
    matrix = model.vectorizer.transform((text,))
    recognized = int(matrix.getnnz(axis=1)[0])
    grades: dict[ScoreKind, Probabilities] = (
        {head.kind: probabilities(head, matrix)[0] for head in model.heads}
        if recognized
        else {}
    )
    return Prediction(
        recognized_features=recognized,
        hate_score=grades["hate"].score if recognized else None,
        abuse_score=grades["abuse"].score if recognized else None,
        swear_score=grades["swear"].score if recognized else None,
        grades=grades,
        dataset_digest=model.dataset_digest,
        revision=model.revision,
    )


def gold_probability(value: Probabilities, grade: Rating) -> float:
    """Select the independently known gold grade probability for NLL scoring."""
    return (value.zero, value.one, value.two)[grade]
