"""Signed binary-model evidence, separate from ordinal ratings and gold spans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from bigdb_mvp.experiments.classification_model import Model

DISPLAY_LIMIT: Final = 6


class Contribution(BaseModel):
    """A signed TF-IDF times coefficient value in logit units."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    ngram: str
    contribution: float


class DecisionEvidence(BaseModel):
    """Full sums reconstruct the score; displayed features are truncated lists."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    intercept: float
    positive_sum: float
    negative_sum: float
    logit: float
    margin_points: float
    positive_count: int
    negative_count: int
    positive_features: tuple[Contribution, ...]
    negative_features: tuple[Contribution, ...]


class FeatureEvidence(BaseModel):
    """A feature contribution is divided across its repeated occurrences."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    ngram: str
    contribution: float = Field(gt=0)
    occurrence_contribution: float = Field(gt=0)


class SpanCandidate(BaseModel):
    """Offsets use Python Unicode code points, with an exclusive end."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    positive_contribution: float = Field(gt=0)
    score_without: float | None = Field(ge=0, le=100)
    score_drop_points: float | None
    recognized_features_without: int = Field(ge=0)
    highlight_as_harmful_candidate: bool
    evidence: tuple[FeatureEvidence, ...]


def explain_decision(model: Model, text: str, score: float) -> DecisionEvidence:
    """Account for every nonzero contribution while showing six per direction."""
    values = model.vectorizer.transform((text,)).toarray()[0]
    features = tuple(
        Contribution(ngram=str(name), contribution=float(value * weight))
        for name, value, weight in zip(
            model.vectorizer.get_feature_names_out(),
            values,
            model.classifier.coef_[0],
            strict=True,
        )
        if value * weight != 0
    )
    positive = tuple(
        sorted(
            (item for item in features if item.contribution > 0),
            key=lambda item: item.contribution,
            reverse=True,
        )
    )
    negative = tuple(
        sorted(
            (item for item in features if item.contribution < 0),
            key=lambda item: item.contribution,
        )
    )
    intercept = float(model.classifier.intercept_[0])
    positive_sum = math.fsum(item.contribution for item in positive)
    negative_sum = math.fsum(item.contribution for item in negative)
    return DecisionEvidence(
        intercept=intercept,
        positive_sum=positive_sum,
        negative_sum=negative_sum,
        logit=intercept + positive_sum + negative_sum,
        margin_points=score - model.threshold * 100,
        positive_count=len(positive),
        negative_count=len(negative),
        positive_features=positive[:DISPLAY_LIMIT],
        negative_features=negative[:DISPLAY_LIMIT],
    )


@dataclass(frozen=True, slots=True)
class Removal:
    """Absence of recognized features withholds the intercept-only score."""

    recognized_features: int
    score: float | None


def score_removals(model: Model, texts: tuple[str, ...]) -> tuple[Removal, ...]:
    """Score retained features without treating an empty vector as evidence."""
    matrix = model.vectorizer.transform(texts)
    counts = matrix.getnnz(axis=1)
    scores = model.classifier.predict_proba(matrix)[:, 1]
    return tuple(
        Removal(int(count), float(score) * 100 if count else None)
        for count, score in zip(counts, scores, strict=True)
    )
