"""Typed persisted contract for the deterministic CPU baseline."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class MetricSummary(BaseModel):
    """Finite binary-classification metrics for one evaluation slice."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    roc_auc: float = Field(ge=0.0, le=1.0)
    brier: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=1)


class SliceMetrics(BaseModel):
    """Metrics tied to a named language slice."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    language: str
    metrics: MetricSummary


class PredictionCount(BaseModel):
    """Predicted count for one required output label."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    label: str
    count: int = Field(ge=0)


class BaselineArtifact(BaseModel):
    """Digest-bound metadata for one trained TF-IDF/logistic artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    artifact_version: int = 1
    model_kind: str = "char_tfidf_logistic"
    seed: int = Field(ge=0)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_file: str = "baseline_model.joblib"
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(ge=0.05, le=0.95)
    score_standard_deviation: float = Field(ge=0.0)
    metrics: tuple[SliceMetrics, ...]
    prediction_distribution: tuple[PredictionCount, ...]
    test_evaluated_once: bool = True
