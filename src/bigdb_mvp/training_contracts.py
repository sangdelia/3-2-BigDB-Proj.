"""Immutable contracts that prevent partial XLM-R checkpoints from serving."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing_extensions import override

from bigdb_mvp.baseline_contracts import MetricSummary, PredictionCount, SliceMetrics
from bigdb_mvp.config import MODEL_REPO_ID

if TYPE_CHECKING:
    from pathlib import Path

    from bigdb_mvp.doctor import DoctorReport

XLM_ROBERTA_REVISION: Final = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
MINIMUM_SCORE_STANDARD_DEVIATION: Final = 1e-6


@dataclass(frozen=True, slots=True)
class XlmRow:
    """One validated classifier row with binary integer target."""

    text: str
    label: int
    language: str
    split: str


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Complete train/calibration/test request consumed by a GPU backend."""

    config: XlmTrainingConfig
    batches: tuple[tuple[XlmRow, ...], ...]
    calibration: tuple[XlmRow, ...]
    test: tuple[XlmRow, ...]


@dataclass(frozen=True, slots=True)
class BackendOutput:
    """Temporary logits and checkpoint outputs from one backend run."""

    calibration_logits: tuple[float, ...]
    test_logits: tuple[float, ...]
    checkpoint_dir: Path
    tokenizer_dir: Path


@dataclass(frozen=True, slots=True)
class XlmBackendError(Exception):
    """Typed terminal GPU backend failure safe for a partial manifest."""

    state: TrainingState
    reason_code: str

    @override
    def __str__(self) -> str:
        """Return the stable terminal reason code."""
        return self.reason_code


class TrainingBackend(Protocol):
    """Narrow execution seam shared by PyTorch and deterministic test fakes."""

    def train(self, request: TrainingRequest, output_dir: Path) -> BackendOutput:
        """Run exactly one prepared training request."""
        ...


class TrainingState(StrEnum):
    """Terminal and transient states for the GPU-only XLM-R run."""

    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETE = "complete"
    OOM = "oom"
    CUDA_ERROR = "cuda_error"
    CANCELLED = "cancelled"


class XlmTrainingConfig(BaseModel):
    """Pinned one-epoch XLM-R training inputs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    model_id: Literal["FacebookAI/xlm-roberta-base"] = MODEL_REPO_ID
    revision: Literal["e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"] = XLM_ROBERTA_REVISION
    training_languages: Literal["ko"] = "ko"
    max_length: int = Field(default=128, ge=1, le=128)
    batch_size: int = Field(default=16, ge=2, le=16)
    seed: int = Field(default=42, ge=0)
    epochs: int = Field(default=1, ge=1, le=1)
    device: Literal["cuda:0"] = "cuda:0"


class SliceScoreStandardDeviation(BaseModel):
    """Score dispersion for one required held-out evaluation slice."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    language: Literal["overall", "ko", "en"]
    value: float = Field(gt=MINIMUM_SCORE_STANDARD_DEVIATION)


class XlmArtifact(BaseModel):
    """Persisted model manifest with a derived, conservative serving gate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    artifact_version: int = 1
    model_kind: str = "xlm_roberta"
    state: TrainingState
    reason_code: str | None = None
    config: XlmTrainingConfig = XlmTrainingConfig()
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tokenizer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibrator_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_complete: bool = False
    checkpoint_digest_verified: bool = False
    calibrator_temperature: float | None = Field(default=None, gt=0.0)
    threshold: float | None = Field(default=None, ge=0.05, le=0.95)
    score_standard_deviation: float | None = Field(default=None, ge=0.0)
    language_score_standard_deviations: tuple[SliceScoreStandardDeviation, ...] = ()
    metrics: tuple[SliceMetrics, ...] = ()
    prediction_distribution: tuple[PredictionCount, ...] = ()

    @computed_field
    @property
    def eligible_for_serving(self) -> bool:
        """Allow only a complete, calibrated, evaluated CUDA checkpoint."""
        has_digests = all(
            value is not None
            for value in (
                self.dataset_sha256,
                self.logical_digest,
                self.tokenizer_sha256,
                self.checkpoint_sha256,
                self.calibrator_sha256,
            )
        )
        has_calibrator = self.calibrator_temperature is not None and math.isfinite(
            self.calibrator_temperature
        )
        has_variance = (
            self.score_standard_deviation is not None
            and math.isfinite(self.score_standard_deviation)
            and self.score_standard_deviation > MINIMUM_SCORE_STANDARD_DEVIATION
        )
        slice_variances_are_finite = all(
            math.isfinite(item.value)
            for item in self.language_score_standard_deviations
        )
        metric_languages = {item.language for item in self.metrics}
        variance_languages = {
            item.language for item in self.language_score_standard_deviations
        }
        prediction_labels = {item.label for item in self.prediction_distribution}
        return (
            self.state is TrainingState.COMPLETE
            and self.checkpoint_complete
            and self.checkpoint_digest_verified
            and self.config.device == "cuda:0"
            and self.config.model_id == MODEL_REPO_ID
            and self.config.revision == XLM_ROBERTA_REVISION
            and has_digests
            and has_calibrator
            and self.threshold is not None
            and has_variance
            and slice_variances_are_finite
            and {"overall", "ko"}.issubset(metric_languages)
            and {"overall", "ko"}.issubset(variance_languages)
            and {"clean", "harmful"}.issubset(prediction_labels)
        )

    @classmethod
    def partial(cls, *, state: TrainingState, reason_code: str) -> XlmArtifact:
        """Build an explicitly ineligible partial-training record."""
        return cls(state=state, reason_code=reason_code)

    @classmethod
    def complete_for_test(cls, *, calibrator_temperature: float | None) -> XlmArtifact:
        """Create an otherwise complete artifact fixture for serving-gate tests."""
        metric = MetricSummary(
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            roc_auc=0.5,
            brier=0.25,
            sample_count=2,
        )
        return cls(
            state=TrainingState.COMPLETE,
            dataset_sha256="a" * 64,
            logical_digest="b" * 64,
            tokenizer_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            calibrator_sha256="e" * 64,
            checkpoint_complete=True,
            checkpoint_digest_verified=True,
            calibrator_temperature=calibrator_temperature,
            threshold=0.5,
            score_standard_deviation=0.1,
            language_score_standard_deviations=tuple(
                SliceScoreStandardDeviation(language=language, value=0.1)
                for language in ("overall", "ko", "en")
            ),
            metrics=tuple(
                SliceMetrics(language=language, metrics=metric)
                for language in ("overall", "ko", "en")
            ),
            prediction_distribution=(
                PredictionCount(label="clean", count=1),
                PredictionCount(label="harmful", count=1),
            ),
        )


class BlockedTrainingReceipt(BaseModel):
    """Machine-readable terminal receipt for a preflight-blocked GPU run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    state: TrainingState = TrainingState.BLOCKED
    eligible_for_serving: bool = False
    command: tuple[str, ...]
    reason_code: str
    diagnostic: str | None
    remediation: str
    config: XlmTrainingConfig


def blocked_receipt(
    report: DoctorReport, config: XlmTrainingConfig
) -> BlockedTrainingReceipt:
    """Preserve the actual GPU failure without creating a model artifact."""
    return BlockedTrainingReceipt(
        command=(
            "bigdb-mvp",
            "train",
            "xlm-roberta",
            "--epochs",
            str(config.epochs),
            "--device",
            config.device,
            "--seed",
            str(config.seed),
            "--training-languages",
            config.training_languages,
        ),
        reason_code=report.primary_reason_code.value,
        diagnostic=report.nvml_diagnostic,
        remediation=(
            "Align the installed NVIDIA kernel driver and NVML user-space library, "
            "then rerun the GPU0 doctor before training."
        ),
        config=config,
    )
