"""Digest-verified, non-persistent bilingual prediction service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from typing_extensions import override

from bigdb_mvp.baseline import (
    ARTIFACT_DIR,
    ArtifactIntegrityError,
    LoadedBaseline,
    extract_positive_ngrams,
    load_baseline,
)
from bigdb_mvp.contracts import (
    PositiveEvidence,
    PredictionRequest,
    PredictionResponse,
    ScoreEvidence,
)
from bigdb_mvp.suggestions import LanguageCode, suggest_rewrite
from bigdb_mvp.xlm_inference import XlmPrediction, predict_xlm_probability
from bigdb_mvp.xlm_training import ARTIFACT_DIR as XLM_ARTIFACT_DIR

if TYPE_CHECKING:
    from pathlib import Path

HANGUL: Final = re.compile(r"[\uac00-\ud7a3]")
LATIN: Final = re.compile(r"[A-Za-z]")
BASELINE_LIMITATION: Final = (
    "Eligible XLM-R artifact is unavailable; using the CPU character n-gram baseline."
)
NGRAM_LIMITATION: Final = (
    "Positive n-gram evidence is a baseline clue, not a learned harmful-span "
    "explanation."
)
XLM_LIMITATION: Final = (
    "XLM-R output is sentence-level classification, not a learned harmful-span "
    "explanation."
)


@dataclass(frozen=True, slots=True)
class ModelUnavailableError(Exception):
    """Report a missing or integrity-invalid local inference artifact."""

    reason_code: str

    @override
    def __str__(self) -> str:
        """Return the safe reason code without request text."""
        return self.reason_code


def detect_language(text: str) -> LanguageCode:
    """Use deterministic script counts; ties intentionally select English."""
    return "ko" if len(HANGUL.findall(text)) > len(LATIN.findall(text)) else "en"


class PredictionService:
    """Prefer verified XLM-R and otherwise serve the digest-verified baseline."""

    def __init__(
        self,
        artifact_dir: Path = ARTIFACT_DIR,
        xlm_artifact_dir: Path = XLM_ARTIFACT_DIR,
    ) -> None:
        """Load fallback state once without retaining request data."""
        self._xlm_artifact_dir: Path = xlm_artifact_dir
        try:
            self._loaded: LoadedBaseline = load_baseline(artifact_dir=artifact_dir)
        except (ArtifactIntegrityError, FileNotFoundError, OSError) as error:
            raise ModelUnavailableError(reason_code="model_unavailable") from error

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Create one in-memory prediction response from a validated request."""
        language = detect_language(request.text)
        xlm_prediction = self._xlm_prediction(request.text)
        if xlm_prediction is not None:
            return self._xlm_response(request.text, language, xlm_prediction)
        return self._baseline_response(request.text, language)

    def _xlm_prediction(self, text: str) -> XlmPrediction | None:
        """Treat malformed or unavailable XLM-R artifacts as an explicit fallback."""
        try:
            return predict_xlm_probability(text, self._xlm_artifact_dir)
        except (OSError, ValueError):
            return None

    def _xlm_response(
        self, text: str, language: LanguageCode, prediction: XlmPrediction
    ) -> PredictionResponse:
        """Render one sentence-level XLM-R outcome with verified provenance."""
        digest = prediction.artifact.checkpoint_sha256
        if digest is None:
            return self._baseline_response(text, language)
        harmful = prediction.harmful_probability >= prediction.threshold
        label: Literal["clean", "harmful"] = "harmful" if harmful else "clean"
        return PredictionResponse(
            label=label,
            harmful_probability=prediction.harmful_probability,
            language=language,
            model_kind=prediction.artifact.model_kind,
            model_digest=digest,
            limitations=(XLM_LIMITATION,),
            evidence=(
                self._score_evidence(
                    prediction.harmful_probability, prediction.threshold, harmful
                ),
            ),
            suggestions=suggest_rewrite(text, language=language, harmful=harmful),
        )

    def _baseline_response(
        self, text: str, language: LanguageCode
    ) -> PredictionResponse:
        """Render the deliberate CPU fallback with baseline-only evidence."""
        probability = self._harmful_probability(text)
        harmful = probability >= self._loaded.artifact.threshold
        label: Literal["clean", "harmful"] = "harmful" if harmful else "clean"
        return PredictionResponse(
            label=label,
            harmful_probability=probability,
            language=language,
            model_kind=self._loaded.artifact.model_kind,
            model_digest=self._loaded.artifact.model_sha256,
            limitations=(BASELINE_LIMITATION, NGRAM_LIMITATION),
            evidence=(
                self._score_evidence(
                    probability, self._loaded.artifact.threshold, harmful
                ),
                *self._positive_evidence(text),
            ),
            suggestions=suggest_rewrite(text, language=language, harmful=harmful),
        )

    def predict_text(self, text: str) -> PredictionResponse:
        """Parse direct CLI text at the same Pydantic request boundary."""
        return self.predict(PredictionRequest(text=text))

    def _harmful_probability(self, text: str) -> float:
        """Read the baseline class-one probability without storing text."""
        scores = tuple(
            float(score)
            for score in self._loaded.model.classifier.predict_proba(
                self._loaded.model.vectorizer.transform((text,))
            )[:, 1]
        )
        return scores[0]

    def _positive_evidence(self, text: str) -> tuple[PositiveEvidence, ...]:
        """Map model-positive n-grams to first Unicode spans when present."""
        return tuple(
            PositiveEvidence(
                ngram=item.ngram,
                contribution=item.contribution,
                start=text.find(item.ngram),
                end=text.find(item.ngram) + len(item.ngram),
            )
            for item in extract_positive_ngrams(text, self._loaded.model)
            if text.find(item.ngram) >= 0
        )

    @staticmethod
    def _score_evidence(
        probability: float, threshold: float, harmful: bool
    ) -> ScoreEvidence:
        """Expose the numeric decision boundary used by the serving model."""
        return ScoreEvidence(
            harmful_probability=probability,
            threshold=threshold,
            decision="above_threshold" if harmful else "below_threshold",
        )
