"""Verified Korean sentence inference with traceable, non-gold span clues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal, assert_never

import joblib
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bigdb_mvp.baseline import BaselineModel, extract_positive_ngrams
from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification_contracts import (
    ClassificationReport,
    DatasetManifest,
    ExperimentError,
)
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.khaters_score import Prediction as RatingPrediction
from bigdb_mvp.korean_explanation import (
    DecisionEvidence,
    FeatureEvidence,
    SpanCandidate,
    explain_decision,
    score_removals,
)
from bigdb_mvp.korean_model_notes import LIMITATIONS

ROOT: Final = Path(__file__).resolve().parents[2]
HANGUL: Final = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")
SPACE_RUN: Final = re.compile(r"\s\s+")
MAX_FEATURES: Final = 24
MAX_CANDIDATES: Final = 6


class KoreanRequest(BaseModel):
    """Keep original offsets while rejecting blank or non-Korean input."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def require_korean(cls, text: str) -> str:
        """Allow mixed-script Korean messages without stripping original text."""
        if not text.strip() or HANGUL.search(text) is None:
            reason = "korean_text_required"
            raise ExperimentError(reason)
        return text


class KoreanPrediction(BaseModel):
    """Separate sentence classification from uncertain explanation candidates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    label: Literal["clean", "harmful", "review"]
    recognized_feature_count: int = Field(ge=0)
    review_reason: Literal["no_recognized_features"] | None = None
    harmful_score: float = Field(ge=0, le=100)
    threshold_score: float = Field(gt=0, lt=100)
    model_sha256: str
    dataset_sha256: str
    offset_unit: Literal["unicode_codepoint"] = "unicode_codepoint"
    explanation_status: Literal["candidates", "insufficient"]
    candidates: tuple[SpanCandidate, ...]
    limitations: tuple[str, ...] = LIMITATIONS
    rating_scores: RatingPrediction | None = None
    decision_evidence: DecisionEvidence | None = None


@dataclass(frozen=True, slots=True)
class _Hit:
    start: int
    end: int
    evidence: FeatureEvidence


def _feature_view(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Map sklearn lowercase and repeated-whitespace normalization to raw text."""
    lowered = text.lower()
    offsets = tuple((i, i + 1) for i, char in enumerate(text) for _ in char.lower())
    characters: list[str] = []
    mapped: list[tuple[int, int]] = []
    cursor = 0
    for match in SPACE_RUN.finditer(lowered):
        characters.extend(lowered[cursor : match.start()])
        mapped.extend(offsets[cursor : match.start()])
        characters.append(" ")
        mapped.append((offsets[match.start()][0], offsets[match.end() - 1][1]))
        cursor = match.end()
    characters.extend(lowered[cursor:])
    mapped.extend(offsets[cursor:])
    return "".join(characters), tuple(mapped)


def _hits(text: str, model: Model) -> tuple[_Hit, ...]:
    """Use actual TF-IDF contributions; never infer absent features from a lexicon."""
    normalized, offsets = _feature_view(text)
    features = extract_positive_ngrams(
        text, BaselineModel(model.vectorizer, model.classifier), MAX_FEATURES
    )
    hits: list[_Hit] = []
    for feature in features:
        matches = tuple(
            match.start()
            for match in re.finditer(f"(?={re.escape(feature.ngram)})", normalized)
        )
        if not matches:
            reason = "feature_offset_mismatch"
            raise ExperimentError(reason)
        evidence = FeatureEvidence(
            ngram=feature.ngram,
            contribution=feature.contribution,
            occurrence_contribution=feature.contribution / len(matches),
        )
        hits.extend(
            _Hit(
                offsets[start][0], offsets[start + len(feature.ngram) - 1][1], evidence
            )
            for start in matches
        )
    return tuple(sorted(hits, key=lambda hit: (hit.start, hit.end)))


def _groups(hits: tuple[_Hit, ...]) -> tuple[tuple[_Hit, ...], ...]:
    """Merge intersecting occurrences without duplicating feature contributions."""
    groups: list[list[_Hit]] = []
    end = 0
    for hit in hits:
        if groups and hit.start < end:
            groups[-1].append(hit)
            end = max(end, hit.end)
        else:
            groups.append([hit])
            end = hit.end
    return tuple(tuple(group) for group in groups)


@dataclass(frozen=True, slots=True)
class KoreanModelService:
    """Hold a verified immutable model; requests are neither stored nor logged."""

    model: Model
    model_sha256: str
    dataset_sha256: str

    def predict(self, request: KoreanRequest) -> KoreanPrediction:
        """Return real sentence scores and ranked, independently rescored spans."""
        text = request.text
        matrix = self.model.vectorizer.transform((text,))
        recognized = int(matrix.getnnz(axis=1)[0])
        score = float(self.model.classifier.predict_proba(matrix)[:, 1][0]) * 100
        selected = sorted(
            _groups(_hits(text, self.model)),
            key=lambda group: sum(
                hit.evidence.occurrence_contribution for hit in group
            ),
            reverse=True,
        )[:MAX_CANDIDATES]
        bounds = tuple(
            (min(hit.start for hit in group), max(hit.end for hit in group))
            for group in selected
        )
        candidates: list[SpanCandidate] = []
        if bounds:
            removed = tuple(text[:start] + text[end:] for start, end in bounds)
            comparisons = score_removals(self.model, removed)
            for group, (start, end), comparison in zip(
                selected, bounds, comparisons, strict=True
            ):
                score_without = comparison.score
                drop = None if score_without is None else score - score_without
                candidates.append(
                    SpanCandidate(
                        start=start,
                        end=end,
                        text=text[start:end],
                        positive_contribution=sum(
                            hit.evidence.occurrence_contribution for hit in group
                        ),
                        score_without=score_without,
                        score_drop_points=drop,
                        recognized_features_without=comparison.recognized_features,
                        highlight_as_harmful_candidate=(
                            score >= self.model.threshold * 100
                            and drop is not None
                            and drop > 0
                        ),
                        evidence=tuple(hit.evidence for hit in group),
                    )
                )
        return KoreanPrediction(
            label=(
                "review"
                if recognized == 0
                else "harmful"
                if score >= self.model.threshold * 100
                else "clean"
            ),
            recognized_feature_count=recognized,
            review_reason="no_recognized_features" if recognized == 0 else None,
            harmful_score=score,
            threshold_score=self.model.threshold * 100,
            model_sha256=self.model_sha256,
            dataset_sha256=self.dataset_sha256,
            explanation_status="candidates" if candidates else "insufficient",
            candidates=tuple(candidates),
            decision_evidence=(
                explain_decision(self.model, text, score) if recognized else None
            ),
        )


def load_korean_model(
    artifact_root: Path = ROOT / "artifacts/korean_v2",
    report_path: Path = ROOT / "reports/korean_v2/metrics.json",
    dataset_root: Path = ROOT / "data/curated/korean_v2",
    method: str = "char_tfidf",
) -> KoreanModelService:
    """Verify local provenance before unpickling a trusted local model artifact."""
    report = ClassificationReport.model_validate_json(report_path.read_bytes())
    dataset = DatasetManifest.model_validate_json(
        (dataset_root / "manifest.json").read_bytes()
    )
    artifact = DatasetManifest.model_validate_json(
        (artifact_root / "manifest.json").read_bytes()
    )
    if not (
        report.configuration.dataset_sha256
        == dataset.dataset_sha256
        == artifact.dataset_sha256
        == sha256_file(dataset_root / "corpus.parquet")
    ):
        reason = "dataset_digest_mismatch"
        raise ExperimentError(reason)
    results = tuple(result for result in report.results if result.method == method)
    if len(results) != 1:
        reason = "baseline_result_missing_or_duplicate"
        raise ExperimentError(reason)
    result = results[0]
    model_path = artifact_root / f"{method}.joblib"
    if sha256_file(model_path) != result.model_sha256:
        reason = "model_digest_mismatch"
        raise ExperimentError(reason)
    match joblib.load(model_path, mmap_mode="r"):
        case Model() as model:
            pass
        case BaselineModel():
            reason = "wrong_model_type"
            raise ExperimentError(reason)
        case unreachable:
            assert_never(unreachable)
    if (
        model.lexicon is not None
        or model.context
        or model.threshold != result.threshold
    ):
        reason = "model_configuration_mismatch"
        raise ExperimentError(reason)
    if not 0 < model.threshold < 1:
        reason = "invalid_threshold"
        raise ExperimentError(reason)
    return KoreanModelService(model, result.model_sha256, dataset.dataset_sha256)
