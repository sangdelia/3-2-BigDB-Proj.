"""Pydantic boundaries for non-persistent local prediction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

HTML_TAG: re.Pattern[str] = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")


@dataclass(frozen=True, slots=True)
class InputPolicyError(ValueError):
    """Describe one rejected text boundary rule without retaining the input."""

    reason_code: str

    @override
    def __str__(self) -> str:
        """Return only the safe machine-readable rejection reason."""
        return self.reason_code


class PredictionRequest(BaseModel):
    """Validate one in-memory prediction request at the API or CLI boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True
    )

    text: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def reject_html(self) -> PredictionRequest:
        """Reject markup because the MVP accepts plain sentences only."""
        if HTML_TAG.search(self.text):
            raise InputPolicyError(reason_code="html_not_allowed")
        return self


class RewriteSuggestion(BaseModel):
    """One deterministic curated replacement or neutral fallback."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["lexicon_substitution", "neutral_template"]
    reason_code: str = Field(min_length=1)
    source: Literal["curated_rule"] = "curated_rule"
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    revised_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_optional_span(self) -> RewriteSuggestion:
        """Require a valid half-open Unicode span exactly when present."""
        if (self.start is None) != (self.end is None):
            raise InputPolicyError(reason_code="incomplete_suggestion_span")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise InputPolicyError(reason_code="invalid_suggestion_span")
        return self


class PositiveEvidence(BaseModel):
    """A positive baseline n-gram contribution used as a limited local clue."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    ngram: str = Field(min_length=1)
    contribution: float = Field(gt=0.0)
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class ScoreEvidence(BaseModel):
    """Expose the score and threshold that produced the binary decision."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["model_score"] = "model_score"
    harmful_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    decision: Literal["above_threshold", "below_threshold"]


class PredictionResponse(BaseModel):
    """Safe public prediction result with no persistence fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    label: Literal["clean", "harmful"]
    harmful_probability: float = Field(ge=0.0, le=1.0)
    language: Literal["ko", "en"]
    model_kind: str = Field(min_length=1)
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[PositiveEvidence | ScoreEvidence, ...]
    suggestions: tuple[RewriteSuggestion, ...]
