"""Deterministic non-generative expression-improvement suggestions."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import ClassVar, Final, Literal, assert_never

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.contracts import RewriteSuggestion

REWRITE_CONFIG_PATH: Final = (
    Path(__file__).resolve().parents[2] / "config/rewrite_lexicon.yaml"
)
LanguageCode = Literal["ko", "en"]
KO_TEMPLATE: Final = "의견이 다르지만 존중하는 방식으로 표현하겠습니다."
EN_TEMPLATE: Final = "I disagree, but I will express it respectfully."


class LexiconEntry(BaseModel):
    """One curated exact-string substitution rule."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    language: LanguageCode
    term: str = Field(min_length=1)
    replacement: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)


class RewriteLexicon(BaseModel):
    """Parse the complete trusted local rewrite configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[LexiconEntry, ...] = Field(min_length=1)


@cache
def load_lexicon(path: Path = REWRITE_CONFIG_PATH) -> RewriteLexicon:
    """Load rules once; this configuration contains no user input."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return RewriteLexicon.model_validate(loaded)


def _first_match(text: str, language: LanguageCode) -> LexiconEntry | None:
    """Choose the earliest, then longest, matching rule deterministically."""
    matches = tuple(
        (text.find(entry.term), entry)
        for entry in load_lexicon().entries
        if entry.language == language and entry.term in text
    )
    valid = tuple(item for item in matches if item[0] >= 0)
    if not valid:
        return None
    return min(valid, key=lambda item: (item[0], -len(item[1].term)))[1]


def _template(language: LanguageCode) -> str:
    """Return the approved neutral template for one classified language."""
    match language:
        case "ko":
            return KO_TEMPLATE
        case "en":
            return EN_TEMPLATE
        case unreachable:
            assert_never(unreachable)


def suggest_rewrite(
    text: str, *, language: LanguageCode, harmful: bool
) -> tuple[RewriteSuggestion, ...]:
    """Return at most one deterministic suggestion, only for harmful output."""
    if not harmful:
        return ()
    entry = _first_match(text, language)
    if entry is None:
        return (
            RewriteSuggestion(
                strategy="neutral_template",
                reason_code="no_exact_curated_match",
                revised_text=_template(language),
            ),
        )
    start = text.find(entry.term)
    end = start + len(entry.term)
    return (
        RewriteSuggestion(
            strategy="lexicon_substitution",
            reason_code=entry.reason_code,
            start=start,
            end=end,
            revised_text=text[:start] + entry.replacement + text[end:],
        ),
    )
