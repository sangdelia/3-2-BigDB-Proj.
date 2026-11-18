"""Stable import path for persisted experiment models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from bigdb_mvp.experiments.lexicon import Lexicon


@dataclass(frozen=True, slots=True)
class Model:
    """Persist all transformations alongside the classifier and frozen threshold."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    lexicon: Lexicon | None
    threshold: float
    context: bool
