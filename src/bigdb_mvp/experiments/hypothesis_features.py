"""Equal-budget original, normalized, and morphology feature branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, assert_never

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from bigdb_mvp.experiments.hypothesis_cases import normalize

if TYPE_CHECKING:
    from kiwipiepy import Kiwi
    from sklearn.feature_extraction.text import FeatureMatrix

ViewKind: TypeAlias = Literal["raw", "normalized", "morph"]


class Views(TypedDict):
    """Align all branches to the same input row order."""

    raw: tuple[str, ...]
    normalized: tuple[str, ...]
    morph: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Branch:
    """Retain branch identity with its fitted vocabulary."""

    kind: ViewKind
    vectorizer: TfidfVectorizer


def branch(kind: ViewKind, budget: int) -> Branch:
    """Allocate matching feature budgets with word-only tokenization options."""
    match kind:
        case "morph":
            pattern = r"(?u)\S+"
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=budget,
                token_pattern=pattern,
            )
        case "raw" | "normalized":
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                sublinear_tf=True,
                max_features=budget,
            )
        case _:
            assert_never(kind)
    return Branch(kind, vectorizer)


def views(texts: tuple[str, ...], kiwi: Kiwi) -> Views:
    """Retain original text and every morphology token, including negation."""
    return Views(
        raw=texts,
        normalized=tuple(normalize(text) for text in texts),
        morph=tuple(
            " ".join(f"{token.form}/{token.tag}" for token in tokens)
            for tokens in kiwi.tokenize(texts)
        ),
    )


def matrix(
    branches: tuple[Branch, ...], prepared: Views, *, fit: bool
) -> FeatureMatrix:
    """Fit vocabularies on training rows only and concatenate aligned branches."""
    blocks = tuple(
        branch.vectorizer.fit_transform(prepared[branch.kind])
        if fit
        else branch.vectorizer.transform(prepared[branch.kind])
        for branch in branches
    )
    return hstack(blocks, format="csr")
