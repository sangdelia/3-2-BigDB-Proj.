"""Learn optional dictionary signals exclusively from training document counts."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from bigdb_mvp.experiments.classification_contracts import Row

TOKEN: Final = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]{2,}")


@dataclass(frozen=True, slots=True)
class Lexicon:
    """Selected words are predictive associations, never automatic decisions."""

    positive: tuple[str, ...]
    negative: tuple[str, ...]


def learn_lexicon(
    rows: tuple[Row, ...], min_count: int = 5, limit: int = 500
) -> Lexicon:
    """Estimate smoothed class document log odds on train only."""
    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    train = tuple(row for row in rows if row.split == "train")
    n_positive = sum(row.label == "harmful" for row in train)
    n_negative = len(train) - n_positive
    for row in train:
        counts = positive if row.label == "harmful" else negative
        counts.update(set(TOKEN.findall(row.text.lower())))
    scores = {
        word: math.log((positive[word] + 1) / (n_positive - positive[word] + 1))
        - math.log((negative[word] + 1) / (n_negative - negative[word] + 1))
        for word in positive.keys() | negative.keys()
        if positive[word] + negative[word] >= min_count
    }
    ordered = sorted(scores, key=lambda word: (-scores[word], word))
    return Lexicon(
        positive=tuple(word for word in ordered if scores[word] > 0)[:limit],
        negative=tuple(word for word in reversed(ordered) if scores[word] < 0)[:limit],
    )


def lexicon_features(texts: tuple[str, ...], lexicon: Lexicon) -> csr_matrix:
    """Append two log-scaled counts so the classifier can learn their weights."""
    positive, negative = set(lexicon.positive), set(lexicon.negative)
    values = np.zeros((len(texts), 2), dtype=np.float64)
    for index, text in enumerate(texts):
        tokens: list[str] = TOKEN.findall(text.lower())
        values[index] = np.log1p(
            [
                sum(token in positive for token in tokens),
                sum(token in negative for token in tokens),
            ]
        )
    return csr_matrix(values)
