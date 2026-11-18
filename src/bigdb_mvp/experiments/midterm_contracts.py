"""Freeze comparable rows and validation-only selection for the midterm study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from bigdb_mvp.experiments.hypothesis_cases import normalize
from bigdb_mvp.experiments.khaters_ratings import Dataset, Rating, Ratings, text_key

TrialName: TypeAlias = Literal["baseline", "normalized", "balanced"]


def normalize_features(text: str) -> str:
    """Preserve TF-IDF's default lowercasing when applying bounded normalization."""
    return normalize(text.lower())


class Audit(BaseModel):
    """Record additional exclusions after the original-rating cleanup."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    before: tuple[int, ...]
    retained: tuple[int, ...]
    removed: tuple[int, ...]
    crossing_keys: int
    conflicting_keys: int
    changed_texts: tuple[int, ...]
    split_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommonData:
    """Hold the identical raw rows used by every experimental treatment."""

    splits: tuple[tuple[Ratings, ...], ...]
    audit: Audit


def common_rows(dataset: Dataset) -> CommonData:
    """Exclude normalized collisions before fitting any candidate vocabulary."""
    membership: dict[str, set[int]] = {}
    signatures: dict[str, set[tuple[Rating, ...]]] = {}
    for split, rows in enumerate(dataset.splits):
        for row in rows:
            key = text_key(normalize_features(row.text))
            membership.setdefault(key, set()).add(split)
            signatures.setdefault(key, set()).add(row.signature())
    crossing = {key for key, splits in membership.items() if len(splits) > 1}
    conflicts = {key for key, labels in signatures.items() if len(labels) > 1}
    excluded = crossing | conflicts
    retained: list[tuple[Ratings, ...]] = []
    for rows in dataset.splits:
        unique: dict[str, Ratings] = {}
        for row in rows:
            key = text_key(normalize_features(row.text))
            if key not in excluded:
                _ = unique.setdefault(key, row)
        retained.append(tuple(unique.values()))
    counts = tuple(map(len, retained))
    before = tuple(map(len, dataset.splits))
    return CommonData(
        tuple(retained),
        Audit(
            before=before,
            retained=counts,
            removed=tuple(a - b for a, b in zip(before, counts, strict=True)),
            crossing_keys=len(crossing),
            conflicting_keys=len(conflicts),
            changed_texts=tuple(
                sum(normalize_features(row.text) != row.text.lower() for row in rows)
                for rows in retained
            ),
            split_sha256=tuple(
                hashlib.sha256(
                    "\n".join(row.model_dump_json() for row in rows).encode()
                ).hexdigest()
                for rows in retained
            ),
        ),
    )


class Metrics(BaseModel):
    """Parse existing evaluator output and derive grade-specific error rates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    count: int
    grade_counts: tuple[int, int, int]
    mae_points: float
    rmse_points: float
    train_mean_mae_points: float
    grade_macro_f1: float
    confusion: tuple[tuple[int, int, int], ...]
    nll: float
    temperature: float

    @property
    def grade_zero_fpr(self) -> float:
        """Count any nonzero predicted grade among gold grade-zero rows."""
        return sum(self.confusion[0][1:]) / self.grade_counts[0]

    @property
    def grade_one_recall(self) -> float:
        """Measure exact mild-grade recognition, distinct from any-hate recall."""
        return self.confusion[1][1] / self.grade_counts[1]


def eligible(candidate: Metrics, baseline: Metrics) -> bool:
    """Apply all three preregistered validation criteria without test access."""
    return (
        candidate.mae_points < baseline.mae_points
        and candidate.grade_macro_f1 >= baseline.grade_macro_f1
        and candidate.grade_zero_fpr <= baseline.grade_zero_fpr
    )
