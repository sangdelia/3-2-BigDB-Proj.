"""Parse original K-HATERS ratings and define explicit project aggregates."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pathlib import Path

REVISION: Final = "fe022af988c3dc5240340fc89629ca1e153fe8bd"
Rating: TypeAlias = Literal[0, 1, 2]
ScoreKind: TypeAlias = Literal["hate", "abuse", "swear"]
KINDS: Final[tuple[ScoreKind, ...]] = ("hate", "abuse", "swear")


class Ratings(BaseModel):
    """Require all thirteen original ratings; absent values are never zero-filled."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="ignore", strict=True
    )
    text: str
    gender: Rating
    age: Rating
    disability: Rating
    job: Rating
    race: Rating
    religion: Rating
    politics: Rating
    individuals: Literal[0, 1]
    others: Literal[0, 1]
    insult: Rating
    swear: Rating
    threat: Rating
    obscenity: Rating

    def grade(self, kind: ScoreKind) -> Rating:
        """Aggregate original ratings before training, rather than class outputs."""
        protected = (
            self.gender,
            self.age,
            self.disability,
            self.job,
            self.race,
            self.religion,
            self.politics,
        )
        match kind:
            case "hate":
                return max(protected)
            case "abuse":
                return max(
                    (
                        *protected,
                        self.individuals,
                        self.others,
                        self.insult,
                        self.swear,
                        self.threat,
                        self.obscenity,
                    )
                )
            case "swear":
                return self.swear
            case unreachable:
                assert_never(unreachable)

    def signature(self) -> tuple[Rating, ...]:
        """Compare all ratings when detecting conflicting duplicate annotations."""
        return (
            self.gender,
            self.age,
            self.disability,
            self.job,
            self.race,
            self.religion,
            self.politics,
            self.individuals,
            self.others,
            self.insult,
            self.swear,
            self.threat,
            self.obscenity,
        )


def text_key(text: str) -> str:
    """Group exact NFC and whitespace normalized text without removing context."""
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Dataset:
    """Retain official split assignments and a reproducible exclusion ledger."""

    splits: tuple[tuple[Ratings, ...], ...]
    original_counts: tuple[int, ...]
    excluded_rows: tuple[int, ...]
    empty_rows: tuple[int, ...]
    cross_split_keys: int
    conflicting_keys: int
    file_sha256: tuple[str, ...]


def load_dataset(root: Path) -> Dataset:
    """Drop cross-split text groups and conflicts; merge same-split duplicates."""
    originals: list[tuple[Ratings, ...]] = []
    digests: list[str] = []
    for split in ("train", "val", "test"):
        data = (root / f"{split}.jsonl").read_bytes()
        digests.append(hashlib.sha256(data).hexdigest())
        originals.append(
            tuple(
                Ratings.model_validate_json(line)
                for line in data.splitlines()
                if line.strip()
            )
        )
    membership: dict[str, set[int]] = {}
    signatures: dict[str, set[tuple[Rating, ...]]] = {}
    for i, rows in enumerate(originals):
        for row in rows:
            if not row.text.strip():
                continue
            key = text_key(row.text)
            membership.setdefault(key, set()).add(i)
            signatures.setdefault(key, set()).add(row.signature())
    crossing = {key for key, splits in membership.items() if len(splits) > 1}
    conflicting = {key for key, values in signatures.items() if len(values) > 1}
    excluded = crossing | conflicting
    cleaned: list[tuple[Ratings, ...]] = []
    for rows in originals:
        unique = {
            text_key(row.text): row
            for row in reversed(rows)
            if row.text.strip() and text_key(row.text) not in excluded
        }
        cleaned.append(tuple(unique.values()))
    return Dataset(
        splits=tuple(cleaned),
        original_counts=tuple(map(len, originals)),
        excluded_rows=tuple(
            len(a) - len(b) for a, b in zip(originals, cleaned, strict=True)
        ),
        cross_split_keys=len(crossing),
        empty_rows=tuple(
            sum(not row.text.strip() for row in rows) for rows in originals
        ),
        conflicting_keys=len(conflicting),
        file_sha256=tuple(digests),
    )


def distribution(rows: tuple[Ratings, ...], kind: ScoreKind) -> tuple[int, ...]:
    """Report each grade count including zero-count categories."""
    counts = Counter(row.grade(kind) for row in rows)
    return tuple(counts[rating] for rating in (0, 1, 2))
