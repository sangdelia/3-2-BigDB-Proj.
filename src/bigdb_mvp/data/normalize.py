"""Source-specific normalization boundaries."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from typing import Final

from typing_extensions import override

from bigdb_mvp.data.models import (
    BinaryLabel,
    RecordDraft,
    SourceName,
    SupervisedRecord,
)

TOXICITY_THRESHOLD: Final = 0.5
KHATERS_LABELS: Final = {
    "normal": BinaryLabel.CLEAN,
    "offensive": BinaryLabel.HARMFUL,
    "l1_hate": BinaryLabel.HARMFUL,
    "l2_hate": BinaryLabel.HARMFUL,
}


@dataclass(frozen=True, slots=True)
class RowExclusionError(Exception):
    """Explain why an untrusted source row is excluded."""

    reason_code: str

    @override
    def __str__(self) -> str:
        """Return the stable exclusion reason code."""
        return self.reason_code


def civil_label(score: float | None) -> BinaryLabel:
    """Map a finite Civil Comments score at the inclusive threshold."""
    if score is None or not math.isfinite(score):
        raise RowExclusionError(reason_code="civil_invalid_toxicity")
    return BinaryLabel.HARMFUL if score >= TOXICITY_THRESHOLD else BinaryLabel.CLEAN


def khaters_label(raw: str | None) -> BinaryLabel:
    """Map only the normalized K-HATERS original-label vocabulary."""
    if raw is None:
        raise RowExclusionError(reason_code="khaters_original_label_missing")
    normalized = raw.strip().lower()
    label = KHATERS_LABELS.get(normalized)
    if label is None:
        raise RowExclusionError(reason_code="khaters_original_label_unknown")
    return label


def require_train_split(split: str) -> None:
    """Reject every source role other than an explicit train split."""
    if split.strip().lower() != "train":
        raise RowExclusionError(reason_code="non_train_split")


def group_id_for(source: SourceName, text: str, preferred_id: str | None) -> str:
    """Create a private group digest from a preferred ID or normalized text."""
    if preferred_id is not None and preferred_id.strip():
        key = f"{source.value}:id:{preferred_id.strip()}"
    else:
        normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
        normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()
        key = f"{source.value}:text:{normalized_hash}"
    return hashlib.sha256(key.encode()).hexdigest()


def build_record(draft: RecordDraft) -> SupervisedRecord:
    """Parse a source draft into the common immutable schema."""
    text = draft.text.strip()
    if not text:
        raise RowExclusionError(reason_code="empty_text")
    require_train_split(draft.source_split)
    return SupervisedRecord(
        text=text,
        label=draft.label,
        language=draft.language,
        source=draft.source,
        source_split="train",
        group_id=group_id_for(draft.source, text, draft.preferred_group_id),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
