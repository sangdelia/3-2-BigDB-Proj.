"""Typed records shared by data ingestion and training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class BinaryLabel(StrEnum):
    """Shared binary target."""

    CLEAN = "clean"
    HARMFUL = "harmful"


class Language(StrEnum):
    """Supported language."""

    KO = "ko"
    EN = "en"


class SourceName(StrEnum):
    """Approved source identifier."""

    NIKL = "nikl"
    BEEP = "beep"
    KMHAS = "kmhas"
    KHATERS = "khaters"
    CIVIL_COMMENTS = "civil_comments"


class SplitName(StrEnum):
    """Runtime dataset split."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class SupervisedRecord(BaseModel):
    """Validated common record without upstream-only metadata."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    label: BinaryLabel
    language: Language
    source: SourceName
    source_split: str
    group_id: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: SplitName | None = None

    def with_split(self, split: SplitName) -> SupervisedRecord:
        """Return this immutable record with its runtime split assigned."""
        return self.model_copy(update={"split": split})


@dataclass(frozen=True, slots=True)
class RecordDraft:
    """Values needed to create one common supervised record."""

    text: str
    label: BinaryLabel
    language: Language
    source: SourceName
    source_split: str
    preferred_group_id: str | None = None
