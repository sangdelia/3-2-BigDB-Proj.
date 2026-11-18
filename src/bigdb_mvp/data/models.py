"""Typed records shared by data ingestion and training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    UNSMILE = "unsmile"
    KOLD = "kold"
    BEEP = "beep"
    KMHAS = "kmhas"
    KHATERS = "khaters"
    CIVIL_COMMENTS = "civil_comments"


class SplitName(StrEnum):
    """Runtime dataset split."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class CanonicalRole(StrEnum):
    """Allowed roles in the cap-free canonical corpus."""

    RESEARCH_INDEX = "research_index"
    CALIBRATION_QUERY = "calibration_query"
    TEST_QUERY = "test_query"
    PRODUCTION = "production"
    UNLABELED_INVENTORY = "unlabeled_inventory"


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


class CanonicalRecord(BaseModel):
    """Validated cap-free row; text is retained only in the canonical artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    label: BinaryLabel | None = None
    language: Language
    source: SourceName
    source_split: str
    role: CanonicalRole
    group_id: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_role_contract(self) -> CanonicalRecord:
        """Require labels for supervised roles and forbid them for inventory."""
        match self.role:
            case CanonicalRole.UNLABELED_INVENTORY:
                if self.label is not None:
                    reason = "unlabeled_inventory_label"
                    raise ValueError(reason)
            case (
                CanonicalRole.RESEARCH_INDEX
                | CanonicalRole.CALIBRATION_QUERY
                | CanonicalRole.TEST_QUERY
                | CanonicalRole.PRODUCTION
            ):
                if self.label is None:
                    reason = "supervised_role_label"
                    raise ValueError(reason)
            case unreachable:
                assert_never(unreachable)
        return self

    def stable_sort_key(self) -> tuple[str, ...]:
        """Return the field order used for deterministic publication."""
        return (
            self.role.value,
            self.language.value,
            self.label.value if self.label is not None else "",
            self.source.value,
            self.source_split,
            self.group_id,
            self.text_sha256,
        )


@dataclass(frozen=True, slots=True)
class RecordDraft:
    """Values needed to create one common supervised record."""

    text: str
    label: BinaryLabel
    language: Language
    source: SourceName
    source_split: str
    preferred_group_id: str | None = None
