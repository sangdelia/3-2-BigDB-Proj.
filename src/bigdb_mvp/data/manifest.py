"""Safe aggregate runtime manifest models."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.data.models import BinaryLabel, Language, SourceName, SplitName


class ExclusionCount(BaseModel):
    """Count for one fail-closed exclusion reason."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    reason: str
    count: int = Field(ge=0)


class SourceReceipt(BaseModel):
    """Aggregate source provenance without filesystem paths or raw text."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: SourceName
    revision: str
    license: str
    input_sha256: str
    asset: str = ""
    staging_sha256: str = ""
    scanned: int = 0
    accepted: int = Field(ge=0)
    excluded: int = Field(ge=0)
    exclusions: tuple[ExclusionCount, ...] = ()


class DataCount(BaseModel):
    """Aggregate record count for one complete data cell."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    language: Language
    label: BinaryLabel
    split: SplitName
    source: SourceName
    count: int = Field(ge=0)


class SplitIntersection(BaseModel):
    """Number of group identifiers shared by two output splits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    left: SplitName
    right: SplitName
    count: int = Field(ge=0)


class Manifest(BaseModel):
    """Runtime dataset manifest safe to commit as evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    seed: int
    sources: tuple[SourceReceipt, ...]
    status: Literal["complete"] = "complete"
    dataset_file: str = ""
    dataset_sha256: str = ""
    logical_digest: str = ""
    record_count: int = 0
    counts: tuple[DataCount, ...] = ()
    exact_duplicates_removed: int = 0
    sampled_non_train_count: int = 0
    split_intersections: tuple[SplitIntersection, ...] = ()
    raw_text_fields: int = 0
    thesis_path_references: int = 0

    @classmethod
    def empty(cls, *, seed: int) -> Manifest:
        """Create a safe empty manifest for composition."""
        return cls(schema_version=1, seed=seed, sources=())

    def with_source(self, source: SourceReceipt) -> Manifest:
        """Return the immutable manifest with one source receipt."""
        return self.model_copy(update={"sources": (*self.sources, source)})
