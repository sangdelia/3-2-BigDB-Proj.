"""Safe aggregate runtime manifest models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    SourceName,
    SplitName,
)

CANONICAL_MANIFEST_SCHEMA_VERSION: Final[int] = 1
DEFAULT_DIGEST: Final[str] = hashlib.sha256(b"bigdb-unavailable-asset").hexdigest()


def schema_digest(columns: tuple[str, ...] | tuple[tuple[str, str], ...]) -> str:
    """Hash a stable typed source schema representation."""
    payload = repr(columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalArtifactLayout:
    """Stable local and HDFS locations for full-canonical artifacts."""

    local_dataset: str = "data/curated/full_canonical.parquet"
    local_manifest: str = "data/curated/full_canonical.manifest.json"
    local_audit: str = "data/curated/full_canonical.audit.json"
    hdfs_bronze: str = "/bigdb/bronze"
    hdfs_silver: str = "/bigdb/silver/full_canonical"
    hdfs_gold: str = "/bigdb/gold/full_canonical"


CANONICAL_ARTIFACT_LAYOUT: Final = CanonicalArtifactLayout()


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
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_sha256: str = Field(default=DEFAULT_DIGEST, pattern=r"^[0-9a-f]{64}$")
    asset: str = "verified-asset"
    staging_sha256: str = Field(default=DEFAULT_DIGEST, pattern=r"^[0-9a-f]{64}$")
    source_split: str = "train"
    role: CanonicalRole = CanonicalRole.RESEARCH_INDEX
    acquisition_mode: Literal["sampled", "full"] = "full"
    expected_rows: int | None = Field(default=None, ge=0)
    schema_sha256: str = Field(default=DEFAULT_DIGEST, pattern=r"^[0-9a-f]{64}$")
    scanned: int = 0
    accepted: int = Field(ge=0)
    excluded: int = Field(ge=0)
    raw_bytes: int = Field(default=0, ge=0)
    exclusions: tuple[ExclusionCount, ...] = ()


class BlockedSourceInventory(BaseModel):
    """Safe inventory entry for an approved asset without a verified pin."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: SourceName
    source_split: str
    role: CanonicalRole
    asset: str
    reason_code: str
    expected_rows: int | None = Field(default=None, ge=0)


class ExpectedSource(BaseModel):
    """Trusted source identity and receipt facts supplied outside the manifest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    source: SourceName
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    staging_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_bytes: int = Field(ge=0)
    source_split: str
    role: CanonicalRole
    scanned: int = Field(ge=0)
    accepted: int = Field(ge=0)
    excluded: int = Field(ge=0)


class ExpectedSourceContract(BaseModel):
    """Independent typed source contract used to audit receipt provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[ExpectedSource, ...]

    @classmethod
    def from_receipts(
        cls, receipts: tuple[SourceReceipt, ...]
    ) -> ExpectedSourceContract:
        """Freeze trusted build-input receipt facts before publication."""
        return cls(
            sources=tuple(
                ExpectedSource(
                    source=receipt.source,
                    input_sha256=receipt.input_sha256,
                    asset_sha256=receipt.asset_sha256,
                    staging_sha256=receipt.staging_sha256,
                    schema_sha256=receipt.schema_sha256,
                    raw_bytes=receipt.raw_bytes,
                    source_split=receipt.source_split,
                    role=receipt.role,
                    scanned=receipt.scanned,
                    accepted=receipt.accepted,
                    excluded=receipt.excluded,
                )
                for receipt in receipts
            )
        )


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


class CanonicalRoleCount(BaseModel):
    """Safe count for one canonical source role."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    role: CanonicalRole
    count: int = Field(ge=0)


class CanonicalCount(BaseModel):
    """Aggregate count for one named canonical dimension value."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    value: str
    count: int = Field(ge=0)


class CanonicalIntersection(BaseModel):
    """Aggregate group intersection between two canonical dimensions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    left: str
    right: str
    count: int = Field(ge=0)


class CanonicalArtifact(BaseModel):
    """Digest and physical format metadata without canonical row text."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    path: str
    format: Literal["parquet"] = "parquet"
    compression: Literal["zstd"] = "zstd"
    row_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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


class CanonicalManifest(BaseModel):
    """Safe manifest for the cap-free canonical artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = CANONICAL_MANIFEST_SCHEMA_VERSION
    profile: Literal["research", "production"] = "research"
    seed: int
    dataset_file: str = CANONICAL_ARTIFACT_LAYOUT.local_dataset
    manifest_file: str = CANONICAL_ARTIFACT_LAYOUT.local_manifest
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(ge=0)
    role_counts: tuple[CanonicalRoleCount, ...]
    source_receipts: tuple[SourceReceipt, ...]
    artifact: CanonicalArtifact | None = None
    exact_duplicates_removed: int = Field(default=0, ge=0)
    normalized_collision_count: int = Field(default=0, ge=0)
    group_collision_count: int = Field(default=0, ge=0)
    near_duplicate_cluster_count: int = Field(default=0, ge=0)
    near_covered_record_count: int = Field(default=0, ge=0)
    near_candidate_comparison_count: int = Field(default=0, ge=0)
    source_counts: tuple[CanonicalCount, ...] = ()
    language_counts: tuple[CanonicalCount, ...] = ()
    label_counts: tuple[CanonicalCount, ...] = ()
    split_counts: tuple[CanonicalCount, ...] = ()
    group_role_intersections: tuple[CanonicalIntersection, ...] = ()
    group_split_intersections: tuple[CanonicalIntersection, ...] = ()
    schema_fingerprint: str = ""
    file_size_bytes: int = Field(default=0, ge=0)
    row_group_count: int = Field(default=0, ge=0)
    row_group_rows: tuple[int, ...] = ()
    compression: tuple[str, ...] = ()
    raw_bytes: int | None = Field(default=None, ge=0)
    canonical_size_bytes: int = Field(default=0, ge=0)
    missing_sources: tuple[str, ...] = ()
    blocked_inventory: tuple[BlockedSourceInventory, ...] = ()

    @model_validator(mode="after")
    def validate_artifact_metadata(self) -> CanonicalManifest:
        """Keep the canonical manifest bound to its declared artifact layout."""
        if self.dataset_file != CANONICAL_ARTIFACT_LAYOUT.local_dataset:
            reason = "canonical_dataset_path"
            raise ValueError(reason)
        if self.manifest_file != CANONICAL_ARTIFACT_LAYOUT.local_manifest:
            reason = "canonical_manifest_path"
            raise ValueError(reason)
        return self
