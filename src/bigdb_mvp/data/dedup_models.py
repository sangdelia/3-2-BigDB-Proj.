"""Typed safe results and errors for canonical duplicate audits."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class PublicationLeakageError(Exception):
    """Report a digest-safe cross-role collision that blocks publication."""

    reason_codes: tuple[str, ...]

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        """Store stable publication-blocking reason codes."""
        self.reason_codes = reason_codes
        super().__init__("publication_leakage:" + ",".join(reason_codes))


class AuditInputError(Exception):
    """Report a stale or malformed canonical row at the audit boundary."""


class CollisionEvidence(BaseModel):
    """One digest-only collision cluster with one representative."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    key_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_ids: tuple[str, ...]


class NearDuplicateCluster(BaseModel):
    """One bounded near-duplicate cluster containing digest IDs only."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    cluster_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_ids: tuple[str, ...]


class DuplicateAuditReport(BaseModel):
    """Safe aggregate audit result used as the publication gate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    exact_duplicates_removed: int = Field(ge=0)
    exact_role_collision_count: int = Field(ge=0)
    normalized_collision_count: int = Field(ge=0)
    group_collision_count: int = Field(ge=0)
    near_duplicate_clusters: tuple[NearDuplicateCluster, ...]
    normalized_collisions: tuple[CollisionEvidence, ...]
    group_collisions: tuple[CollisionEvidence, ...]
    near_audited_records: int = Field(ge=0)
    near_skipped_records: int = Field(ge=0)
    near_candidate_comparisons: int = Field(ge=0)
    blocking_reasons: tuple[str, ...]
    publication_blocked: bool
