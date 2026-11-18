"""Digest-safe duplicate and role leakage audits for canonical records."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from bigdb_mvp.data.dedup_models import (
    AuditInputError,
    CollisionEvidence,
    DuplicateAuditReport,
    NearDuplicateCluster,
    PublicationLeakageError,
)
from bigdb_mvp.data.models import CanonicalRecord, CanonicalRole

__all__ = (
    "AuditInputError",
    "DuplicateAuditReport",
    "PublicationLeakageError",
    "audit_canonical_duplicates",
    "deduplicate_canonical_exact",
    "normalized_text",
)

if TYPE_CHECKING:
    from collections.abc import Callable

    _Key = Callable[[CanonicalRecord], str]

QUERY_ROLES: Final = frozenset(
    {CanonicalRole.CALIBRATION_QUERY, CanonicalRole.TEST_QUERY}
)
MIN_CLUSTER_MEMBERS: Final = 2
BLOCK_NEIGHBOR_WINDOW: Final = 8


@dataclass(frozen=True, slots=True)
class _NearRow:
    row: CanonicalRecord
    grams: frozenset[str]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized_text(text: str) -> str:
    """Return the collision key after NFKC, case, and whitespace normalization."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def deduplicate_canonical_exact(
    records: tuple[CanonicalRecord, ...],
) -> tuple[tuple[CanonicalRecord, ...], int]:
    """Keep one stable canonical representative for each exact text digest."""
    unique: dict[str, CanonicalRecord] = {}
    for record in sorted(records, key=CanonicalRecord.stable_sort_key):
        _ = unique.setdefault(record.text_sha256, record)
    ordered = tuple(unique.values())
    return ordered, len(records) - len(ordered)


def _validate_rows(records: tuple[CanonicalRecord, ...]) -> None:
    for record in records:
        if _digest(record.text) != record.text_sha256:
            reason = "text_digest_stale"
            raise AuditInputError(reason)


def _boundary_evidence(
    records: tuple[CanonicalRecord, ...],
    key: _Key,
) -> tuple[CollisionEvidence, ...]:
    grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    evidence: list[CollisionEvidence] = []
    for collision_key, members in sorted(grouped.items()):
        sources = {member.source for member in members}
        roles = {member.role for member in members}
        if len(sources) == 1 and len(roles) == 1:
            continue
        member_ids = tuple(sorted(member.text_sha256 for member in members))
        evidence.append(
            CollisionEvidence(
                key_id=_digest(collision_key),
                representative_id=member_ids[0],
                member_ids=member_ids,
            )
        )
    return tuple(evidence)


def _research_query_evidence(
    records: tuple[CanonicalRecord, ...],
    key: _Key,
) -> tuple[CollisionEvidence, ...]:
    evidence = _boundary_evidence(records, key)
    research_ids = {
        row.text_sha256 for row in records if row.role is CanonicalRole.RESEARCH_INDEX
    }
    query_ids = {row.text_sha256 for row in records if row.role in QUERY_ROLES}
    return tuple(
        item
        for item in evidence
        if set(item.member_ids) & research_ids and set(item.member_ids) & query_ids
    )


def _grams(text: str, size: int) -> frozenset[str]:
    normalized = normalized_text(text)
    if len(normalized) <= size:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    )


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    root_left, root_right = _find(parents, left), _find(parents, right)
    if root_left != root_right:
        parents[max(root_left, root_right)] = min(root_left, root_right)


def _near_pairs(
    near_rows: tuple[_NearRow, ...], limit: int
) -> tuple[tuple[int, int], ...]:
    blocks: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(near_rows):
        for gram in item.grams:
            blocks[gram].append(index)
    pairs: set[tuple[int, int]] = set()
    for block in sorted(blocks):
        indexes = sorted(
            blocks[block], key=lambda index: near_rows[index].row.text_sha256
        )
        for left_offset, left in enumerate(indexes):
            for right in indexes[
                left_offset + 1 : left_offset + 1 + BLOCK_NEIGHBOR_WINDOW
            ]:
                left_role = near_rows[left].row.role
                right_role = near_rows[right].row.role
                research_query_pair = (
                    left_role is CanonicalRole.RESEARCH_INDEX
                    and right_role in QUERY_ROLES
                ) or (
                    right_role is CanonicalRole.RESEARCH_INDEX
                    and left_role in QUERY_ROLES
                )
                if not research_query_pair:
                    continue
                pairs.add((left, right) if left < right else (right, left))
        if len(pairs) > limit:
            pairs = set(
                sorted(
                    pairs,
                    key=lambda pair: tuple(
                        sorted(
                            (
                                near_rows[pair[0]].row.text_sha256,
                                near_rows[pair[1]].row.text_sha256,
                            )
                        )
                    ),
                )[:limit]
            )
    ordered = sorted(
        pairs,
        key=lambda pair: tuple(
            sorted(
                (near_rows[pair[0]].row.text_sha256, near_rows[pair[1]].row.text_sha256)
            )
        ),
    )
    return tuple(ordered[:limit])


def _near_reports(
    near_rows: tuple[_NearRow, ...],
    pairs: set[tuple[int, int]],
    threshold: float,
) -> tuple[NearDuplicateCluster, ...]:
    parents = list(range(len(near_rows)))
    for left, right in sorted(pairs):
        left_grams, right_grams = near_rows[left].grams, near_rows[right].grams
        similarity = len(left_grams & right_grams) / len(left_grams | right_grams)
        if similarity >= threshold:
            _union(parents, left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, _item in enumerate(near_rows):
        groups[_find(parents, index)].append(index)
    clusters: list[NearDuplicateCluster] = []
    for member_indexes in groups.values():
        if len(member_indexes) < MIN_CLUSTER_MEMBERS:
            continue
        member_ids = tuple(
            sorted({near_rows[index].row.text_sha256 for index in member_indexes})
        )
        if len(member_ids) < MIN_CLUSTER_MEMBERS:
            continue
        representative_id = member_ids[0]
        cluster_id = _digest("near:" + ":".join(member_ids))
        clusters.append(
            NearDuplicateCluster(
                cluster_id=cluster_id,
                evidence_id=cluster_id,
                representative_id=representative_id,
                member_ids=member_ids,
            )
        )
    return tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))


def _near_clusters(
    records: tuple[CanonicalRecord, ...],
    *,
    size: int,
    threshold: float,
    limit: int,
) -> tuple[tuple[NearDuplicateCluster, ...], int, int, int]:
    selected = tuple(sorted(records, key=CanonicalRecord.stable_sort_key))
    near_rows = tuple(_NearRow(row, _grams(row.text, size)) for row in selected)
    pairs = _near_pairs(near_rows, limit)
    clusters = _near_reports(near_rows, set(pairs), threshold)
    return clusters, len(selected), 0, len(pairs)


def audit_canonical_duplicates(
    records: tuple[CanonicalRecord, ...],
    *,
    near_ngram_size: int = 5,
    near_threshold: float = 0.85,
    max_near_records: int = 10_000,
) -> DuplicateAuditReport:
    """Audit exact, normalized, group, and bounded research/query near collisions."""
    if near_ngram_size < 1 or max_near_records < 1 or not 0 < near_threshold <= 1:
        reason = "near_audit_bounds_invalid"
        raise AuditInputError(reason)
    _validate_rows(records)
    _unique, exact_removed = deduplicate_canonical_exact(records)
    exact = _boundary_evidence(records, lambda row: row.text_sha256)
    normalized = _boundary_evidence(records, lambda row: normalized_text(row.text))
    groups = _boundary_evidence(records, lambda row: row.group_id)
    exact_leakage = _research_query_evidence(records, lambda row: row.text_sha256)
    normalized_leakage = _research_query_evidence(
        records, lambda row: normalized_text(row.text)
    )
    group_leakage = _research_query_evidence(records, lambda row: row.group_id)
    near, audited, skipped, comparisons = _near_clusters(
        records, size=near_ngram_size, threshold=near_threshold, limit=max_near_records
    )
    reasons = tuple(
        reason
        for reason, evidence in (
            ("exact_role_leakage", exact_leakage),
            ("normalized_role_leakage", normalized_leakage),
            ("group_role_leakage", group_leakage),
        )
        if evidence
    )
    return DuplicateAuditReport(
        exact_duplicates_removed=exact_removed,
        exact_role_collision_count=len(exact),
        normalized_collision_count=len(normalized),
        group_collision_count=len(groups),
        near_duplicate_clusters=near,
        normalized_collisions=normalized,
        group_collisions=groups,
        near_audited_records=audited,
        near_skipped_records=skipped,
        near_candidate_comparisons=comparisons,
        blocking_reasons=reasons,
        publication_blocked=bool(reasons),
    )
