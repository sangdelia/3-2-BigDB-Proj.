from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from bigdb_mvp.data.canonical_builder import CanonicalInput, build_full_canonical
from bigdb_mvp.data.dedup_audit import (
    PublicationLeakageError,
    audit_canonical_duplicates,
    deduplicate_canonical_exact,
)
from bigdb_mvp.data.manifest import SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRecord,
    CanonicalRole,
    Language,
    SourceName,
    SupervisedRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    text: str,
    *,
    role: CanonicalRole = CanonicalRole.RESEARCH_INDEX,
    group_id: str = "group-1",
    source: SourceName = SourceName.NIKL,
) -> CanonicalRecord:
    return CanonicalRecord(
        text=text,
        label=BinaryLabel.CLEAN,
        language=Language.EN,
        source=source,
        source_split="train",
        role=role,
        group_id=group_id,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def _input(rows: tuple[SupervisedRecord, ...], role: CanonicalRole) -> CanonicalInput:
    return CanonicalInput(
        records=rows,
        receipt=SourceReceipt(
            source=rows[0].source,
            revision="test",
            license="test",
            input_sha256="a" * 64,
            source_split="train",
            role=role,
            scanned=len(rows),
            accepted=len(rows),
            excluded=0,
        ),
    )


def test_exact_dedup_uses_stable_representative_and_counts_rows() -> None:
    first = _row("same", group_id="z-group", source=SourceName.UNSMILE)
    second = _row("same", group_id="a-group", source=SourceName.NIKL)
    unique, removed = deduplicate_canonical_exact((first, second))
    assert removed == 1
    assert unique == (second,)


def test_normalized_and_group_collisions_cross_roles_block() -> None:
    rows = (
        _row(" Café   TEST ", group_id="article-7"),
        _row("café test", role=CanonicalRole.TEST_QUERY, group_id="article-7"),
    )
    report = audit_canonical_duplicates(rows)
    assert report.publication_blocked is True
    assert report.normalized_collision_count == 1
    assert report.group_collision_count == 1
    assert report.blocking_reasons == (
        "normalized_role_leakage",
        "group_role_leakage",
    )
    assert "café" not in report.model_dump_json()


def test_near_duplicate_clusters_are_bounded_and_digest_only() -> None:
    rows = (
        _row("research sentence with enough shared content", group_id="r"),
        _row(
            "research sentence with enough shared contenx",
            role=CanonicalRole.CALIBRATION_QUERY,
            group_id="q",
        ),
    )
    report = audit_canonical_duplicates(
        rows, near_ngram_size=3, near_threshold=0.8, max_near_records=10
    )
    assert len(report.near_duplicate_clusters) == 1
    cluster = report.near_duplicate_clusters[0]
    assert cluster.representative_id == min(row.text_sha256 for row in rows)
    assert cluster.member_ids == tuple(sorted(row.text_sha256 for row in rows))
    assert cluster.evidence_id == cluster.cluster_id
    assert "research sentence" not in report.model_dump_json()


def test_near_audit_covers_tail_rows_without_unbounded_comparisons() -> None:
    # Given: more than the former audit cap, with the only duplicate at the tail.
    fillers = tuple(
        _row(f"unique filler sentence {index} with stable words", group_id=f"f-{index}")
        for index in range(10_000)
    )
    tail_research = _row(
        "tail sentence with enough shared content", group_id="tail-research"
    )
    tail_query = _row(
        "tail sentence with enough shared contenx",
        role=CanonicalRole.CALIBRATION_QUERY,
        group_id="tail-query",
    )

    # When: the bounded near-duplicate audit runs on the complete input.
    report = audit_canonical_duplicates(
        (*fillers, tail_research, tail_query),
        near_ngram_size=3,
        near_threshold=0.8,
        max_near_records=64,
    )

    # Then: every row is covered, the tail pair is found, and comparisons stay bounded.
    assert report.near_audited_records == 10_002
    assert report.near_skipped_records == 0
    assert report.near_candidate_comparisons <= 64
    assert any(
        set(cluster.member_ids)
        == {tail_research.text_sha256, tail_query.text_sha256}
        for cluster in report.near_duplicate_clusters
    )


def test_near_audit_is_input_order_invariant_with_common_shingles() -> None:
    # Given: adversarial rows that share a very common shingle block.
    rows = tuple(
        _row(
            f"common shingle payload with deterministic suffix {index}",
            role=(
                CanonicalRole.RESEARCH_INDEX
                if index % 2 == 0
                else CanonicalRole.CALIBRATION_QUERY
            ),
            group_id=f"common-{index}",
        )
        for index in range(240)
    )
    shuffled = list(rows[::2] + rows[1::2])

    # When: both orders are audited with the same candidate budget.
    first = audit_canonical_duplicates(rows, near_ngram_size=3, max_near_records=120)
    second = audit_canonical_duplicates(
        tuple(shuffled), near_ngram_size=3, max_near_records=120
    )

    # Then: output is digest-only and stable, with a hard candidate bound.
    assert first.model_dump() == second.model_dump()
    assert first.near_audited_records == len(rows)
    assert first.near_skipped_records == 0
    assert first.near_candidate_comparisons <= 120
    assert "common shingle payload" not in first.model_dump_json()


def test_builder_blocks_cross_role_collision_before_publication(tmp_path: Path) -> None:
    research = SupervisedRecord(
        text="Shared text",
        label=BinaryLabel.CLEAN,
        language=Language.EN,
        source=SourceName.NIKL,
        source_split="train",
        group_id="research",
        text_sha256=hashlib.sha256(b"Shared text").hexdigest(),
    )
    query_text = " shared  text "
    query = research.model_copy(
        update={
            "text": query_text,
            "text_sha256": hashlib.sha256(query_text.encode()).hexdigest(),
        }
    )
    with pytest.raises(PublicationLeakageError, match="normalized_role_leakage"):
        _ = build_full_canonical(
            (
                _input((research,), CanonicalRole.RESEARCH_INDEX),
                _input((query,), CanonicalRole.TEST_QUERY),
            ),
            output_root=tmp_path,
            seed=42,
        )
    assert not (tmp_path / "full_canonical.current.json").exists()
