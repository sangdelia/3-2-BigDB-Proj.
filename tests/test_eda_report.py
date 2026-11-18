"""Check EDA semantics independently of presentation wording."""

from pathlib import Path

import polars as pl
import pytest

from bigdb_mvp.experiments.dataset_types import DatasetError
from bigdb_mvp.experiments.eda_contracts import CorpusManifest
from bigdb_mvp.experiments.eda_profile import (
    length_profile,
    missing_profile,
    retention_methods,
    validate_corpus,
)
from bigdb_mvp.experiments.eda_report import build_report


@pytest.fixture
def corpus() -> pl.DataFrame:
    """Synthetic original strings deliberately differ in bytes and code points."""
    return pl.DataFrame(
        {
            "text": ["가🙂", "둘  셋", "문장"],
            "text_sha256": ["a", "b", "c"],
            "group_id": ["g1", "g2", "g3"],
            "split": ["train", "validation", "test"],
            "source": ["beep", "kold", "beep"],
            "label": ["clean", "harmful", "harmful"],
            "context_text": ["기사 제목", "", "다른 제목"],
            "target_labels": [["beep:bias:none"], [], ["beep:bias:others"]],
            "expression_labels": [["beep:none"], [], ["beep:offensive"]],
        }
    )


@pytest.fixture
def manifest() -> CorpusManifest:
    return CorpusManifest.model_validate(
        {
            "dataset_sha256": "0" * 64,
            "raw_rows": 3,
            "parsed_rows": 3,
            "accepted_rows": 3,
            "sources": [
                {"source": "beep", "raw_rows": 2, "accepted_rows": 2},
                {"source": "kold", "raw_rows": 1, "accepted_rows": 1},
            ],
            "dedup_exclusions": {
                "conflicting_binary_rows": 0,
                "consistent_duplicate_rows": 0,
            },
        }
    )


def test_lengths_count_original_unicode_codepoints(corpus: pl.DataFrame) -> None:
    # Given: original whitespace and emoji are preserved in the fixture.
    # When: lengths are profiled.
    profile = length_profile(corpus)
    # Then: UTF-8 byte counts and collapsed whitespace would give different values.
    assert (profile.stats.min, profile.stats.max, profile.stats.mean) == (2, 4, 8 / 3)


def test_optional_annotations_remain_distinct_from_required_values(
    corpus: pl.DataFrame,
) -> None:
    # Given: one source legitimately supplies no optional annotations.
    # When: missing values are measured.
    profile = missing_profile(corpus)
    # Then: only optional fields have an unavailable row.
    assert {item.field: item.missing_rows for item in profile} == {
        "text": 0,
        "label": 0,
        "source": 0,
        "context_text": 1,
        "target_labels": 1,
        "expression_labels": 1,
    }


def test_complete_case_comparison_uses_same_final_denominator(
    corpus: pl.DataFrame,
) -> None:
    # Given: two rows have all optional fields and a third lacks them.
    # When: retention policies are compared.
    selected, comparison = retention_methods(corpus)
    # Then: the comparison quantifies source loss instead of filling labels.
    assert (
        selected.retained_rows,
        comparison.retained_rows,
        comparison.removed_rows,
        comparison.start_rows,
    ) == (3, 2, 1, 3)


def test_cross_split_group_is_rejected(
    corpus: pl.DataFrame,
    manifest: CorpusManifest,
) -> None:
    # Given: distinct strings in different splits share one document group.
    overlapping = corpus.with_columns(pl.lit("shared").alias("group_id"))
    # When / Then: the reporting boundary rejects the leaking artifact.
    with pytest.raises(DatasetError, match="eda_split_overlap"):
        validate_corpus(overlapping, manifest)


def test_changed_dataset_is_rejected_before_profiling(
    tmp_path: Path,
    manifest: CorpusManifest,
) -> None:
    # Given: a manifest whose checksum disagrees with the actual artifact.
    _ = (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    _ = (tmp_path / "corpus.parquet").write_bytes(b"changed")
    # When / Then: no aggregate report is published from the changed input.
    with pytest.raises(DatasetError, match="eda_dataset_checksum_mismatch"):
        _ = build_report(tmp_path, tmp_path)
