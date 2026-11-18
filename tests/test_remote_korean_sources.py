from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from bigdb_mvp.data.civil_download import DownloadSpec
from bigdb_mvp.data.models import BinaryLabel, CanonicalRole, SourceName
from bigdb_mvp.data.remote_integrity import RemoteSourceError
from bigdb_mvp.data.remote_registry import RemoteSourceSpec
from bigdb_mvp.data.remote_sources import ingest_remote_source
from bigdb_mvp.data.staging import write_staged

if TYPE_CHECKING:
    from pathlib import Path


def _asset(path: Path, payload: bytes) -> DownloadSpec:
    _ = path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _ = path.with_name(path.name + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    return DownloadSpec(
        url=f"https://raw.githubusercontent.com/a/{path.name}",
        target=path,
        expected_sha256=digest,
        expected_bytes=len(payload),
    )


def _spec(source: SourceName, *assets: DownloadSpec) -> RemoteSourceSpec:
    return RemoteSourceSpec(
        source=source,
        revision="a" * 40,
        license="test-license",
        repository="https://invalid.example/repository",
        assets=assets,
    )


def test_remote_ingest_rejects_schema_drift_before_normalization(
    tmp_path: Path,
) -> None:
    # Given: a K-MHaS asset whose pinned header schema is wrong.
    payload = b"document\tlabel\nplain\t8\n"
    asset = _asset(tmp_path / "schema-drift.txt", payload)
    spec = RemoteSourceSpec(
        source=SourceName.KMHAS,
        revision="a" * 40,
        license="test-license",
        repository="https://invalid.example/repository",
        assets=(asset,),
        expected_rows=1,
        expected_schema=("not", "checked"),
    )

    # When / Then: the pinned remote schema fails closed at the source boundary.
    with pytest.raises(RemoteSourceError, match="remote_source_schema_drift"):
        _ = ingest_remote_source(spec, tmp_path / "staging")


def test_stale_remote_receipt_rebuilds_from_pinned_assets(tmp_path: Path) -> None:
    # Given: valid pinned assets and a digest-valid stage with stale metadata.
    payload = b"document\tlabel\nplain\t8\n"
    asset = _asset(tmp_path / "train.txt", payload)
    spec = RemoteSourceSpec(
        source=SourceName.KMHAS,
        revision="a" * 40,
        license="test-license",
        repository="https://invalid.example/repository",
        assets=(asset,),
        expected_rows=1,
        expected_schema=("document", "label"),
    )
    staging_root = tmp_path / "staging"
    records, receipt = ingest_remote_source(spec, staging_root)
    stale = receipt.model_copy(update={"revision": "stale-revision"})
    _ = write_staged(staging_root, stale, records)

    # When: the adapter sees stale role-aware metadata.
    rebuilt_records, rebuilt = ingest_remote_source(spec, staging_root)

    # Then: it revalidates and rebuilds from the pinned asset.
    assert rebuilt_records == records
    assert rebuilt.revision == spec.revision
    assert rebuilt.role is CanonicalRole.RESEARCH_INDEX


def test_kmhas_full_train_maps_multilabel_rows_fail_closed(tmp_path: Path) -> None:
    # Given: documented clean, harmful, conflicting, and unknown K-MHaS labels.
    payload = (
        b"document\tlabel\n"
        b"clean text\t8\n"
        b"harmful text\t1,3\n"
        b"conflict text\t1,8\n"
        b"unknown text\t9\n"
    )
    spec = _spec(SourceName.KMHAS, _asset(tmp_path / "train.txt", payload))

    # When: the pinned official train file is normalized.
    records, receipt = ingest_remote_source(spec, tmp_path / "staging")

    # Then: only the documented unambiguous projection is accepted.
    assert tuple(row.label for row in records) == (
        BinaryLabel.CLEAN,
        BinaryLabel.HARMFUL,
    )
    assert {item.reason: item.count for item in receipt.exclusions} == {
        "kmhas_label_conflict": 1,
        "kmhas_label_unknown": 1,
    }


def test_beep_full_train_groups_comments_by_aligned_article(tmp_path: Path) -> None:
    # Given: two BEEP comments aligned to the same source article title.
    comments = (
        b"comments\tcontain_gender_bias\tbias\thate\n"
        b"clean text\tFalse\tnone\tnone\n"
        b"harmful text\tFalse\tnone\thate\n"
    )
    titles = b"same article\nsame article\n"
    spec = _spec(
        SourceName.BEEP,
        _asset(tmp_path / "train.tsv", comments),
        _asset(tmp_path / "train.news_title.txt", titles),
    )

    # When: the paired immutable assets are normalized.
    records, receipt = ingest_remote_source(spec, tmp_path / "staging")

    # Then: binary labels map correctly and article groups cannot cross splits.
    assert tuple(row.label for row in records) == (
        BinaryLabel.CLEAN,
        BinaryLabel.HARMFUL,
    )
    assert records[0].group_id == records[1].group_id
    assert receipt.scanned == 2


def test_khaters_full_train_uses_only_original_four_label_vocabulary(
    tmp_path: Path,
) -> None:
    # Given: all documented labels and one unknown K-HATERS label.
    payload = b"".join(
        (
            b'{"text":"clean","label":"normal"}\n',
            b'{"text":"offensive","label":"offensive"}\n',
            b'{"text":"level one","label":"L1_hate"}\n',
            b'{"text":"level two","label":"L2_hate"}\n',
            b'{"text":"unknown","label":"hate"}\n',
        )
    )
    spec = _spec(SourceName.KHATERS, _asset(tmp_path / "train.jsonl", payload))

    # When: the official JSONL boundary is applied.
    records, receipt = ingest_remote_source(spec, tmp_path / "staging")

    # Then: unknown labels are excluded rather than inferred.
    assert tuple(row.label for row in records) == (
        BinaryLabel.CLEAN,
        BinaryLabel.HARMFUL,
        BinaryLabel.HARMFUL,
        BinaryLabel.HARMFUL,
    )
    assert {item.reason: item.count for item in receipt.exclusions} == {
        "khaters_original_label_unknown": 1
    }
