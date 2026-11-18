from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from bigdb_mvp.data import civil_download, civil_source, local_sources
from bigdb_mvp.data.civil_download import DownloadSpec
from bigdb_mvp.data.civil_parquet import CivilBatch, CivilScanLimits
from bigdb_mvp.data.local_sources import LocalSourceSpec
from bigdb_mvp.data.manifest import SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    SourceName,
    SupervisedRecord,
)
from bigdb_mvp.data.remote_registry import RemoteSourceSpec
from bigdb_mvp.data.remote_sources import ingest_remote_source
from bigdb_mvp.data.staging import StagingIntegrityError, write_staged

if TYPE_CHECKING:
    from pathlib import Path


def test_source_receipt_requires_role_aware_pinned_contract() -> None:
    # Given: an acquisition receipt for one labeled source split.
    receipt = SourceReceipt(
        source=SourceName.KMHAS,
        revision="a" * 40,
        license="CC-BY-SA-4.0",
        input_sha256="b" * 64,
        asset="kmhas-train.txt",
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=1,
        scanned=1,
        accepted=1,
        excluded=0,
    )

    # When: the receipt is serialized at the safe boundary.
    payload = receipt.model_dump()

    # Then: the source split and canonical role remain explicit.
    assert payload["source_split"] == "train"
    assert payload["role"] is CanonicalRole.RESEARCH_INDEX
    assert payload["expected_rows"] == 1


def test_hidden_or_unlabeled_rows_never_become_supervised_records(
    tmp_path: Path,
) -> None:
    # Given: a pinned test asset whose rows have no usable supervised label.
    asset = tmp_path / "hidden.jsonl"
    payload = b'{"text":"hidden label"}\n'
    _ = asset.write_bytes(payload)
    _ = asset.with_name(asset.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    spec = RemoteSourceSpec(
        source=SourceName.KHATERS,
        revision="a" * 40,
        license="test-license",
        repository="https://invalid.example/repository",
        assets=(
            DownloadSpec(
                    url="https://raw.githubusercontent.com/a/hidden.jsonl",
                target=asset,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_bytes=len(payload),
            ),
        ),
        source_split="test",
        role=CanonicalRole.UNLABELED_INVENTORY,
        expected_rows=1,
    )

    # When: the hidden-label asset crosses the adapter boundary.
    records, receipt = ingest_remote_source(spec, tmp_path / "staging")

    # Then: it is accounted for as excluded inventory, never supervised output.
    assert records == ()
    assert receipt.role is CanonicalRole.UNLABELED_INVENTORY
    assert receipt.accepted == 0
    assert receipt.excluded == 1
    assert receipt.exclusions[0].reason == "invalid_schema"


def test_unsupported_role_fails_closed_with_source_scope() -> None:
    # Given: a receipt contract carrying an unknown role value.
    with pytest.raises(ValueError, match="role"):
        _ = RemoteSourceSpec.model_validate(
            {
                "source": SourceName.KMHAS,
                "revision": "a" * 40,
                "license": "test-license",
                "repository": "https://invalid.example/repository",
                "assets": (),
                "role": "not-a-role",
            }
        )


def test_stale_local_staged_receipt_rebuilds_from_pinned_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a staged local receipt whose pinned revision no longer matches.
    monkeypatch.setattr(local_sources, "STAGING_ROOT", tmp_path / "staging")
    source_file = tmp_path / "nikl.json"
    payload = b"[]"
    _ = source_file.write_bytes(payload)
    spec = LocalSourceSpec(
        source=SourceName.NIKL,
        upstream=source_file,
        license="test-license",
        revision="current-revision",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_rows=0,
    )
    row = SupervisedRecord(
        text="synthetic",
        label=BinaryLabel.CLEAN,
        language=Language.KO,
        source=SourceName.NIKL,
        source_split="train",
        group_id="group",
        text_sha256="a" * 64,
    )
    receipt = SourceReceipt(
        source=SourceName.NIKL,
        revision="stale-revision",
        license="test-license",
        input_sha256=spec.expected_sha256,
        asset_sha256=spec.expected_sha256,
        asset=source_file.name,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=0,
        scanned=1,
        accepted=1,
        excluded=0,
    )
    _ = write_staged(tmp_path / "staging", receipt, (row,))

    # When: stale metadata is encountered after records passed the digest check.
    records, rebuilt = local_sources.ingest_local_source(spec)

    # Then: the pinned upstream is re-ingested and publishes current metadata.
    assert records == ()
    assert rebuilt.revision == spec.revision
    assert rebuilt.asset == source_file.name
    assert rebuilt.expected_rows == 0


def test_staged_local_records_tampering_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a stale receipt paired with records whose bytes no longer match it.
    monkeypatch.setattr(local_sources, "STAGING_ROOT", tmp_path / "staging")
    source_file = tmp_path / "nikl.json"
    payload = b"[]"
    _ = source_file.write_bytes(payload)
    spec = LocalSourceSpec(
        source=SourceName.NIKL,
        upstream=source_file,
        license="test-license",
        revision="current-revision",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_rows=0,
    )
    receipt = SourceReceipt(
        source=SourceName.NIKL,
        revision="stale-revision",
        license="test-license",
        input_sha256=spec.expected_sha256,
        asset_sha256=spec.expected_sha256,
        asset=source_file.name,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=0,
        scanned=0,
        accepted=0,
        excluded=0,
    )
    _ = write_staged(tmp_path / "staging", receipt, ())
    records_path = tmp_path / "staging" / "nikl.jsonl"
    _ = records_path.write_bytes(b"tampered\n")

    # When / Then: upstream refresh cannot mask staged-record tampering.
    with pytest.raises(StagingIntegrityError, match="staging_checksum_drift"):
        _ = local_sources.ingest_local_source(spec)


def test_stale_civil_staged_receipt_rebuilds_from_pinned_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a Civil staged receipt with a stale asset digest.
    monkeypatch.setattr(civil_source, "STAGING_ROOT", tmp_path / "staging")
    monkeypatch.setattr(
        civil_source, "SHARD_PATH", tmp_path / "staging" / "civil.parquet"
    )
    monkeypatch.setattr(civil_download, "STAGING_ROOT", tmp_path / "staging")
    row = SupervisedRecord(
        text="synthetic",
        label=BinaryLabel.CLEAN,
        language=Language.EN,
        source=SourceName.CIVIL_COMMENTS,
        source_split="train",
        group_id="group",
        text_sha256="b" * 64,
    )
    receipt = SourceReceipt(
        source=SourceName.CIVIL_COMMENTS,
        revision=civil_source.CIVIL_REVISION,
        license=civil_source.CIVIL_LICENSE,
        input_sha256="0" * 64,
        asset_sha256="0" * 64,
        asset=civil_source.CIVIL_SHARD_URL,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=1,
        scanned=1,
        accepted=1,
        excluded=0,
    )
    _ = write_staged(tmp_path / "staging", receipt, (row,))

    # When: stale pinned metadata is found after the staged digest is valid.
    def fake_download(_spec: DownloadSpec) -> str:
        _ = _spec.target.write_bytes(b"synthetic")
        return "1" * 64

    def fake_scan(_path: Path, _limits: CivilScanLimits) -> CivilBatch:
        return CivilBatch(
            records=(row,),
            clean_count=2500,
            harmful_count=2500,
            exclusions=(),
            scanned=2,
        )

    monkeypatch.setattr(civil_source, "download_immutable_shard", fake_download)
    monkeypatch.setattr(
        civil_source,
        "scan_civil_parquet",
        fake_scan,
    )
    records, rebuilt = civil_source.ingest_civil_comments()

    # Then: the authoritative acquisition path publishes current metadata.
    assert records == (row,)
    assert rebuilt.input_sha256 == "1" * 64
    assert rebuilt.expected_rows == 2
    assert rebuilt.excluded == 1
    assert rebuilt.exclusions[0].reason == "civil_quota_capped"
