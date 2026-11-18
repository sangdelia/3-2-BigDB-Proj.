from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl

from bigdb_mvp.data.audit import audit_dataset
from bigdb_mvp.data.canonical_audit import audit_full_canonical
from bigdb_mvp.data.canonical_builder import CanonicalInput, build_full_canonical
from bigdb_mvp.data.canonical_publication import CanonicalPointer
from bigdb_mvp.data.canonical_report import parquet_facts
from bigdb_mvp.data.manifest import (
    CanonicalManifest,
    ExpectedSourceContract,
    SourceReceipt,
)
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    SourceName,
    SupervisedRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build(tmp_path: Path) -> ExpectedSourceContract:
    rows = tuple(
        SupervisedRecord(
            text=f"audit row {index}",
            label=BinaryLabel.CLEAN if index else BinaryLabel.HARMFUL,
            language=Language.EN,
            source=SourceName.CIVIL_COMMENTS,
            source_split="train",
            group_id=f"group-{index}",
            text_sha256=hashlib.sha256(f"audit row {index}".encode()).hexdigest(),
        )
        for index in range(3)
    )
    receipt = SourceReceipt(
        source=SourceName.CIVIL_COMMENTS,
        revision="fixture-1",
        license="test",
        input_sha256="a" * 64,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        scanned=3,
        accepted=3,
        excluded=0,
    )
    _ = build_full_canonical(
        (CanonicalInput(records=rows, receipt=receipt),), output_root=tmp_path, seed=7
    )
    return ExpectedSourceContract.from_receipts((receipt,))


def _rebind_manifest(tmp_path: Path, manifest: CanonicalManifest) -> None:
    pointer_path = tmp_path / "full_canonical.current.json"
    pointer = CanonicalPointer.model_validate_json(pointer_path.read_bytes())
    manifest_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.manifest.json"
    )
    _ = manifest_path.write_text(manifest.model_dump_json())
    _ = pointer_path.write_text(
        pointer.model_copy(
            update={
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
            }
        ).model_dump_json()
    )


def test_full_audit_reports_reconciled_safe_aggregates(tmp_path: Path) -> None:
    expected = _build(tmp_path)
    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is True
    assert report.reason_codes == ()
    assert report.record_count == 3
    assert sum(item.count for item in report.role_counts) == 3
    assert sum(item.count for item in report.source_counts) == 3
    assert sum(item.count for item in report.language_counts) == 3
    assert sum(item.count for item in report.label_counts) == 3
    assert report.row_group_count >= 1
    assert report.compression == ("ZSTD",)
    assert "audit row" not in report.model_dump_json()


def test_full_audit_rejects_dataset_digest_corruption(tmp_path: Path) -> None:
    expected = _build(tmp_path)
    pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    dataset = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.parquet"
    )
    _ = dataset.write_bytes(dataset.read_bytes() + b"tamper")

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert report.reason_codes == ("dataset_digest_mismatch",)


def test_full_audit_rejects_manifest_and_pointer_gate_corruption(
    tmp_path: Path,
) -> None:
    expected = _build(tmp_path)
    pointer_path = tmp_path / "full_canonical.current.json"
    pointer = CanonicalPointer.model_validate_json(pointer_path.read_bytes())
    _ = pointer_path.write_text(
        pointer.model_copy(update={"generation": "f" * 64}).model_dump_json()
    )

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert report.reason_codes == ("current_pointer_mismatch",)


def test_full_audit_rejects_row_count_schema_role_receipt_and_leakage_drift(
    tmp_path: Path,
) -> None:
    expected = _build(tmp_path)
    pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    manifest_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.manifest.json"
    )
    manifest = CanonicalManifest.model_validate_json(manifest_path.read_bytes())
    roles = (manifest.role_counts[0].model_copy(update={"count": 4}),)
    receipts = (manifest.source_receipts[0].model_copy(update={"accepted": 4}),)
    _ = manifest_path.write_text(
        manifest.model_copy(
            update={
                "record_count": 4,
                "role_counts": roles,
                "source_receipts": receipts,
            }
        ).model_dump_json()
    )

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert report.reason_codes == ("manifest_digest_mismatch",)


def test_rebound_receipt_digest_fails_against_trusted_contract(tmp_path: Path) -> None:
    expected = _build(tmp_path)
    pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    manifest_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.manifest.json"
    )
    manifest = CanonicalManifest.model_validate_json(manifest_path.read_bytes())
    receipt = manifest.source_receipts[0].model_copy(update={"input_sha256": "b" * 64})
    _rebind_manifest(
        tmp_path, manifest.model_copy(update={"source_receipts": (receipt,)})
    )

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert report.reason_codes == ("source_provenance_digest_mismatch",)


def test_rebound_receipt_source_fails_against_trusted_identity(tmp_path: Path) -> None:
    expected = _build(tmp_path)
    pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    manifest_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.manifest.json"
    )
    manifest = CanonicalManifest.model_validate_json(manifest_path.read_bytes())
    receipt = manifest.source_receipts[0].model_copy(update={"source": SourceName.NIKL})
    _rebind_manifest(
        tmp_path, manifest.model_copy(update={"source_receipts": (receipt,)})
    )

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert report.reason_codes == ("source_provenance_source_mismatch",)


def test_rebound_snappy_artifact_fails_compression_policy(tmp_path: Path) -> None:
    expected = _build(tmp_path)
    pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    dataset_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.parquet"
    )
    frame = pl.read_parquet(dataset_path)
    _ = frame.write_parquet(dataset_path, compression="snappy")
    facts = parquet_facts(dataset_path)
    manifest_path = (
        tmp_path
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.manifest.json"
    )
    manifest = CanonicalManifest.model_validate_json(manifest_path.read_bytes())
    rebound = manifest.model_copy(
        update={
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "file_size_bytes": facts.file_size_bytes,
            "row_group_rows": facts.row_group_rows,
            "compression": facts.compression,
            "canonical_size_bytes": facts.file_size_bytes,
            "artifact": manifest.artifact.model_copy(
                update={"sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest()}
            )
            if manifest.artifact is not None
            else None,
        }
    )
    _rebind_manifest(tmp_path, rebound)
    rebound_pointer = CanonicalPointer.model_validate_json(
        (tmp_path / "full_canonical.current.json").read_bytes()
    )
    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    _ = (tmp_path / "full_canonical.current.json").write_text(
        rebound_pointer.model_copy(
            update={"dataset_sha256": dataset_digest}
        ).model_dump_json()
    )

    report = audit_full_canonical(tmp_path, expected_sources=expected)
    assert report.valid is False
    assert "compression_policy_violation" in report.reason_codes


def test_existing_sampled_audit_remains_importable() -> None:
    assert callable(audit_dataset)
