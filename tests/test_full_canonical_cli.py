from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from bigdb_mvp.data.canonical_audit import FullCanonicalAuditReport
from bigdb_mvp.data.canonical_publication import CanonicalPointer
from bigdb_mvp.data.full_canonical_cli import (
    FullBuildReport,
    FullCommandFailure,
    TrustedContractArtifact,
)

CONFIGURED_BUILD_MEASURED_SECONDS = 268
CONFIGURED_BUILD_TIMEOUT_SECONDS = CONFIGURED_BUILD_MEASURED_SECONDS + 92


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        (str(Path(sys.executable).with_name("bigdb-mvp")), *args),
        check=False,
        capture_output=True,
        text=True,
        timeout=CONFIGURED_BUILD_TIMEOUT_SECONDS,
    )


def test_data_help_exposes_distinct_full_commands() -> None:
    result = _cli("data", "--help")

    assert result.returncode == 0
    assert "full-build" in result.stdout
    assert "full-audit" in result.stdout


def test_fixture_full_build_and_audit_are_json_and_reproducible(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "canonical"

    build = _cli(
        "data",
        "full-build",
        "--fixture",
        "--seed",
        "17",
        "--output-root",
        str(output_root),
        "--json",
    )
    audit = _cli(
        "data",
        "full-audit",
        "--output-root",
        str(output_root),
        "--json",
    )

    assert build.returncode == 0
    assert audit.returncode == 0
    built = FullBuildReport.model_validate_json(build.stdout)
    audited = FullCanonicalAuditReport.model_validate_json(audit.stdout)
    assert built.record_count == 8
    assert audited.valid is True
    assert audited.record_count == 8
    assert "fixture canonical row" not in build.stdout
    assert "fixture canonical row" not in audit.stdout
    assert (output_root / "full_canonical.current.json").is_file()


def test_full_audit_fails_closed_for_corrupt_artifact(tmp_path: Path) -> None:
    output_root = tmp_path / "canonical"
    built = _cli(
        "data",
        "full-build",
        "--fixture",
        "--output-root",
        str(output_root),
        "--json",
    )
    assert built.returncode == 0
    pointer = CanonicalPointer.model_validate_json(
        (output_root / "full_canonical.current.json").read_text(encoding="utf-8")
    )
    dataset = (
        output_root
        / "full_canonical.generations"
        / pointer.generation
        / "full_canonical.parquet"
    )
    _ = dataset.write_bytes(dataset.read_bytes() + b"corrupt")

    audit = _cli(
        "data",
        "full-audit",
        "--output-root",
        str(output_root),
        "--json",
    )

    assert audit.returncode != 0
    assert FullCanonicalAuditReport.model_validate_json(audit.stdout).valid is False
    assert "dataset_digest_mismatch" in audit.stdout
    assert "corrupt" not in audit.stdout


def test_full_build_configured_sources_publishes_complete_generation(
    tmp_path: Path,
) -> None:
    result = _cli(
        "data",
        "full-build",
        "--output-root",
        str(tmp_path / "canonical"),
        "--json",
    )

    assert result.returncode == 0
    payload = FullBuildReport.model_validate_json(result.stdout)
    assert payload.valid is True
    assert payload.record_count == 1_204_676
    civil = next(
        receipt
        for receipt in payload.source_receipts
        if receipt.source.value == "civil_comments"
    )
    assert civil.scanned == 902_437
    assert civil.accepted == 902_437
    assert civil.excluded == 0
    assert (tmp_path / "canonical" / "full_canonical.current.json").exists()


def test_full_audit_rejects_stale_trusted_contract(tmp_path: Path) -> None:
    output_root = tmp_path / "canonical"
    built = _cli(
        "data",
        "full-build",
        "--fixture",
        "--output-root",
        str(output_root),
        "--json",
    )
    assert built.returncode == 0
    contract_path = output_root / "full_canonical.trusted-contract.json"
    contract = TrustedContractArtifact.model_validate_json(contract_path.read_bytes())
    _ = contract_path.write_text(
        contract.model_copy(update={"generation": "f" * 64}).model_dump_json(),
        encoding="utf-8",
    )

    audit = _cli(
        "data",
        "full-audit",
        "--output-root",
        str(output_root),
        "--json",
    )

    assert audit.returncode == 1
    report = FullCanonicalAuditReport.model_validate_json(audit.stdout)
    assert report.reason_codes == ("trusted_source_contract_generation_mismatch",)


def test_full_audit_rejects_rebound_trusted_contract_digest(tmp_path: Path) -> None:
    # Given: a valid publication whose contract bytes are changed in place.
    output_root = tmp_path / "canonical"
    built = _cli(
        "data", "full-build", "--fixture", "--output-root", str(output_root), "--json"
    )
    assert built.returncode == 0
    contract_path = output_root / "full_canonical.trusted-contract.json"
    _ = contract_path.write_bytes(contract_path.read_bytes() + b"\n")

    # When: the independent full audit reloads the publication.
    audit = _cli("data", "full-audit", "--output-root", str(output_root), "--json")

    # Then: pointer-to-contract binding rejects the rebound bytes.
    assert audit.returncode == 1
    report = FullCanonicalAuditReport.model_validate_json(audit.stdout)
    assert report.reason_codes == ("trusted_contract_digest_mismatch",)


def test_full_build_output_failure_is_structured(tmp_path: Path) -> None:
    output_file = tmp_path / "not-a-directory"
    _ = output_file.write_text("occupied", encoding="utf-8")

    result = _cli(
        "data",
        "full-build",
        "--fixture",
        "--output-root",
        str(output_file),
        "--json",
    )

    assert result.returncode == 2
    failure = FullCommandFailure.model_validate_json(result.stdout)
    assert failure.reason_code == "output_write_failed"
    assert "Traceback" not in result.stdout
