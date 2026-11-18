from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bigdb_mvp.data.arrow_sources import ArrowSourceError
from bigdb_mvp.data.canonical_audit import (
    FullCanonicalAuditReport,
    audit_full_canonical,
)
from bigdb_mvp.data.canonical_builder import (
    CanonicalInput,
    build_full_canonical,
    ingest_accepted_sources,
)
from bigdb_mvp.data.canonical_publication import (
    CanonicalPointer,
    PublicationIntegrityError,
    TrustedContractArtifact,
    load_published_canonical,
)
from bigdb_mvp.data.civil_source import CivilAcquisitionError, CivilSourceError
from bigdb_mvp.data.dedup_models import PublicationLeakageError
from bigdb_mvp.data.local_sources import LocalSourceError
from bigdb_mvp.data.manifest import ExpectedSourceContract, SourceReceipt
from bigdb_mvp.data.models import (
    BinaryLabel,
    CanonicalRole,
    Language,
    RecordDraft,
    SourceName,
)
from bigdb_mvp.data.normalize import build_record
from bigdb_mvp.data.remote_integrity import RemoteSourceError
from bigdb_mvp.data.staging import StagingIntegrityError

TRUSTED_CONTRACT_NAME = "full_canonical.trusted-contract.json"
__all__ = ("FullBuildReport", "FullCommandFailure", "TrustedContractArtifact")


class FullBuildReport(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    valid: bool = True
    record_count: int = Field(ge=0)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_receipts: tuple[SourceReceipt, ...]
    profile: Literal["research", "production"] = "research"


class FullCommandFailure(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    valid: Literal[False] = False
    reason_code: str
    source: str


class FullBuildError(RuntimeError):
    reason_code: str
    source: str

    def __init__(self, reason_code: str, source: str) -> None:
        self.reason_code = reason_code
        self.source = source
        super().__init__(reason_code)


def failure_report(error: BaseException) -> FullCommandFailure:
    match error:
        case LocalSourceError() | ArrowSourceError() | RemoteSourceError():
            return FullCommandFailure(
                reason_code=error.reason_code,
                source=error.source.value,
            )
        case StagingIntegrityError():
            return FullCommandFailure(
                reason_code=error.reason_code, source=error.source
            )
        case CivilAcquisitionError() | CivilSourceError():
            return FullCommandFailure(
                reason_code=error.reason_code,
                source=SourceName.CIVIL_COMMENTS.value,
            )
        case PublicationIntegrityError() | PublicationLeakageError():
            return FullCommandFailure(reason_code=str(error), source="publication")
        case FullBuildError():
            return FullCommandFailure(
                reason_code=error.reason_code,
                source=error.source,
            )
        case _:
            return FullCommandFailure(
                reason_code="full_build_failed", source="publication"
            )


def _fixture_inputs(seed: int) -> tuple[CanonicalInput, ...]:
    rows = tuple(
        build_record(
            RecordDraft(
                text=f"fixture canonical row {seed}-{index}",
                label=(BinaryLabel.CLEAN if index % 2 == 0 else BinaryLabel.HARMFUL),
                language=Language.EN,
                source=SourceName.BEEP,
                source_split="train",
                preferred_group_id=f"fixture-group-{seed}-{index}",
            )
        )
        for index in range(8)
    )
    digest = hashlib.sha256(f"bigdb-fixture-v1:{seed}".encode()).hexdigest()
    receipt = SourceReceipt(
        source=SourceName.BEEP,
        revision="fixture-v1",
        license="local-only synthetic QA fixture",
        input_sha256=digest,
        asset_sha256=digest,
        asset="synthetic-fixture",
        staging_sha256=digest,
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=len(rows),
        scanned=len(rows),
        accepted=len(rows),
        excluded=0,
    )
    return (CanonicalInput(records=rows, receipt=receipt),)


def build_full_command(
    *,
    output_root: Path,
    seed: int,
    fixture: bool,
    profile: Literal["research", "production"] = "research",
) -> FullBuildReport:
    try:
        inputs = _fixture_inputs(seed) if fixture else ingest_accepted_sources()
        manifest = build_full_canonical(
            inputs, output_root=output_root, seed=seed, profile=profile
        )
        pointer = CanonicalPointer.model_validate_json(
            (output_root / "full_canonical.current.json").read_bytes()
        )
        contract_sha256 = pointer.trusted_contract_sha256
    except FullBuildError:
        raise
    except OSError as error:
        del error
        reason = "output_write_failed"
        raise FullBuildError(reason, "publication") from None
    return FullBuildReport(
        record_count=manifest.record_count,
        dataset_sha256=manifest.dataset_sha256,
        logical_digest=manifest.logical_digest,
        contract_sha256=contract_sha256,
        source_receipts=manifest.source_receipts,
        profile=manifest.profile,
    )


def _load_contract(root: Path) -> ExpectedSourceContract:
    try:
        return load_published_canonical(root).trusted_contract
    except PublicationIntegrityError as error:
        reason = error.reason_code
        if reason == "trusted_contract_generation_mismatch":
            reason = "trusted_source_contract_generation_mismatch"
        raise FullBuildError(reason, "publication") from error
    except (ValidationError, ValueError, OSError) as error:
        reason = "trusted_source_contract_invalid"
        raise FullBuildError(reason, "publication") from error


def audit_full_command(output_root: Path) -> FullCanonicalAuditReport:
    try:
        contract = _load_contract(output_root)
    except FullBuildError as error:
        return FullCanonicalAuditReport(
            valid=False,
            reason_codes=(error.reason_code,),
            record_count=0,
        )
    return audit_full_canonical(output_root, expected_sources=contract)
