"""Atomic generation publication and digest-validated pointer loading."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, NoReturn

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.data.canonical_path_safety import (
    ensure_publication_directories,
    hard_link,
    read_bytes,
    validate_directory,
    validate_file,
)
from bigdb_mvp.data.manifest import CanonicalManifest, ExpectedSourceContract
from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from bigdb_mvp.data.models import CanonicalRecord

DEFAULT_DATASET_NAME: Final[str] = "full_canonical.parquet"
DEFAULT_MANIFEST_NAME: Final[str] = "full_canonical.manifest.json"
DEFAULT_GENERATIONS_DIR: Final[str] = "full_canonical.generations"
DEFAULT_POINTER_NAME: Final[str] = "full_canonical.current.json"
_ReplacePath = Callable[[Path, Path], None]


class PublicationIntegrityError(Exception):
    """Report an unreadable or digest-inconsistent current publication."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize the digest-only integrity reason."""
        self.reason_code = reason_code
        super().__init__(reason_code)


class CanonicalPointer(BaseModel):
    """Typed atomic pointer to one complete canonical generation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    generation: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]+$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    trusted_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrustedContractArtifact(BaseModel):
    """Generation-bound trusted source contract stored beside the pointer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    generation: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract: ExpectedSourceContract


@dataclass(frozen=True, slots=True)
class PublishedCanonical:
    """Digest-validated paths selected by the current publication pointer."""

    manifest: CanonicalManifest
    trusted_contract: ExpectedSourceContract
    dataset_path: Path
    generation_path: Path


def write_canonical_dataset(rows: tuple[CanonicalRecord, ...], path: Path) -> None:
    """Write canonical rows with the stable compressed publication schema."""
    _ = pl.DataFrame(
        {
            "text": [row.text for row in rows],
            "label": [
                row.label.value if row.label is not None else None for row in rows
            ],
            "language": [row.language.value for row in rows],
            "source": [row.source.value for row in rows],
            "source_split": [row.source_split for row in rows],
            "role": [row.role.value for row in rows],
            "group_id": [row.group_id for row in rows],
            "text_sha256": [row.text_sha256 for row in rows],
        }
    ).write_parquet(path, compression="zstd", statistics=True)


def _generation_paths(
    output_root: Path, generation: str
) -> tuple[Path, Path, Path, Path]:
    generation_path = output_root / DEFAULT_GENERATIONS_DIR / generation
    return (
        generation_path,
        generation_path / DEFAULT_DATASET_NAME,
        generation_path / DEFAULT_MANIFEST_NAME,
        generation_path / "full_canonical.trusted-contract.json",
    )


def publish_generation(
    rows: tuple[CanonicalRecord, ...],
    manifest: CanonicalManifest,
    *,
    output_root: Path,
    replace: _ReplacePath,
    trusted_contract: ExpectedSourceContract,
) -> None:
    """Write one immutable generation, then atomically advance its pointer."""
    generation = manifest.logical_digest
    ensure_publication_directories(output_root)
    generations_root = output_root / DEFAULT_GENERATIONS_DIR
    generation_path, _dataset_path, manifest_path, contract_path = _generation_paths(
        output_root, generation
    )
    if not generation_path.exists():
        generation_tmp = generations_root / f".{generation}.tmp"
        validate_directory(generations_root)
        generation_tmp.mkdir()
        write_canonical_dataset(rows, generation_tmp / DEFAULT_DATASET_NAME)
        _ = (generation_tmp / DEFAULT_MANIFEST_NAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        generation_contract = TrustedContractArtifact(
            generation=generation, contract=trusted_contract
        )
        _ = (generation_tmp / contract_path.name).write_text(
            generation_contract.model_dump_json(indent=2), encoding="utf-8"
        )
        replace(generation_tmp, generation_path)
    else:
        validate_directory(generation_path)
        manifest_path = generation_path / DEFAULT_MANIFEST_NAME
        manifest_tmp = generation_path / f".{DEFAULT_MANIFEST_NAME}.tmp"
        try:
            validate_file(manifest_tmp, allow_missing=True)
            _ = manifest_tmp.write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            replace(manifest_tmp, manifest_path)
        finally:
            _ = manifest_tmp.unlink(missing_ok=True)
    validate_directory(generation_path)
    validate_file(contract_path)
    contract_tmp = output_root / ".full_canonical.trusted-contract.json.tmp"
    try:
        validate_file(contract_tmp, allow_missing=True)
        hard_link(contract_path, contract_tmp)
        replace(contract_tmp, output_root / "full_canonical.trusted-contract.json")
    finally:
        validate_file(contract_tmp, allow_missing=True)
        _ = contract_tmp.unlink(missing_ok=True)
    pointer = CanonicalPointer(
        generation=generation,
        manifest_sha256=sha256_file(manifest_path),
        dataset_sha256=manifest.dataset_sha256,
        logical_digest=manifest.logical_digest,
        trusted_contract_sha256=sha256_file(contract_path),
    )
    pointer_tmp = output_root / f".{DEFAULT_POINTER_NAME}.tmp"
    try:
        validate_file(pointer_tmp, allow_missing=True)
        _ = pointer_tmp.write_text(pointer.model_dump_json(indent=2), encoding="utf-8")
        replace(pointer_tmp, output_root / DEFAULT_POINTER_NAME)
    finally:
        validate_file(pointer_tmp, allow_missing=True)
        _ = pointer_tmp.unlink(missing_ok=True)


def _fail_integrity(reason_code: str) -> NoReturn:
    raise PublicationIntegrityError(reason_code)


def load_published_canonical(output_root: Path) -> PublishedCanonical:  # noqa: C901, PLR0912
    """Accept only the complete generation named by the current pointer."""
    pointer_path = output_root / DEFAULT_POINTER_NAME
    try:
        validate_directory(output_root)
        validate_directory(output_root / DEFAULT_GENERATIONS_DIR)
        validate_file(pointer_path, allow_missing=True)
    except FileNotFoundError:
        _fail_integrity("publication_pointer_missing")
    except OSError:
        _fail_integrity("publication_path_invalid")
    if not pointer_path.is_file():
        _fail_integrity("publication_pointer_missing")
    pointer = CanonicalPointer.model_validate_json(read_bytes(pointer_path))
    generation_path, dataset_path, manifest_path, contract_path = _generation_paths(
        output_root, pointer.generation
    )
    try:
        validate_directory(generation_path)
        validate_file(dataset_path, allow_missing=True)
        validate_file(manifest_path, allow_missing=True)
        validate_file(contract_path, allow_missing=True)
    except FileNotFoundError:
        _fail_integrity("publication_generation_missing")
    except OSError:
        _fail_integrity("publication_path_invalid")
    if not generation_path.is_dir():
        _fail_integrity("publication_generation_missing")
    if not dataset_path.is_file() or not manifest_path.is_file():
        _fail_integrity("publication_generation_incomplete")
    manifest = CanonicalManifest.model_validate_json(read_bytes(manifest_path))
    if sha256_file(manifest_path) != pointer.manifest_sha256:
        _fail_integrity("publication_manifest_digest_mismatch")
    if sha256_file(dataset_path) != pointer.dataset_sha256:
        _fail_integrity("publication_dataset_digest_mismatch")
    if manifest.dataset_sha256 != pointer.dataset_sha256:
        _fail_integrity("publication_manifest_dataset_mismatch")
    if manifest.logical_digest != pointer.logical_digest:
        _fail_integrity("publication_logical_digest_mismatch")
    contract = TrustedContractArtifact.model_validate_json(read_bytes(contract_path))
    if contract.generation != pointer.generation:
        _fail_integrity("trusted_contract_generation_mismatch")
    if sha256_file(contract_path) != pointer.trusted_contract_sha256:
        _fail_integrity("trusted_contract_digest_mismatch")
    return PublishedCanonical(
        manifest=manifest,
        trusted_contract=contract.contract,
        dataset_path=dataset_path,
        generation_path=generation_path,
    )
