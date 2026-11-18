"""Pointer, generation, manifest, and trusted-contract binding checks."""

# ruff: noqa: EM101

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from bigdb_mvp.data.canonical_path_safety import (
    read_bytes,
    validate_directory,
    validate_file,
)
from bigdb_mvp.data.canonical_publication import (
    DEFAULT_DATASET_NAME,
    DEFAULT_GENERATIONS_DIR,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_POINTER_NAME,
    CanonicalPointer,
    TrustedContractArtifact,
)
from bigdb_mvp.data.manifest import CanonicalManifest
from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from pathlib import Path


class PublicationBindingError(Exception):
    """Identify one failed publication binding gate."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize a digest-only publication binding failure."""
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class PublicationSelection:
    """Digest-consistent paths selected by the current publication pointer."""

    pointer: CanonicalPointer
    manifest: CanonicalManifest
    dataset_path: Path
    manifest_path: Path


def load_publication(root: Path) -> PublicationSelection:  # noqa: C901
    """Load only a digest-consistent current publication."""
    pointer_path = root / DEFAULT_POINTER_NAME
    try:
        validate_directory(root)
        validate_directory(root / DEFAULT_GENERATIONS_DIR)
        validate_file(pointer_path)
    except OSError as error:
        raise PublicationBindingError("current_pointer_missing") from error
    try:
        pointer = CanonicalPointer.model_validate_json(read_bytes(pointer_path))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise PublicationBindingError("current_pointer_invalid") from error
    generation = root / DEFAULT_GENERATIONS_DIR / pointer.generation
    dataset_path = generation / DEFAULT_DATASET_NAME
    manifest_path = generation / DEFAULT_MANIFEST_NAME
    contract_path = generation / "full_canonical.trusted-contract.json"
    try:
        validate_directory(generation)
        validate_file(dataset_path)
        validate_file(manifest_path)
        validate_file(contract_path)
    except OSError as error:
        raise PublicationBindingError("current_pointer_mismatch") from error
    if sha256_file(manifest_path) != pointer.manifest_sha256:
        raise PublicationBindingError("manifest_digest_mismatch")
    try:
        manifest = CanonicalManifest.model_validate_json(read_bytes(manifest_path))
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise PublicationBindingError("manifest_invalid") from error
    if manifest.logical_digest not in {pointer.logical_digest, pointer.generation}:
        raise PublicationBindingError("current_pointer_mismatch")
    if (
        sha256_file(dataset_path) != pointer.dataset_sha256
        or sha256_file(dataset_path) != manifest.dataset_sha256
    ):
        raise PublicationBindingError("dataset_digest_mismatch")
    if sha256_file(contract_path) != pointer.trusted_contract_sha256:
        raise PublicationBindingError("trusted_contract_digest_mismatch")
    try:
        contract = TrustedContractArtifact.model_validate_json(
            read_bytes(contract_path)
        )
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise PublicationBindingError("trusted_contract_invalid") from error
    if contract.generation != pointer.generation:
        raise PublicationBindingError("trusted_contract_generation_mismatch")
    return PublicationSelection(pointer, manifest, dataset_path, manifest_path)
