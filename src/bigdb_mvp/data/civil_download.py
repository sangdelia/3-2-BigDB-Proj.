"""Atomic size-limited download of an immutable dataset asset."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final
from urllib.parse import urljoin, urlparse

import httpx2
from typing_extensions import override

from bigdb_mvp.data.http_client import create_client
from bigdb_mvp.data.staging import sha256_file

MAX_SHARD_BYTES: Final = 250 * 1024 * 1024
HTTP_PARTIAL_CONTENT: Final = 206
MAX_REDIRECTS: Final = 5
DOWNLOAD_ROOT: Final = Path(__file__).resolve().parents[3] / "data/raw/downloads"
STAGING_ROOT: Final = Path(__file__).resolve().parents[3] / "data/raw/staging"
OFFICIAL_HOSTS: Final = frozenset({"raw.githubusercontent.com", "huggingface.co"})
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")


class DownloadArea(StrEnum):
    """Approved local roots for immutable download targets."""

    RAW = "raw"
    STAGING = "staging"


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """Immutable remote asset and local cache boundary."""

    url: str
    target: Path
    expected_sha256: str
    expected_bytes: int
    maximum_bytes: int = MAX_SHARD_BYTES
    area: DownloadArea = DownloadArea.RAW

    def __post_init__(self) -> None:
        """Parse and validate the remote and local download boundary."""
        parsed = urlparse(self.url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.hostname not in OFFICIAL_HOSTS
            or not parsed.path
        ):
            reason = "download URL must be an official HTTPS asset"
            raise ValueError(reason)
        if SHA256_PATTERN.fullmatch(self.expected_sha256) is None:
            reason = "expected_sha256 must be lowercase SHA-256"
            raise ValueError(reason)
        if self.expected_bytes <= 0 or self.maximum_bytes <= 0:
            reason = "download sizes must be positive"
            raise ValueError(reason)
        root = STAGING_ROOT if self.area is DownloadArea.STAGING else DOWNLOAD_ROOT
        target = self.target
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
            reason = "download target cannot use a symlink"
            raise ValueError(reason)
        try:
            _ = target.resolve(strict=False).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as error:
            reason = "download target must be under an approved root"
            raise ValueError(reason) from error


class DownloadIntegrityError(Exception):
    """Report incomplete, oversized, or checksum-drifted cached assets."""

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        """Initialize the traceback-compatible typed failure."""
        self.reason_code = reason_code
        super().__init__(reason_code)

    @override
    def __str__(self) -> str:
        """Return the stable download-integrity code."""
        return self.reason_code


def _read_verified_cache(spec: DownloadSpec) -> str | None:
    checksum_path = spec.target.with_name(spec.target.name + ".sha256")
    if not spec.target.exists() and not checksum_path.exists():
        return None
    if not spec.target.exists() or not checksum_path.exists():
        reason = "civil_download_cache_incomplete"
        raise DownloadIntegrityError(reason)
    expected = checksum_path.read_text(encoding="ascii").strip()
    actual = sha256_file(spec.target)
    if (
        actual != expected
        or actual != spec.expected_sha256
        or spec.target.stat().st_size != spec.expected_bytes
    ):
        reason = "civil_download_checksum_drift"
        raise DownloadIntegrityError(reason)
    return actual


def _promote_complete_partial(spec: DownloadSpec, partial: Path) -> str | None:
    if not partial.exists() or partial.stat().st_size != spec.expected_bytes:
        return None
    actual = sha256_file(partial)
    if actual != spec.expected_sha256:
        partial.unlink()
        reason = "civil_download_checksum_drift"
        raise DownloadIntegrityError(reason)
    checksum_path = spec.target.with_name(spec.target.name + ".sha256")
    checksum_temporary = checksum_path.with_name(checksum_path.name + ".tmp")
    try:
        _ = checksum_temporary.write_text(actual + "\n", encoding="ascii")
        _ = partial.replace(spec.target)
        _ = checksum_temporary.replace(checksum_path)
    finally:
        checksum_temporary.unlink(missing_ok=True)
    return actual


def _download_into_partial(spec: DownloadSpec, partial: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        with partial.open("rb") as existing:
            while chunk := existing.read(1024 * 1024):
                digest.update(chunk)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    current_url = spec.url
    with create_client() as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            with client.stream("GET", current_url, headers=headers) as response:
                _validate_remote_url(str(response.url))
                if response.is_redirect:
                    location = response.headers.get("location")
                    if location is None or redirect_count == MAX_REDIRECTS:
                        reason = "civil_download_redirect_limit"
                        raise DownloadIntegrityError(reason)
                    current_url = urljoin(current_url, location)
                    _validate_remote_url(current_url)
                    continue
                _ = response.raise_for_status()
                append = offset > 0 and response.status_code == HTTP_PARTIAL_CONTENT
                if offset > 0 and not append:
                    offset = 0
                    digest = hashlib.sha256()
                advertised = response.headers.get("content-length")
                if advertised is not None and int(advertised) != (
                    spec.expected_bytes - offset
                ):
                    partial.unlink(missing_ok=True)
                    reason = "civil_download_content_length_drift"
                    raise DownloadIntegrityError(reason)
                downloaded = offset
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > spec.expected_bytes:
                            partial.unlink(missing_ok=True)
                            reason = "civil_download_size_exceeded"
                            raise DownloadIntegrityError(reason)
                        digest.update(chunk)
                        _ = handle.write(chunk)
                return digest.hexdigest(), downloaded
    reason = "civil_download_redirect_limit"
    raise DownloadIntegrityError(reason)


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        reason = "civil_download_redirect_origin"
        raise DownloadIntegrityError(reason)


def download_immutable_shard(spec: DownloadSpec) -> str:
    """Reuse a verified cache or atomically stream one bounded HTTP asset."""
    cached = _read_verified_cache(spec)
    if cached is not None:
        return cached
    spec.target.parent.mkdir(parents=True, exist_ok=True)
    partial = spec.target.with_name(spec.target.name + ".part")
    promoted = _promote_complete_partial(spec, partial)
    if promoted is not None:
        return promoted
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec.expected_bytes or spec.expected_bytes > spec.maximum_bytes:
        partial.unlink(missing_ok=True)
        reason = "civil_download_size_exceeded"
        raise DownloadIntegrityError(reason)
    try:
        actual, downloaded = _download_into_partial(spec, partial)
    except httpx2.TimeoutException as error:
        partial.unlink(missing_ok=True)
        reason = "civil_download_timeout"
        raise DownloadIntegrityError(reason) from error
    if downloaded != spec.expected_bytes:
        reason = "civil_download_incomplete"
        raise DownloadIntegrityError(reason)
    if actual != spec.expected_sha256:
        partial.unlink()
        reason = "civil_download_checksum_drift"
        raise DownloadIntegrityError(reason)
    promoted = _promote_complete_partial(spec, partial)
    if promoted is None:
        reason = "civil_download_incomplete"
        raise DownloadIntegrityError(reason)
    return promoted
