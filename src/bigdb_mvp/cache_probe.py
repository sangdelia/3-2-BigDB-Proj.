"""Bounded, non-mutating preflight for the pinned Hugging Face model cache."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from bigdb_mvp.config import MODEL_REPO_ID, DoctorSettings

HF_HUB_CACHE_ENV: Final = "HF_HUB_CACHE"
HF_HOME_ENV: Final = "HF_HOME"
MAX_SNAPSHOT_PROBES: Final = 64
REPO_CACHE_NAME: Final = "models--FacebookAI--xlm-roberta-base"


class CacheRootSource(StrEnum):
    """Source selected for the resolved Hugging Face cache root."""

    EXPLICIT = "explicit"
    HF_HUB_CACHE = "HF_HUB_CACHE"
    HF_HOME = "HF_HOME"
    DEFAULT = "default"


class ModelCacheReasonCode(StrEnum):
    """Stable reason codes for model-cache availability."""

    READY = "ready"
    ROOT_INVALID = "model_cache_root_invalid"
    ROOT_SYMLINK = "model_cache_root_symlink"
    ROOT_MISSING = "model_cache_root_missing"
    ROOT_NOT_DIRECTORY = "model_cache_root_not_directory"
    ROOT_NOT_WRITABLE = "model_cache_root_not_writable"
    REPO_CACHE_MISSING = "model_repo_cache_missing"
    SNAPSHOT_MISSING = "model_snapshot_missing"


class ModelCacheReport(BaseModel):
    """JSON-safe model-cache preflight result separate from GPU readiness."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    repo_id: str
    cache_root: str
    cache_root_source: CacheRootSource
    exists: bool
    is_directory: bool
    writable: bool
    free_bytes: int | None
    repo_cache_exists: bool
    model_snapshot_cached: bool
    reason_codes: tuple[ModelCacheReasonCode, ...]


@dataclass(frozen=True, slots=True)
class _CacheState:
    root_exists: bool
    root_is_directory: bool
    root_is_symlink: bool
    root_invalid: bool
    writable: bool
    repo_exists: bool
    snapshot_cached: bool


def probe_model_cache(settings: DoctorSettings) -> ModelCacheReport:
    """Inspect only the pinned repository path and at most 64 snapshots."""
    cache_root, source = _resolve_cache_root(settings)
    (
        root_exists,
        root_is_directory,
        root_is_symlink,
        root_invalid,
    ) = _root_state(cache_root)
    root_is_usable = (
        root_exists and root_is_directory and not root_is_symlink and not root_invalid
    )
    writable = _is_writable(cache_root) if root_is_usable else False
    free_bytes = None if root_invalid else _free_bytes(cache_root)
    repo_cache = cache_root / REPO_CACHE_NAME
    repo_exists = root_is_usable and _is_real_directory(repo_cache)
    snapshot_cached = repo_exists and _has_snapshot(repo_cache)
    reasons = _reasons(
        _CacheState(
            root_exists=root_exists,
            root_is_directory=root_is_directory,
            root_is_symlink=root_is_symlink,
            root_invalid=root_invalid,
            writable=writable,
            repo_exists=repo_exists,
            snapshot_cached=snapshot_cached,
        ),
    )
    return ModelCacheReport(
        repo_id=MODEL_REPO_ID,
        cache_root=str(cache_root),
        cache_root_source=source,
        exists=root_exists,
        is_directory=root_is_directory,
        writable=writable,
        free_bytes=free_bytes,
        repo_cache_exists=repo_exists,
        model_snapshot_cached=snapshot_cached,
        reason_codes=reasons or (ModelCacheReasonCode.READY,),
    )


def _resolve_cache_root(settings: DoctorSettings) -> tuple[Path, CacheRootSource]:
    """Apply explicit, standard HF, then default cache-root precedence."""
    if settings.model_cache_root is not None:
        return _absolute(settings.model_cache_root), CacheRootSource.EXPLICIT
    hub_cache = os.environ.get(HF_HUB_CACHE_ENV, "").strip()
    if hub_cache:
        return _absolute(Path(hub_cache)), CacheRootSource.HF_HUB_CACHE
    hf_home = os.environ.get(HF_HOME_ENV, "").strip()
    if hf_home:
        return _absolute(Path(hf_home) / "hub"), CacheRootSource.HF_HOME
    default_root = Path.home() / ".cache" / "huggingface" / "hub"
    return _absolute(default_root), CacheRootSource.DEFAULT


def _absolute(path: Path) -> Path:
    """Make a path absolute while preserving symlink identity for inspection."""
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _root_state(path: Path) -> tuple[bool, bool, bool, bool]:
    """Read root state with lstat so symlinks are never followed."""
    try:
        metadata = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False, False, False, False
    except ValueError:
        return False, False, False, True
    except OSError:
        return False, False, False, True
    is_symlink = stat.S_ISLNK(metadata.st_mode)
    return True, stat.S_ISDIR(metadata.st_mode) and not is_symlink, is_symlink, False


def _is_real_directory(path: Path) -> bool:
    """Return true only for a non-symlink directory."""
    exists, is_directory, is_symlink, invalid = _root_state(path)
    return exists and is_directory and not is_symlink and not invalid


def _is_writable(path: Path) -> bool:
    """Check write access without creating or modifying a sentinel file."""
    try:
        return os.access(path, os.W_OK)
    except (OSError, ValueError):
        return False


def _free_bytes(path: Path) -> int | None:
    """Return free bytes for the nearest existing ancestor, if queryable."""
    candidate = path
    for _ in range(MAX_SNAPSHOT_PROBES):
        try:
            if candidate.exists():
                return shutil.disk_usage(candidate).free
        except (OSError, ValueError):
            return None
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    return None


def _has_snapshot(repo_cache: Path) -> bool:
    """Find one populated revision directory under the exact repo cache."""
    snapshots = repo_cache / "snapshots"
    if not _is_real_directory(snapshots):
        return False
    try:
        revisions = tuple(snapshots.iterdir())[:MAX_SNAPSHOT_PROBES]
    except (OSError, ValueError):
        return False
    for revision in revisions:
        if not _is_real_directory(revision):
            continue
        try:
            _ = next(revision.iterdir())
        except (StopIteration, OSError, ValueError):
            continue
        return True
    return False


def _reasons(
    state: _CacheState,
) -> tuple[ModelCacheReasonCode, ...]:
    """Build stable reasons without changing the independent GPU predicates."""
    reasons: list[ModelCacheReasonCode] = []
    if state.root_is_symlink:
        reasons.append(ModelCacheReasonCode.ROOT_SYMLINK)
    elif state.root_invalid:
        reasons.append(ModelCacheReasonCode.ROOT_INVALID)
    elif not state.root_exists:
        reasons.append(ModelCacheReasonCode.ROOT_MISSING)
    elif not state.root_is_directory:
        reasons.append(ModelCacheReasonCode.ROOT_NOT_DIRECTORY)
    elif not state.writable:
        reasons.append(ModelCacheReasonCode.ROOT_NOT_WRITABLE)
    if not state.repo_exists:
        reasons.append(ModelCacheReasonCode.REPO_CACHE_MISSING)
    if not state.snapshot_cached:
        reasons.append(ModelCacheReasonCode.SNAPSHOT_MISSING)
    return tuple(reasons)
