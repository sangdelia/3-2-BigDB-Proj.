"""Descriptor-relative path operations for canonical publication files."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Final

_DIRECTORY_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _open_directory(path: Path, *, create: bool) -> int:
    absolute = _absolute_path(path)
    anchor = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    descriptor = anchor
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component, _DIRECTORY_FLAGS, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                next_descriptor = os.open(
                    component, _DIRECTORY_FLAGS, dir_fd=descriptor
                )
            if descriptor != anchor:
                os.close(descriptor)
            descriptor = next_descriptor
        if descriptor == anchor:
            descriptor = os.dup(anchor)
        else:
            return descriptor
    except (KeyboardInterrupt, SystemExit, OSError, ValueError):
        if descriptor != anchor:
            os.close(descriptor)
        raise
    finally:
        os.close(anchor)


def ensure_publication_directories(output_root: Path) -> None:
    """Create publication directories without following links."""
    root_descriptor = _open_directory(output_root, create=True)
    try:
        try:
            generation_descriptor = os.open(
                "full_canonical.generations",
                _DIRECTORY_FLAGS,
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            os.mkdir("full_canonical.generations", mode=0o755, dir_fd=root_descriptor)
            generation_descriptor = os.open(
                "full_canonical.generations",
                _DIRECTORY_FLAGS,
                dir_fd=root_descriptor,
            )
        os.close(generation_descriptor)
    finally:
        os.close(root_descriptor)


def validate_directory(path: Path) -> None:
    """Reject missing, symlinked, or non-directory publication ancestors."""
    descriptor = _open_directory(path, create=False)
    os.close(descriptor)


def validate_file(path: Path, *, allow_missing: bool = False) -> None:
    """Reject symlinked or directory final targets and unsafe ancestors."""
    parent = _open_directory(path.parent, create=False)
    try:
        try:
            descriptor = os.open(path.name, _FILE_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            if not allow_missing:
                raise
            return
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            reason = f"publication target is not a regular file: {path}"
            raise OSError(reason)
        os.close(descriptor)
    finally:
        os.close(parent)


def read_bytes(path: Path) -> bytes:
    """Read a publication file through a no-follow descriptor."""
    validate_file(path)
    descriptor = os.open(_absolute_path(path), _FILE_FLAGS)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def replace_path(source: Path, target: Path) -> None:
    """Atomically replace a final path after descriptor-relative validation."""
    source_parent = _open_directory(source.parent, create=False)
    target_parent = _open_directory(target.parent, create=False)
    try:
        _validate_entry(source, allow_missing=False)
        _validate_entry(target, allow_missing=True)
        os.replace(
            source.name,
            target.name,
            src_dir_fd=source_parent,
            dst_dir_fd=target_parent,
        )
    finally:
        os.close(source_parent)
        os.close(target_parent)


def hard_link(source: Path, target: Path) -> None:
    """Create a no-follow hard link used as an atomic replacement source."""
    source_parent = _open_directory(source.parent, create=False)
    target_parent = _open_directory(target.parent, create=False)
    try:
        validate_file(source)
        validate_file(target, allow_missing=True)
        os.link(
            source.name,
            target.name,
            src_dir_fd=source_parent,
            dst_dir_fd=target_parent,
            follow_symlinks=False,
        )
    finally:
        os.close(source_parent)
        os.close(target_parent)


def _validate_entry(path: Path, *, allow_missing: bool) -> None:
    parent = _open_directory(path.parent, create=False)
    try:
        try:
            entry = os.lstat(path.name, dir_fd=parent)
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        if stat.S_ISLNK(entry.st_mode):
            reason = f"symlinked publication target: {path}"
            raise OSError(reason)
    finally:
        os.close(parent)
