from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from bigdb_mvp.config import DoctorSettings


def test_project_uses_strict_python_311_contract() -> None:
    # Given: the project metadata file.
    pyproject = Path("pyproject.toml")

    # When: its text is inspected as a project-level contract.
    text = pyproject.read_text(encoding="utf-8")

    # Then: it pins the intended Python line and strict checker mode.
    assert 'requires-python = ">=3.11,<3.12"' in text
    assert 'typeCheckingMode = "all"' in text


def test_doctor_settings_reject_negative_cuda_device() -> None:
    # Given: a device selector outside the valid CUDA index range.
    invalid_device = -1

    # When / Then: the typed boundary rejects it before probing hardware.
    with pytest.raises(ValidationError):
        _: DoctorSettings = DoctorSettings(device=invalid_device)


def test_cli_exposes_doctor_as_a_subcommand() -> None:
    # Given: the installed console-script entry point.
    entrypoint = Path(sys.executable).with_name("bigdb-mvp")

    # When: a user invokes the documented doctor subcommand.
    result = subprocess.run(  # noqa: S603
        (str(entrypoint), "doctor", "--json"),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    # Then: the documented doctor invocation remains addressable as a subcommand.
    assert result.returncode == 0
    assert '"gpu_ready"' in result.stdout
