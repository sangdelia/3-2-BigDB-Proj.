"""Bounded host probes used by the non-mutating GPU readiness doctor."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from bigdb_mvp.cache_probe import probe_model_cache
from bigdb_mvp.doctor import NvmlStatus, ProbeSnapshot

if TYPE_CHECKING:
    from bigdb_mvp.config import DoctorSettings

COMMAND_TIMEOUT_SECONDS: Final = 5
GPU_CSV_FIELD_COUNT: Final = 3
MAX_NVML_DIAGNOSTIC_CHARACTERS: Final = 512


@dataclass(frozen=True, slots=True)
class NvmlProbe:
    """The minimal GPU host state collected through nvidia-smi."""

    status: NvmlStatus
    device_name: str | None
    free_vram_bytes: int | None
    foreign_compute_process_count: int | None
    diagnostic: str | None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded subprocess output without propagating host-tool failures."""

    return_code: int | None
    stdout: str
    stderr: str


def collect_live_snapshot(settings: DoctorSettings) -> ProbeSnapshot:
    """Run bounded live checks without caching or changing GPU state."""
    nvml_probe = _collect_nvml_probe(settings)
    torch_cuda_available, tensor_smoke_passed = _torch_cuda_probe(settings.device)
    return ProbeSnapshot(
        disk_free_bytes=_disk_free_bytes(Path.cwd()),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_index=settings.device,
        device_name=nvml_probe.device_name,
        nvml_status=nvml_probe.status,
        nvml_diagnostic=nvml_probe.diagnostic,
        foreign_compute_process_count=nvml_probe.foreign_compute_process_count,
        free_vram_bytes=nvml_probe.free_vram_bytes,
        torch_cuda_available=torch_cuda_available,
        tensor_smoke_passed=tensor_smoke_passed,
        model_cache=probe_model_cache(settings),
    )


def _disk_free_bytes(path: Path) -> int:
    """Return the current filesystem free-byte count for the project path."""
    filesystem = os.statvfs(path)
    return filesystem.f_bavail * filesystem.f_frsize


def _collect_nvml_probe(settings: DoctorSettings) -> NvmlProbe:
    """Query selected GPU details and foreign process count through nvidia-smi."""
    device_result = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ),
    )
    if device_result.return_code != 0:
        return NvmlProbe(
            status=_nvml_status_from_output(device_result.stdout, device_result.stderr),
            device_name=None,
            free_vram_bytes=None,
            foreign_compute_process_count=None,
            diagnostic=_bounded_diagnostic(device_result.stdout, device_result.stderr),
        )
    device_name, free_vram_bytes = _parse_gpu_row(device_result.stdout, settings.device)
    process_result = _run_command(
        ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"),
    )
    foreign_count = (
        _foreign_process_count(process_result.stdout)
        if process_result.return_code == 0
        else None
    )
    return NvmlProbe(
        status=NvmlStatus.READY,
        device_name=device_name,
        free_vram_bytes=free_vram_bytes,
        foreign_compute_process_count=foreign_count,
        diagnostic=None,
    )


def _nvml_status_from_output(stdout: str, stderr: str) -> NvmlStatus:
    """Classify the documented nvidia-smi driver/library compatibility error."""
    if "driver/library version mismatch" in f"{stdout}\n{stderr}".lower():
        return NvmlStatus.DRIVER_LIBRARY_MISMATCH
    return NvmlStatus.UNAVAILABLE


def _bounded_diagnostic(stdout: str, stderr: str) -> str | None:
    """Preserve a bounded, single-line nvidia-smi failure diagnostic."""
    diagnostic = " ".join(f"{stdout}\n{stderr}".split())
    if not diagnostic:
        return None
    return diagnostic[:MAX_NVML_DIAGNOSTIC_CHARACTERS]


def _parse_gpu_row(stdout: str, device_index: int) -> tuple[str | None, int | None]:
    """Extract selected GPU name and free VRAM from nvidia-smi CSV output."""
    for row in stdout.splitlines():
        fields = tuple(field.strip() for field in row.split(","))
        if len(fields) != GPU_CSV_FIELD_COUNT or fields[0] != str(device_index):
            continue
        try:
            free_vram_bytes = int(fields[2]) * 1024**2
        except ValueError:
            return fields[1], None
        return fields[1], free_vram_bytes
    return None, None


def _foreign_process_count(stdout: str) -> int | None:
    """Count compute processes not owned by the current Unix user."""
    current_user_result = _run_command(("id", "-un"))
    current_user = current_user_result.stdout.strip()
    if current_user_result.return_code != 0 or not current_user:
        return None
    foreign_count = 0
    for row in stdout.splitlines():
        process_id = row.strip()
        if not process_id:
            continue
        owner_result = _run_command(("ps", "-o", "user=", "-p", process_id))
        owner = owner_result.stdout.strip()
        if owner_result.return_code != 0 or not owner:
            return None
        if owner != current_user:
            foreign_count += 1
    return foreign_count


def _torch_cuda_probe(device: int) -> tuple[bool, bool]:
    """Require a real CUDA tensor operation, not only import-time availability."""
    code = "\n".join(
        (
            "import torch",
            "available = torch.cuda.is_available()",
            "print(f'available={int(available)}')",
            "if available:",
            f"    tensor = torch.tensor([1.0], device='cuda:{device}')",
            "    result = tensor + 1.0",
            "    torch.cuda.synchronize()",
            "    print(f'smoke={int(result.item() == 2.0)}')",
            "else:",
            "    print('smoke=0')",
        ),
    )
    result = _run_command((sys.executable, "-c", code))
    fields = tuple(line.partition("=") for line in result.stdout.splitlines())
    available = any(key == "available" and value == "1" for key, _, value in fields)
    smoke_passed = any(key == "smoke" and value == "1" for key, _, value in fields)
    return available, smoke_passed


def _run_command(arguments: tuple[str, ...]) -> CommandResult:
    """Execute a short host probe with a bounded timeout and captured output."""
    try:
        completed = subprocess.run(  # noqa: S603
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return CommandResult(return_code=None, stdout="", stderr="")
    except subprocess.TimeoutExpired:
        return CommandResult(return_code=None, stdout="", stderr="")
    return CommandResult(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
