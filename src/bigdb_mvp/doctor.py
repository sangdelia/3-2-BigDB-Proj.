"""Live, conservative GPU readiness checks for supervised training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, assert_never

from pydantic import BaseModel, ConfigDict

from bigdb_mvp.cache_probe import ModelCacheReport  # noqa: TC001
from bigdb_mvp.config import DoctorSettings


class NvmlStatus(StrEnum):
    """NVML query states that affect the GPU safety decision."""

    READY = "ready"
    DRIVER_LIBRARY_MISMATCH = "driver_library_mismatch"
    UNAVAILABLE = "unavailable"


class ReasonCode(StrEnum):
    """Stable machine-readable reason codes for GPU readiness."""

    READY = "ready"
    NVML_DRIVER_LIBRARY_MISMATCH = "nvml_driver_library_mismatch"
    NVML_UNAVAILABLE = "nvml_unavailable"
    CUDA_VISIBLE_DEVICES_MISMATCH = "cuda_visible_devices_mismatch"
    GPU_DEVICE_NOT_FOUND = "gpu_device_not_found"
    GPU_NAME_MISMATCH = "gpu_name_mismatch"
    FOREIGN_GPU_PROCESS_PRESENT = "foreign_gpu_process_present"
    GPU_PROCESS_QUERY_UNAVAILABLE = "gpu_process_query_unavailable"
    INSUFFICIENT_FREE_VRAM = "insufficient_free_vram"
    INSUFFICIENT_DISK = "insufficient_disk"
    TORCH_CUDA_UNAVAILABLE = "torch_cuda_unavailable"
    CUDA_TENSOR_SMOKE_FAILED = "cuda_tensor_smoke_failed"


@dataclass(frozen=True, slots=True)
class ProbeSnapshot:
    """Typed outputs from live host, NVML, and torch probes."""

    disk_free_bytes: int
    cuda_visible_devices: str | None
    device_index: int
    device_name: str | None
    nvml_status: NvmlStatus
    foreign_compute_process_count: int | None
    free_vram_bytes: int | None
    torch_cuda_available: bool
    tensor_smoke_passed: bool
    nvml_diagnostic: str | None = None
    model_cache: ModelCacheReport | None = None


class DoctorReport(BaseModel):
    """JSON-safe readiness result with all observed probe values."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    gpu_ready: bool
    primary_reason_code: ReasonCode
    reason_codes: tuple[ReasonCode, ...]
    disk_free_bytes: int
    cuda_visible_devices: str | None
    device_index: int
    device_name: str | None
    nvml_status: NvmlStatus
    nvml_diagnostic: str | None
    foreign_compute_process_count: int | None
    free_vram_bytes: int | None
    torch_cuda_available: bool
    tensor_smoke_passed: bool
    model_cache: ModelCacheReport | None


def derive_doctor_report(
    snapshot: ProbeSnapshot,
    settings: DoctorSettings | None = None,
) -> DoctorReport:
    """Derive readiness solely from injected probe results and strict limits."""
    active_settings = settings if settings is not None else DoctorSettings()
    reason_codes = _reason_codes(snapshot, active_settings)
    gpu_ready = len(reason_codes) == 0
    final_reason_codes = reason_codes or (ReasonCode.READY,)
    return DoctorReport(
        gpu_ready=gpu_ready,
        primary_reason_code=final_reason_codes[0],
        reason_codes=final_reason_codes,
        disk_free_bytes=snapshot.disk_free_bytes,
        cuda_visible_devices=snapshot.cuda_visible_devices,
        device_index=snapshot.device_index,
        device_name=snapshot.device_name,
        nvml_status=snapshot.nvml_status,
        nvml_diagnostic=snapshot.nvml_diagnostic,
        foreign_compute_process_count=snapshot.foreign_compute_process_count,
        free_vram_bytes=snapshot.free_vram_bytes,
        torch_cuda_available=snapshot.torch_cuda_available,
        tensor_smoke_passed=snapshot.tensor_smoke_passed,
        model_cache=snapshot.model_cache,
    )


def _reason_codes(
    snapshot: ProbeSnapshot,
    settings: DoctorSettings,
) -> tuple[ReasonCode, ...]:
    """Collect every failed prerequisite in stable presentation order."""
    foreign_processes_are_safe = snapshot.foreign_compute_process_count is None
    if not foreign_processes_are_safe:
        foreign_processes_are_safe = snapshot.foreign_compute_process_count == 0
    checks = (
        (_nvml_reason(snapshot.nvml_status), snapshot.nvml_status == NvmlStatus.READY),
        (
            ReasonCode.CUDA_VISIBLE_DEVICES_MISMATCH,
            snapshot.cuda_visible_devices == str(settings.device),
        ),
        (ReasonCode.GPU_DEVICE_NOT_FOUND, snapshot.device_name is not None),
        (
            ReasonCode.GPU_NAME_MISMATCH,
            snapshot.device_name is not None
            and settings.expected_gpu_name in snapshot.device_name,
        ),
        (
            ReasonCode.GPU_PROCESS_QUERY_UNAVAILABLE,
            snapshot.foreign_compute_process_count is not None,
        ),
        (
            ReasonCode.FOREIGN_GPU_PROCESS_PRESENT,
            foreign_processes_are_safe,
        ),
        (
            ReasonCode.INSUFFICIENT_FREE_VRAM,
            snapshot.free_vram_bytes is not None
            and snapshot.free_vram_bytes >= settings.minimum_free_vram_bytes,
        ),
        (
            ReasonCode.INSUFFICIENT_DISK,
            snapshot.disk_free_bytes >= settings.minimum_disk_free_bytes,
        ),
        (ReasonCode.TORCH_CUDA_UNAVAILABLE, snapshot.torch_cuda_available),
        (ReasonCode.CUDA_TENSOR_SMOKE_FAILED, snapshot.tensor_smoke_passed),
    )
    return tuple(reason for reason, passed in checks if not passed)


def _nvml_reason(status: NvmlStatus) -> ReasonCode:
    """Map the closed NVML state set to its public reason code."""
    match status:
        case NvmlStatus.READY:
            return ReasonCode.READY
        case NvmlStatus.DRIVER_LIBRARY_MISMATCH:
            return ReasonCode.NVML_DRIVER_LIBRARY_MISMATCH
        case NvmlStatus.UNAVAILABLE:
            return ReasonCode.NVML_UNAVAILABLE
        case _ as unreachable:
            assert_never(unreachable)
