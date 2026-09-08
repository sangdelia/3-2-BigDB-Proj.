from __future__ import annotations

from typing import TYPE_CHECKING

from bigdb_mvp import cache_probe, gpu_probe
from bigdb_mvp.cache_probe import ModelCacheReasonCode
from bigdb_mvp.config import DoctorSettings
from bigdb_mvp.doctor import NvmlStatus, ProbeSnapshot, derive_doctor_report
from bigdb_mvp.gpu_probe import CommandResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_doctor_marks_gpu_unready_when_nvml_driver_library_mismatch() -> None:
    # Given: every check except the NVML driver/library compatibility check passes.
    snapshot = ProbeSnapshot(
        disk_free_bytes=20 * 1024**3,
        cuda_visible_devices="0",
        device_index=0,
        device_name="NVIDIA RTX A6000",
        nvml_status=NvmlStatus.DRIVER_LIBRARY_MISMATCH,
        foreign_compute_process_count=0,
        free_vram_bytes=20 * 1024**3,
        torch_cuda_available=True,
        tensor_smoke_passed=True,
    )

    # When: the doctor derives the live readiness report.
    report = derive_doctor_report(snapshot)

    # Then: it is not a false positive and exposes the stable blocker code.
    assert report.gpu_ready is False
    assert report.primary_reason_code == "nvml_driver_library_mismatch"


def test_nvml_probe_returns_unavailable_when_current_user_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a real compute PID but a failed current-user query.
    calls: list[tuple[str, ...]] = []
    responses: dict[tuple[str, ...], CommandResult] = {
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ): CommandResult(0, "0, NVIDIA RTX A6000, 16384", ""),
        (
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ): CommandResult(0, "4321", ""),
        ("id", "-un"): CommandResult(1, "", ""),
    }

    def failed_owner_query(arguments: tuple[str, ...]) -> CommandResult:
        calls.append(arguments)
        return responses.get(arguments, CommandResult(None, "", ""))

    monkeypatch.setattr(gpu_probe, "_run_command", failed_owner_query)

    # When: NVML probing evaluates process ownership safety.
    snapshot = gpu_probe.collect_live_snapshot(DoctorSettings())

    # Then: failed ownership discovery is unavailable, not a safe zero count.
    assert snapshot.foreign_compute_process_count is None
    report = derive_doctor_report(
        ProbeSnapshot(
            disk_free_bytes=20 * 1024**3,
            cuda_visible_devices="0",
            device_index=0,
            device_name=snapshot.device_name,
            nvml_status=snapshot.nvml_status,
            foreign_compute_process_count=snapshot.foreign_compute_process_count,
            free_vram_bytes=20 * 1024**3,
            torch_cuda_available=True,
            tensor_smoke_passed=True,
        ),
    )
    assert report.gpu_ready is False
    assert report.primary_reason_code == "gpu_process_query_unavailable"


def test_collect_nvml_probe_returns_unavailable_owner_count_when_owner_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a parsed compute PID but a failed process-owner lookup.
    calls: list[tuple[str, ...]] = []
    responses: dict[tuple[str, ...], CommandResult] = {
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ): CommandResult(0, "0, NVIDIA RTX A6000, 16384", ""),
        (
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ): CommandResult(0, "4321", ""),
        ("id", "-un"): CommandResult(0, "kyu", ""),
        ("ps", "-o", "user=", "-p", "4321"): CommandResult(1, "", ""),
    }

    def failed_owner_query(arguments: tuple[str, ...]) -> CommandResult:
        calls.append(arguments)
        return responses.get(arguments, CommandResult(None, "", ""))

    monkeypatch.setattr(gpu_probe, "_run_command", failed_owner_query)

    # When: NVML probing evaluates the parsed PID's owner.
    snapshot = gpu_probe.collect_live_snapshot(DoctorSettings())

    # Then: the PID was not dropped and its failed owner lookup is unavailable.
    assert ("ps", "-o", "user=", "-p", "4321") in calls
    assert snapshot.foreign_compute_process_count is None


def test_model_cache_reports_writable_matching_snapshot(tmp_path: Path) -> None:
    repo_cache = tmp_path / "models--FacebookAI--xlm-roberta-base"
    snapshot = repo_cache / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    _ = (snapshot / "config.json").write_text("{}", encoding="utf-8")

    report = cache_probe.probe_model_cache(
        DoctorSettings(model_cache_root=tmp_path),
    )

    assert report.exists is True
    assert report.is_directory is True
    assert report.writable is True
    assert report.repo_cache_exists is True
    assert report.model_snapshot_cached is True
    assert report.free_bytes is not None
    assert report.free_bytes > 0
    assert report.reason_codes == (ModelCacheReasonCode.READY,)


def test_model_cache_reports_missing_snapshot_as_downloadable_later(
    tmp_path: Path,
) -> None:
    report = cache_probe.probe_model_cache(
        DoctorSettings(model_cache_root=tmp_path / "not-downloaded"),
    )

    assert report.exists is False
    assert report.is_directory is False
    assert report.writable is False
    assert report.repo_cache_exists is False
    assert report.model_snapshot_cached is False
    assert "model_cache_root_missing" in report.reason_codes
    assert "model_snapshot_missing" in report.reason_codes
    assert report.free_bytes is not None
    assert report.free_bytes > 0


def test_model_cache_reports_non_directory_root(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache-file"
    _ = cache_file.write_text("not a directory", encoding="utf-8")

    report = cache_probe.probe_model_cache(
        DoctorSettings(model_cache_root=cache_file),
    )

    assert report.exists is True
    assert report.is_directory is False
    assert report.writable is False
    assert report.model_snapshot_cached is False
    assert "model_cache_root_not_directory" in report.reason_codes
    assert "model_snapshot_missing" in report.reason_codes


def test_model_cache_reports_unwritable_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "unwritable"
    cache_root.mkdir()
    cache_root.chmod(0o500)

    report = cache_probe.probe_model_cache(
        DoctorSettings(model_cache_root=cache_root),
    )

    assert report.exists is True
    assert report.is_directory is True
    assert report.writable is False
    assert "model_cache_root_not_writable" in report.reason_codes


def test_model_cache_hf_hub_cache_overrides_hf_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_cache = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    report = cache_probe.probe_model_cache(DoctorSettings())

    assert report.cache_root == str(hub_cache)
    assert report.cache_root_source == "HF_HUB_CACHE"


def test_model_cache_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    report = cache_probe.probe_model_cache(
        DoctorSettings(model_cache_root=link),
    )

    assert report.exists is True
    assert report.is_directory is False
    assert report.writable is False
    assert "model_cache_root_symlink" in report.reason_codes
    assert report.model_snapshot_cached is False


def test_model_cache_reports_malformed_path_without_raising() -> None:
    report = cache_probe.probe_model_cache(
        DoctorSettings.model_validate({"model_cache_root": "\x00invalid"}),
    )

    assert report.exists is False
    assert report.is_directory is False
    assert report.writable is False
    assert "model_cache_root_invalid" in report.reason_codes
    assert report.free_bytes is None
