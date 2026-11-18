from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from bigdb_mvp.doctor import (
    DoctorReport,
    NvmlStatus,
    ProbeSnapshot,
    derive_doctor_report,
)
from bigdb_mvp.training_contracts import (
    TrainingState,
    XlmArtifact,
    XlmTrainingConfig,
    blocked_receipt,
)
from bigdb_mvp.xlm_training import GpuTrainingReady, start_xlm_training


def _blocked_report() -> DoctorReport:
    return derive_doctor_report(
        ProbeSnapshot(
            disk_free_bytes=20 * 1024**3,
            cuda_visible_devices="0",
            device_index=0,
            device_name=None,
            nvml_status=NvmlStatus.DRIVER_LIBRARY_MISMATCH,
            foreign_compute_process_count=None,
            free_vram_bytes=None,
            torch_cuda_available=False,
            tensor_smoke_passed=False,
            nvml_diagnostic="driver/library version mismatch",
        )
    )


def _ready_report() -> DoctorReport:
    return derive_doctor_report(
        ProbeSnapshot(
            disk_free_bytes=20 * 1024**3,
            cuda_visible_devices="0",
            device_index=0,
            device_name="NVIDIA RTX A6000",
            nvml_status=NvmlStatus.READY,
            foreign_compute_process_count=0,
            free_vram_bytes=20 * 1024**3,
            torch_cuda_available=True,
            tensor_smoke_passed=True,
        )
    )


def test_xlm_config_freezes_the_documented_gpu_training_shape() -> None:
    # Given: the default fine-tuning contract.
    config = XlmTrainingConfig()

    # When: its immutable hyperparameters are inspected.
    # Then: the model revision and one-epoch GPU0 shape are reproducible.
    assert config.model_id == "FacebookAI/xlm-roberta-base"
    assert config.revision == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    assert (config.max_length, config.batch_size, config.seed, config.epochs) == (
        128,
        16,
        42,
        1,
    )
    assert config.training_languages == "ko"
    assert config.device == "cuda:0"


def test_blocked_receipt_preserves_cuda_diagnostic_and_remediation() -> None:
    # Given: the reproduced host driver/library mismatch.
    report = _blocked_report()

    # When: training converts the failed preflight to a terminal receipt.
    receipt = blocked_receipt(report, XlmTrainingConfig())

    # Then: it is explicitly ineligible and tells the operator how to unblock it.
    assert receipt.state is TrainingState.BLOCKED
    assert receipt.eligible_for_serving is False
    assert receipt.reason_code == "nvml_driver_library_mismatch"
    assert receipt.command == (
        "bigdb-mvp",
        "train",
        "xlm-roberta",
        "--epochs",
        "1",
        "--device",
        "cuda:0",
        "--seed",
        "42",
        "--training-languages",
        "ko",
    )
    assert "NVIDIA" in receipt.remediation


def test_partial_states_are_never_eligible_for_serving() -> None:
    # Given: each terminal failure path can leave checkpoint bytes behind.
    partial_states = (
        TrainingState.BLOCKED,
        TrainingState.OOM,
        TrainingState.CUDA_ERROR,
        TrainingState.CANCELLED,
    )

    # When: each partial artifact is evaluated by the shared eligibility gate.
    # Then: no partial state can be selected by a prediction surface.
    for state in partial_states:
        artifact = XlmArtifact.partial(state=state, reason_code=state.value)
        assert artifact.eligible_for_serving is False


def test_complete_artifact_requires_checkpoint_calibrator_and_variance() -> None:
    # Given: a nominally complete artifact missing its fitted calibrator.
    artifact = XlmArtifact.complete_for_test(calibrator_temperature=None)

    # When / Then: the serving gate rejects the incomplete model contract.
    assert artifact.eligible_for_serving is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("device", "cuda:1"),
        ("model_id", "arbitrary/model"),
        ("revision", "main"),
        ("training_languages", "en"),
    ],
)
def test_xlm_config_rejects_non_pinned_execution_contract(
    field: str, value: str
) -> None:
    # Given: an attempt to weaken one pinned GPU/model boundary.
    values = {field: value}

    # When / Then: parsing fails before training or eligibility is considered.
    with pytest.raises(ValidationError):
        _ = XlmTrainingConfig.model_validate(values)


def test_complete_artifact_rejects_non_finite_or_unverified_outputs() -> None:
    # Given: otherwise complete manifests with unsafe calibration/checkpoint claims.
    non_finite = XlmArtifact.complete_for_test(calibrator_temperature=math.inf)
    unverified = XlmArtifact.complete_for_test(calibrator_temperature=1.0).model_copy(
        update={"checkpoint_digest_verified": False}
    )

    # When / Then: neither can cross the serving eligibility boundary.
    assert non_finite.eligible_for_serving is False
    assert unverified.eligible_for_serving is False


def test_ready_preflight_never_falls_back_to_cpu() -> None:
    # Given: every GPU0 predicate is satisfied by an injected live report.
    report = _ready_report()

    # When: the training gate is opened.
    result = start_xlm_training(XlmTrainingConfig(), report)

    # Then: it exposes a CUDA-only ready state rather than a CPU fallback.
    assert isinstance(result, GpuTrainingReady)
    assert result.config.device == "cuda:0"
