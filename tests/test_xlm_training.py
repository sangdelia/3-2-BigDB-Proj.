from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
import pytest

from bigdb_mvp.doctor import (
    DoctorReport,
    NvmlStatus,
    ProbeSnapshot,
    derive_doctor_report,
)
from bigdb_mvp.training_contracts import TrainingState, XlmTrainingConfig
from bigdb_mvp.xlm_training import (
    BackendOutput,
    TrainingRequest,
    XlmBackendError,
    XlmRunPaths,
    balanced_training_batches,
    load_xlm_artifact,
    run_xlm_training,
)

if TYPE_CHECKING:
    from pathlib import Path


class RecordingBackend:
    captured: list[TrainingRequest]

    def __init__(self) -> None:
        self.captured = []

    def train(self, request: TrainingRequest, output_dir: Path) -> BackendOutput:
        self.captured.append(request)
        checkpoint = output_dir / "checkpoint"
        tokenizer = output_dir / "tokenizer"
        checkpoint.mkdir(parents=True)
        tokenizer.mkdir(parents=True)
        _ = (checkpoint / "model.safetensors").write_bytes(b"trained")
        _ = (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
        return BackendOutput(
            calibration_logits=tuple(
                -2.0 if row.label == 0 else 2.0 for row in request.calibration
            ),
            test_logits=tuple(-2.0 if row.label == 0 else 2.0 for row in request.test),
            checkpoint_dir=checkpoint,
            tokenizer_dir=tokenizer,
        )


class FailingBackend:
    state: TrainingState

    def __init__(self, state: TrainingState) -> None:
        self.state = state

    def train(self, request: TrainingRequest, output_dir: Path) -> BackendOutput:
        _ = request
        _ = (output_dir / "partial.bin").write_bytes(b"partial")
        if self.state is TrainingState.CANCELLED:
            raise KeyboardInterrupt
        raise XlmBackendError(state=self.state, reason_code=self.state.value)


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


def _write_dataset(root: Path) -> tuple[Path, Path]:
    rows = [
        {
            "text": f"{language}-{split}-{label}-{index}",
            "label": label,
            "language": language,
            "split": split,
        }
        for language in ("ko",)
        for split in ("train", "calibration", "test")
        for label in ("clean", "harmful")
        for index in range(8 if split == "train" else 2)
    ]
    dataset_path = root / "sample.parquet"
    pl.DataFrame(rows).write_parquet(dataset_path)
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    _ = manifest_path.write_text(
        '{"schema_version":1,"seed":42,"sources":[],"dataset_sha256":"'
        + digest
        + '","logical_digest":"'
        + "1" * 64
        + '"}',
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def test_training_batches_contain_only_korean_rows(tmp_path: Path) -> None:
    # Given: Korean train rows and the fixed batch-16 configuration.
    dataset_path, _ = _write_dataset(tmp_path)

    # When: train-only batches are assembled.
    batches = balanced_training_batches(dataset_path, XlmTrainingConfig())

    # Then: every full batch contains Korean train rows only.
    assert batches
    assert all(len(batch) == 16 for batch in batches)
    assert all({row.language for row in batch} == {"ko"} for batch in batches)
    assert all(row.split == "train" for batch in batches for row in batch)


def test_ready_training_produces_digest_bound_eligible_artifact(tmp_path: Path) -> None:
    # Given: a ready GPU report, complete Korean splits, and a tiny backend.
    dataset_path, manifest_path = _write_dataset(tmp_path)
    backend = RecordingBackend()

    # When: the ready path runs the same orchestration used by the CLI.
    artifact = run_xlm_training(
        report=_ready_report(),
        backend=backend,
        paths=XlmRunPaths(
            dataset=dataset_path,
            manifest=manifest_path,
            artifact_dir=tmp_path / "artifact",
        ),
    )

    # Then: training ran once and yielded a calibrated, evaluated CUDA0 checkpoint.
    assert len(backend.captured) == 1
    assert artifact.state is TrainingState.COMPLETE
    assert artifact.eligible_for_serving is True
    assert artifact.config.device == "cuda:0"
    assert artifact.calibrator_temperature is not None
    assert artifact.calibrator_sha256 is not None
    assert artifact.threshold == 0.05
    assert {item.language for item in artifact.metrics} == {"overall", "ko"}
    assert {item.label for item in artifact.prediction_distribution} == {
        "clean",
        "harmful",
    }
    assert artifact.checkpoint_digest_verified is True
    assert load_xlm_artifact(tmp_path / "artifact") is not None


def test_loader_rejects_checkpoint_digest_drift(tmp_path: Path) -> None:
    # Given: a finalized eligible artifact whose checkpoint is later modified.
    dataset_path, manifest_path = _write_dataset(tmp_path)
    artifact_dir = tmp_path / "artifact"
    _ = run_xlm_training(
        report=_ready_report(),
        backend=RecordingBackend(),
        paths=XlmRunPaths(dataset_path, manifest_path, artifact_dir),
    )
    _ = (artifact_dir / "checkpoint/model.safetensors").write_bytes(b"changed")

    # When: the serving loader recomputes live directory digests.
    loaded = load_xlm_artifact(artifact_dir)

    # Then: digest drift makes the XLM-R artifact unavailable.
    assert loaded is None


def test_loader_treats_corrupt_manifest_as_unavailable(tmp_path: Path) -> None:
    # Given: bytes that cannot satisfy the persisted XLM artifact schema.
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _ = (artifact_dir / "xlm_manifest.json").write_text("not-json", encoding="utf-8")

    # When: the prediction surface probes optional XLM availability.
    loaded = load_xlm_artifact(artifact_dir)

    # Then: corruption cannot escape or become eligible.
    assert loaded is None


@pytest.mark.parametrize(
    "state",
    [TrainingState.OOM, TrainingState.CUDA_ERROR, TrainingState.CANCELLED],
)
def test_terminal_gpu_failures_leave_no_checkpoint(
    tmp_path: Path, state: TrainingState
) -> None:
    # Given: a backend that writes temporary bytes before a terminal GPU failure.
    dataset_path, manifest_path = _write_dataset(tmp_path)
    artifact_dir = tmp_path / "artifact"

    # When: the orchestrator handles OOM, CUDA error, or operator cancellation.
    artifact = run_xlm_training(
        report=_ready_report(),
        backend=FailingBackend(state),
        paths=XlmRunPaths(dataset_path, manifest_path, artifact_dir),
    )

    # Then: the partial state is ineligible and staging cleanup removes all weights.
    assert artifact.state is state
    assert artifact.eligible_for_serving is False
    assert not (artifact_dir / "checkpoint").exists()
