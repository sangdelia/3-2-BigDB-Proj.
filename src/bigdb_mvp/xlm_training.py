"""GPU-only XLM-R training orchestration and artifact finalization."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from bigdb_mvp.config import DoctorSettings
from bigdb_mvp.data.builder import DATASET_PATH, MANIFEST_PATH
from bigdb_mvp.data.manifest import Manifest
from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.doctor import DoctorReport, derive_doctor_report
from bigdb_mvp.gpu_probe import collect_live_snapshot
from bigdb_mvp.training_contracts import (
    BackendOutput,
    BlockedTrainingReceipt,
    TrainingBackend,
    TrainingRequest,
    TrainingState,
    XlmArtifact,
    XlmBackendError,
    XlmRow,
    XlmTrainingConfig,
    blocked_receipt,
)
from bigdb_mvp.xlm_artifacts import finalize_xlm_artifact, load_xlm_artifact

ARTIFACT_DIR: Final = Path(__file__).resolve().parents[2] / "artifacts/xlm"

__all__ = [
    "BackendOutput",
    "TrainingRequest",
    "XlmBackendError",
    "XlmRow",
    "load_xlm_artifact",
]


@dataclass(frozen=True, slots=True)
class XlmRunPaths:
    """Filesystem boundaries for one XLM-R training invocation."""

    dataset: Path = DATASET_PATH
    manifest: Path = MANIFEST_PATH
    artifact_dir: Path = ARTIFACT_DIR


@dataclass(frozen=True, slots=True)
class GpuTrainingReady:
    """Preflight result authorizing only the pinned CUDA0 configuration."""

    report: DoctorReport
    config: XlmTrainingConfig


XlmTrainingStart = GpuTrainingReady | BlockedTrainingReceipt


def _rows(dataset_path: Path) -> tuple[XlmRow, ...]:
    table = pq.read_table(dataset_path, columns=["text", "label", "language", "split"])
    return tuple(
        XlmRow(
            text=row["text"],
            label=int(row["label"] == "harmful"),
            language=row["language"],
            split=row["split"],
        )
        for row in table.to_pylist()
    )


def balanced_training_batches(
    dataset_path: Path, config: XlmTrainingConfig
) -> tuple[tuple[XlmRow, ...], ...]:
    """Create deterministic train-only batches for the selected languages."""
    rows = _rows(dataset_path)
    korean = [row for row in rows if row.split == "train" and row.language == "ko"]
    korean = sorted(
        korean,
        key=lambda row: hashlib.sha256(f"{config.seed}:{row.text}".encode()).digest(),
    )
    return tuple(
        tuple(korean[index : index + config.batch_size])
        for index in range(0, len(korean) - config.batch_size + 1, config.batch_size)
    )


def start_xlm_training(
    config: XlmTrainingConfig | None = None,
    report: DoctorReport | None = None,
) -> XlmTrainingStart:
    """Refuse CPU fallback and return an explicit GPU training gate outcome."""
    active_config = config if config is not None else XlmTrainingConfig()
    live_report = report
    if live_report is None:
        settings = DoctorSettings(device=0)
        live_report = derive_doctor_report(collect_live_snapshot(settings), settings)
    if not live_report.gpu_ready:
        return blocked_receipt(live_report, active_config)
    return GpuTrainingReady(report=live_report, config=active_config)


def run_xlm_training(
    *,
    report: DoctorReport,
    backend: TrainingBackend,
    config: XlmTrainingConfig | None = None,
    paths: XlmRunPaths | None = None,
) -> XlmArtifact:
    """Train, calibrate, evaluate once, and finalize a verified CUDA checkpoint."""
    active_paths = paths if paths is not None else XlmRunPaths()
    active_config = config if config is not None else XlmTrainingConfig()
    if not report.gpu_ready:
        return XlmArtifact.partial(
            state=TrainingState.BLOCKED, reason_code="gpu_blocked"
        )
    manifest = Manifest.model_validate_json(active_paths.manifest.read_bytes())
    if sha256_file(active_paths.dataset) != manifest.dataset_sha256:
        return XlmArtifact.partial(
            state=TrainingState.CUDA_ERROR, reason_code="dataset_digest_mismatch"
        )
    rows = _rows(active_paths.dataset)
    calibration = tuple(row for row in rows if row.split == "calibration")
    test = tuple(row for row in rows if row.split == "test")
    request = TrainingRequest(
        config=active_config,
        batches=balanced_training_batches(active_paths.dataset, active_config),
        calibration=calibration,
        test=test,
    )
    active_paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=active_paths.artifact_dir) as staging_name:
            output = backend.train(request, Path(staging_name))
            return finalize_xlm_artifact(
                output, request, manifest, active_paths.artifact_dir
            )
    except KeyboardInterrupt:
        return XlmArtifact.partial(
            state=TrainingState.CANCELLED, reason_code="training_cancelled"
        )
    except XlmBackendError as error:
        return XlmArtifact.partial(state=error.state, reason_code=error.reason_code)
