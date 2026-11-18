"""Finalization and live integrity loading for XLM-R artifacts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from pydantic import ValidationError

from bigdb_mvp.baseline_contracts import PredictionCount, SliceMetrics
from bigdb_mvp.baseline_metrics import binary_metrics, select_threshold
from bigdb_mvp.training_contracts import (
    SliceScoreStandardDeviation,
    TrainingState,
    XlmArtifact,
)
from bigdb_mvp.xlm_calibration import calibrated_probabilities, fit_temperature

if TYPE_CHECKING:
    from bigdb_mvp.data.manifest import Manifest
    from bigdb_mvp.training_contracts import BackendOutput, TrainingRequest, XlmRow

ARTIFACT_DIR: Final = Path(__file__).resolve().parents[2] / "artifacts/xlm"
MANIFEST_FILENAME: Final = "xlm_manifest.json"


def _tree_sha256(root: Path) -> str | None:
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _selected(
    rows: tuple[XlmRow, ...], scores: tuple[float, ...], language: str
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    pairs = tuple(
        (row.label, score)
        for row, score in zip(rows, scores, strict=True)
        if language in ("overall", row.language)
    )
    return tuple(label for label, _ in pairs), tuple(score for _, score in pairs)


def _metrics(
    rows: tuple[XlmRow, ...], scores: tuple[float, ...], threshold: float
) -> tuple[SliceMetrics, ...]:
    languages: tuple[Literal["overall", "ko"], ...] = ("overall", "ko")
    return tuple(
        SliceMetrics(
            language=language,
            metrics=binary_metrics(*_selected(rows, scores, language), threshold),
        )
        for language in languages
    )


def _variances(
    rows: tuple[XlmRow, ...], scores: tuple[float, ...]
) -> tuple[SliceScoreStandardDeviation, ...]:
    languages: tuple[Literal["overall", "ko"], ...] = ("overall", "ko")
    return tuple(
        SliceScoreStandardDeviation(
            language=language,
            value=float(np.std(_selected(rows, scores, language)[1])),
        )
        for language in languages
    )


def finalize_xlm_artifact(
    output: BackendOutput,
    request: TrainingRequest,
    manifest: Manifest,
    artifact_dir: Path,
) -> XlmArtifact:
    """Calibrate, evaluate once, and atomically expose complete model outputs."""
    calibration_labels = tuple(row.label for row in request.calibration)
    temperature = fit_temperature(calibration_labels, output.calibration_logits)
    calibration_scores = calibrated_probabilities(
        output.calibration_logits, temperature
    )
    threshold = select_threshold(calibration_labels, calibration_scores)
    calibrator_sha256 = hashlib.sha256(
        f"{temperature:.17g}:{threshold:.2f}".encode()
    ).hexdigest()
    test_scores = calibrated_probabilities(output.test_logits, temperature)
    checkpoint_dir = artifact_dir / "checkpoint"
    tokenizer_dir = artifact_dir / "tokenizer"
    _ = shutil.copytree(output.checkpoint_dir, checkpoint_dir)
    _ = shutil.copytree(output.tokenizer_dir, tokenizer_dir)
    checkpoint_sha256 = _tree_sha256(checkpoint_dir)
    tokenizer_sha256 = _tree_sha256(tokenizer_dir)
    if checkpoint_sha256 is None or tokenizer_sha256 is None:
        return XlmArtifact.partial(
            state=TrainingState.CUDA_ERROR, reason_code="incomplete_checkpoint"
        )
    variances = _variances(request.test, test_scores)
    artifact = XlmArtifact(
        state=TrainingState.COMPLETE,
        dataset_sha256=manifest.dataset_sha256,
        logical_digest=manifest.logical_digest,
        tokenizer_sha256=tokenizer_sha256,
        checkpoint_sha256=checkpoint_sha256,
        calibrator_sha256=calibrator_sha256,
        checkpoint_complete=True,
        checkpoint_digest_verified=_tree_sha256(checkpoint_dir) == checkpoint_sha256,
        calibrator_temperature=temperature,
        threshold=threshold,
        score_standard_deviation=variances[0].value,
        language_score_standard_deviations=variances,
        metrics=_metrics(request.test, test_scores, threshold),
        prediction_distribution=tuple(
            PredictionCount(
                label=label,
                count=sum((score >= threshold) == target for score in test_scores),
            )
            for label, target in (("clean", False), ("harmful", True))
        ),
    )
    _ = (artifact_dir / MANIFEST_FILENAME).write_text(
        artifact.model_dump_json(indent=2, exclude_computed_fields=True),
        encoding="utf-8",
    )
    return artifact


def load_xlm_artifact(artifact_dir: Path = ARTIFACT_DIR) -> XlmArtifact | None:
    """Return only an exact pinned eligible artifact with live digest agreement."""
    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        artifact = XlmArtifact.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValidationError):
        return None
    digest_verified = (
        _tree_sha256(artifact_dir / "checkpoint") == artifact.checkpoint_sha256
        and _tree_sha256(artifact_dir / "tokenizer") == artifact.tokenizer_sha256
    )
    verified = artifact.model_copy(
        update={"checkpoint_digest_verified": digest_verified}
    )
    return verified if verified.eligible_for_serving else None
