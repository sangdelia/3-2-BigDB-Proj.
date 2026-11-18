"""Digest-gated local XLM-R inference for the prediction surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bigdb_mvp.xlm_calibration import sigmoid
from bigdb_mvp.xlm_training import ARTIFACT_DIR, load_xlm_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from bigdb_mvp.training_contracts import XlmArtifact


@dataclass(frozen=True, slots=True)
class XlmPrediction:
    """One calibrated harmful probability from an eligible pinned artifact."""

    harmful_probability: float
    threshold: float
    artifact: XlmArtifact


def predict_xlm_probability(
    text: str, artifact_dir: Path = ARTIFACT_DIR
) -> XlmPrediction | None:
    """Score text only when the local XLM-R artifact passes every serving gate."""
    artifact = load_xlm_artifact(artifact_dir)
    if (
        artifact is None
        or artifact.calibrator_temperature is None
        or artifact.threshold is None
    ):
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(
            artifact_dir / "checkpoint"
        ).to(artifact.config.device)
    except OSError:
        return None
    except RuntimeError as error:
        if "cuda" not in str(error).lower():
            raise
        return None
    _ = model.eval()
    _ = model.requires_grad_(requires_grad=False)
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=artifact.config.max_length,
        return_tensors="pt",
    ).to(artifact.config.device)
    logits = model(**encoded).logits
    harmful_logit = float((logits[:, 1] - logits[:, 0]).cpu().tolist()[0])
    return XlmPrediction(
        harmful_probability=sigmoid(harmful_logit / artifact.calibrator_temperature),
        threshold=artifact.threshold,
        artifact=artifact,
    )
