from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import bigdb_mvp.predict as prediction_module
from bigdb_mvp.contracts import ScoreEvidence
from bigdb_mvp.predict import ModelUnavailableError, PredictionService
from bigdb_mvp.training_contracts import XlmArtifact
from bigdb_mvp.xlm_inference import XlmPrediction


def test_prediction_service_prefers_eligible_xlm_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an exact eligible XLM-R result and a usable baseline fallback artifact.
    artifact = XlmArtifact.complete_for_test(calibrator_temperature=1.0)
    xlm_result = XlmPrediction(
        harmful_probability=0.91,
        threshold=0.5,
        artifact=artifact,
    )

    def eligible_xlm_prediction(_text: str, _artifact_dir: Path) -> XlmPrediction:
        return xlm_result

    monkeypatch.setattr(
        prediction_module,
        "predict_xlm_probability",
        eligible_xlm_prediction,
    )
    service = PredictionService()

    # When: a caller submits a harmful English sentence.
    result = service.predict_text("You are an idiot")

    # Then: XLM-R provenance and threshold determine the public prediction.
    assert result.label == "harmful"
    assert result.harmful_probability == 0.91
    assert result.model_kind == "xlm_roberta"
    assert result.model_digest == artifact.checkpoint_sha256
    score = result.evidence[0]
    assert isinstance(score, ScoreEvidence)
    assert score.kind == "model_score"
    assert score.harmful_probability == 0.91
    assert score.threshold == 0.5
    assert score.decision == "above_threshold"


def test_prediction_service_falls_back_when_xlm_artifact_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the XLM-R serving boundary rejected a corrupt or incomplete artifact.

    def rejected_xlm_prediction(_text: str, _artifact_dir: Path) -> None:
        return None

    monkeypatch.setattr(
        prediction_module,
        "predict_xlm_probability",
        rejected_xlm_prediction,
    )
    service = PredictionService()

    # When: the service predicts using the remaining verified local artifact.
    result = service.predict_text("You are an idiot")

    # Then: the digest-verified CPU baseline is explicit rather than fabricated XLM-R.
    assert result.model_kind == "char_tfidf_logistic"
    assert "Eligible XLM-R artifact is unavailable" in result.limitations[0]


def test_prediction_service_returns_baseline_provenance_and_suggestion() -> None:
    # Given: the checked-in local CPU baseline artifact and a harmful English input.
    service = PredictionService()

    # When: the service predicts through the real digest-verified baseline.
    result = service.predict_text("You are an idiot")

    # Then: the public result has provenance and an exact curated substitution.
    assert result.model_kind == "char_tfidf_logistic"
    assert len(result.model_digest) == 64
    assert result.language == "en"
    assert 0.0 <= result.harmful_probability <= 1.0
    assert result.label == "harmful"
    score = result.evidence[0]
    assert isinstance(score, ScoreEvidence)
    assert score.kind == "model_score"
    assert score.decision == "above_threshold"
    assert result.suggestions[0].strategy == "lexicon_substitution"
    assert result.suggestions[0].source == "curated_rule"


def test_prediction_service_uses_neutral_template_for_unmatched_harmful_text() -> None:
    # Given: a real baseline input that is harmful but has no curated lexicon match.
    service = PredictionService()

    # When: the public inference surface classifies the sentence.
    result = service.predict_text("You should die")

    # Then: it offers the approved English fallback without a fabricated span.
    assert result.label == "harmful"
    suggestion = result.suggestions[0]
    assert suggestion.strategy == "neutral_template"
    assert suggestion.start is None
    assert suggestion.end is None
    assert suggestion.source == "curated_rule"


def test_prediction_service_keeps_suggestions_empty_for_clean_text() -> None:
    # Given: a real Korean baseline input that is classified as clean.
    service = PredictionService()

    # When: the public inference surface classifies the sentence.
    result = service.predict_text("오늘 날씨가 좋습니다.")

    # Then: it returns no expression-rewrite proposal.
    assert result.label == "clean"
    assert result.suggestions == ()


def test_prediction_service_refuses_missing_artifact(tmp_path: Path) -> None:
    # Given: an empty artifact location rather than a trained model.
    missing = tmp_path / "missing"

    # When / Then: serving fails explicitly rather than fabricating a prediction.
    with pytest.raises(ModelUnavailableError, match="model_unavailable"):
        _ = PredictionService(artifact_dir=missing)


def test_cli_exposes_predict_and_localhost_serve_commands() -> None:
    # Given: the installed CLI entry point.
    entrypoint = str(Path(sys.executable).with_name("bigdb-mvp"))

    # When: a user asks for the two public prediction surfaces.
    predict = subprocess.run(  # noqa: S603
        (entrypoint, "predict", "--text", "You are an idiot", "--json"),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    serve = subprocess.run(  # noqa: S603
        (entrypoint, "serve", "--help"),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    # Then: CLI JSON and the localhost server command remain addressable.
    assert predict.returncode == 0
    assert '"model_kind"' in predict.stdout
    assert serve.returncode == 0
    assert "--host" not in serve.stdout


def test_cli_does_not_offer_a_public_host_override() -> None:
    # Given: the installed local-only serving command.
    entrypoint = str(Path(sys.executable).with_name("bigdb-mvp"))
    public_host = ".".join(("0",) * 4)

    # When: a caller tries to request a non-loopback binding.
    serve = subprocess.run(  # noqa: S603
        (entrypoint, "serve", "--host", public_host, "--help"),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    # Then: the CLI rejects it instead of exposing submitted text on a network.
    assert serve.returncode != 0
