from __future__ import annotations

from bigdb_mvp.api import create_app
from bigdb_mvp.contracts import PredictionResponse
from bigdb_mvp.predict import ModelUnavailableError, PredictionService


def test_api_predict_sets_no_store_and_returns_schema_valid_result() -> None:
    # Given: the localhost Flask app using the real CPU baseline.
    app = create_app(PredictionService())
    client = app.test_client()

    # When: a Korean harmful request is sent to the actual HTTP route.
    response = client.post("/api/predict", json={"text": "너는 바보다"})

    # Then: the result is JSON with the privacy cache-control contract.
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = PredictionResponse.model_validate_json(response.text)
    assert payload.language == "ko"
    assert payload.label in {"clean", "harmful"}
    assert isinstance(payload.suggestions, tuple)


def test_api_rejects_empty_oversized_and_html_inputs() -> None:
    # Given: the actual local API boundary.
    app = create_app(PredictionService())
    client = app.test_client()

    # When / Then: all prohibited input forms receive deterministic client errors.
    for text in ("", "x" * 2001, "<b>not accepted</b>"):
        response = client.post("/api/predict", json={"text": text})
        assert response.status_code == 422
        assert response.headers["Cache-Control"] == "no-store"


def test_api_returns_503_without_a_verified_model() -> None:
    # Given: a server started with a safe model-unavailable state.
    app = create_app(ModelUnavailableError(reason_code="model_unavailable"))
    client = app.test_client()

    # When: a valid request reaches the prediction route.
    response = client.post("/api/predict", json={"text": "ordinary sentence"})

    # Then: the model failure is public as 503 without echoing submitted text.
    assert response.status_code == 503
    assert response.headers["Cache-Control"] == "no-store"
    assert response.get_json() == {"error": "model_unavailable"}
