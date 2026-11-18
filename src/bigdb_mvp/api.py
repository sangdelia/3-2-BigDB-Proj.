"""Local Flask API for typed, non-persistent prediction."""

from __future__ import annotations

from typing import assert_never

from flask import Flask, Response, jsonify, request
from pydantic import ValidationError

from bigdb_mvp.contracts import PredictionRequest
from bigdb_mvp.predict import ModelUnavailableError, PredictionService


def _no_store(response: Response) -> Response:
    """Prevent browsers and intermediaries from retaining submitted text."""
    response.headers["Cache-Control"] = "no-store"
    return response


def create_app(
    service: PredictionService | ModelUnavailableError | None = None,
) -> Flask:
    """Create a localhost-only prediction surface with typed boundaries."""
    app = Flask(__name__)
    try:
        predictor = service if service is not None else PredictionService()
    except ModelUnavailableError as error:
        predictor = error

    def predict() -> Response:
        """Return one non-persistent prediction or a typed public error."""
        try:
            payload = request.get_data(cache=False)
            parsed = PredictionRequest.model_validate_json(payload)
        except ValidationError:
            response = _no_store(jsonify({"error": "invalid_request"}))
            response.status_code = 422
            return response
        match predictor:
            case PredictionService():
                result = predictor.predict(parsed).model_dump(mode="json")
                return _no_store(jsonify(result))
            case ModelUnavailableError():
                response = _no_store(jsonify({"error": "model_unavailable"}))
                response.status_code = 503
                return response
            case unreachable:
                assert_never(unreachable)

    _ = app.post("/api/predict")(predict)
    return app
