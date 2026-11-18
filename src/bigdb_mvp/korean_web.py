"""Local Korean inference API and demo, without storing submitted messages."""

from __future__ import annotations

from typing import ClassVar, Final, Literal, assert_never

from flask import Flask, Response, jsonify, render_template, request
from pydantic import BaseModel, ConfigDict, ValidationError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.khaters_score import ScoreModel
from bigdb_mvp.khaters_score import predict as predict_ratings
from bigdb_mvp.khaters_score_artifact import load_score_model
from bigdb_mvp.korean_model import (
    ROOT,
    KoreanModelService,
    KoreanRequest,
    load_korean_model,
)

MAX_BODY: Final = 16384


class _Selection(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    method: Literal[
        "char_tfidf", "char_2_5_60000", "char_3_5_200000", "char_2_5_200000"
    ]


def _load_demo_model() -> KoreanModelService:
    selection_path = ROOT / "reports/korean_short_v1/selection.json"
    if not selection_path.exists():
        return load_korean_model()
    selected = _Selection.model_validate_json(selection_path.read_bytes())
    if selected.method == "char_tfidf":
        return load_korean_model()
    return load_korean_model(
        ROOT / "artifacts/korean_short_v1",
        ROOT / "reports/korean_short_v1/metrics.json",
        method=selected.method,
    )


def _predict(
    predictor: KoreanModelService | ExperimentError, score_model: ScoreModel | None
) -> Response:
    """Validate at the boundary and return real model scores and evidence."""
    if request.mimetype != "application/json":
        response = jsonify({"error": "json_required"})
        response.status_code = 415
        return response
    origin = request.headers.get("Origin")
    if origin is not None and origin != request.host_url.rstrip("/"):
        response = jsonify({"error": "origin_not_allowed"})
        response.status_code = 403
        return response
    try:
        parsed = KoreanRequest.model_validate_json(request.get_data(cache=False))
    except ValidationError:
        response = jsonify({"error": "invalid_request"})
        response.status_code = 422
        return response
    match predictor:
        case KoreanModelService():
            try:
                result = predictor.predict(parsed)
                if score_model is not None:
                    result = result.model_copy(
                        update={
                            "rating_scores": predict_ratings(score_model, parsed.text),
                        }
                    )
            except ExperimentError:
                response = jsonify({"error": "analysis_unavailable"})
                response.status_code = 503
                return response
            return jsonify(result.model_dump(mode="json"))
        case ExperimentError():
            response = jsonify({"error": "model_unavailable"})
            response.status_code = 503
            return response
        case unreachable:
            assert_never(unreachable)


def create_korean_app(
    service: KoreanModelService | ExperimentError | None = None,
    *,
    trust_proxy_https: bool = False,
    score_model: ScoreModel | None = None,
) -> Flask:
    """Load one verified model and expose same-origin, no-store demo routes."""
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY
    if trust_proxy_https:
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=0, x_proto=1, x_host=0, x_port=0, x_prefix=0
        )
    try:
        predictor = service if service is not None else _load_demo_model()
    except (OSError, ValidationError, ExperimentError):
        predictor = ExperimentError("model_unavailable")
    scorer = score_model
    if service is None and scorer is None:
        try:
            scorer = load_score_model(ROOT)
        except (OSError, ValidationError, ExperimentError):
            scorer = None

    def secure(response: Response) -> Response:
        """Apply privacy and script isolation to success and error responses."""
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    def index() -> Response:
        return Response(render_template("index.html"), mimetype="text/html")

    def predict() -> Response:
        return _predict(predictor, scorer)

    def too_large(_error: RequestEntityTooLarge) -> Response:
        response = jsonify({"error": "request_too_large"})
        response.status_code = 413
        return response

    _ = app.after_request(secure)
    _ = app.get("/")(index)
    _ = app.post("/api/korean/predict")(predict)
    app.register_error_handler(RequestEntityTooLarge, too_large)
    return app


if __name__ == "__main__":
    create_korean_app(trust_proxy_https=True).run(
        host="127.0.0.1", port=8086, debug=False, use_reloader=False
    )
