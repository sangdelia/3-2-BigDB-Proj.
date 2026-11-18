from __future__ import annotations

import pytest

from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.korean_model import (
    KoreanModelService,
    KoreanPrediction,
    KoreanRequest,
    load_korean_model,
)
from bigdb_mvp.korean_web import create_korean_app


@pytest.fixture(scope="module")
def model_service() -> KoreanModelService:
    return load_korean_model()


def test_real_api_scores_match_direct_model(model_service: KoreanModelService) -> None:
    client = create_korean_app(model_service).test_client()
    text = "🙂 너는\t\n정말  멍청한 바보야."
    response = client.post("/api/korean/predict", json={"text": text})
    assert response.status_code == 200
    result = KoreanPrediction.model_validate_json(response.text)
    assert result == model_service.predict(KoreanRequest(text=text))
    assert result.label == "harmful"
    assert result.model_sha256 == model_service.model_sha256
    assert response.headers["Cache-Control"] == "no-store"
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "Set-Cookie" not in response.headers


@pytest.mark.parametrize(
    "payload",
    [
        '{"text":""}',
        '{"text":"only english"}',
        '{"text":"  "}',
        '{"text":"한글","extra":true}',
        '{"text":5}',
        "not json",
    ],
)
def test_api_rejects_invalid_input_without_echo(
    model_service: KoreanModelService, payload: str
) -> None:
    response = (
        create_korean_app(model_service)
        .test_client()
        .post("/api/korean/predict", data=payload, content_type="application/json")
    )
    assert response.status_code == 422
    assert response.get_json() == {"error": "invalid_request"}
    assert response.headers["Cache-Control"] == "no-store"


def test_api_body_length_media_and_origin(model_service: KoreanModelService) -> None:
    client = create_korean_app(model_service).test_client()
    response = client.post("/api/korean/predict", json={"text": "한" * 1001})
    assert response.status_code == 422
    response = client.post(
        "/api/korean/predict", data="x" * 17000, content_type="application/json"
    )
    assert response.status_code == 413
    assert response.get_json() == {"error": "request_too_large"}
    assert response.headers["Cache-Control"] == "no-store"
    response = client.post("/api/korean/predict", data="한글")
    assert response.status_code == 415
    response = client.post(
        "/api/korean/predict",
        json={"text": "한글"},
        headers={"Origin": "https://external.invalid"},
    )
    assert response.status_code == 403
    response = client.post(
        "/api/korean/predict",
        json={"text": "한글"},
        headers={"Origin": "http://localhost"},
    )
    assert response.status_code == 200


def test_model_unavailable_keeps_page_and_safe_error() -> None:
    reason = "model_unavailable"
    client = create_korean_app(ExperimentError(reason)).test_client()
    assert client.get("/").status_code == 200
    response = client.post("/api/korean/predict", json={"text": "안녕하세요"})
    assert response.status_code == 503
    assert response.get_json() == {"error": "model_unavailable"}
    assert "안녕하세요" not in response.text
    assert response.headers["Cache-Control"] == "no-store"


def test_api_abstains_when_no_features_are_recognized(
    model_service: KoreanModelService,
) -> None:
    client = create_korean_app(model_service).test_client()
    response = client.post("/api/korean/predict", json={"text": "힣힣힣"})
    result = KoreanPrediction.model_validate_json(response.text)
    assert result.label == "review"
    assert result.recognized_feature_count == 0
    assert result.review_reason == "no_recognized_features"


def test_https_tunnel_origin_requires_explicit_proxy_trust(
    model_service: KoreanModelService,
) -> None:
    base = "http://demo.trycloudflare.com"
    headers = {
        "Origin": "https://demo.trycloudflare.com",
        "X-Forwarded-Proto": "https",
    }
    client = create_korean_app(model_service).test_client()
    assert (
        client.post(
            "/api/korean/predict",
            base_url=base,
            headers=headers,
            json={"text": "안녕하세요"},
        ).status_code
        == 403
    )
    client = create_korean_app(model_service, trust_proxy_https=True).test_client()
    assert (
        client.post(
            "/api/korean/predict",
            base_url=base,
            headers=headers,
            json={"text": "안녕하세요"},
        ).status_code
        == 200
    )
    headers["Origin"] = "https://external.invalid"
    headers["X-Forwarded-Host"] = "external.invalid"
    assert (
        client.post(
            "/api/korean/predict",
            base_url=base,
            headers=headers,
            json={"text": "안녕하세요"},
        ).status_code
        == 403
    )


def test_page_and_assets_are_local_and_serve(model_service: KoreanModelService) -> None:
    client = create_korean_app(model_service).test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert 'lang="ko"' in response.text
    assert 'id="message"' in response.text
    assert 'id="candidate-list"' in response.text
    assert "https://" not in response.text
    for path in ("/static/app.css", "/static/app.js", "/static/icon.svg"):
        with client.get(path) as asset:
            assert asset.status_code == 200
            assert asset.headers["Cache-Control"] == "no-store"
