"""Record fixed authored demo cases, never service submissions or corpus text."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar, Final

import typer
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.korean_model import (
    ROOT,
    KoreanPrediction,
    KoreanRequest,
    load_korean_model,
)

if TYPE_CHECKING:
    from pathlib import Path

CASES: Final = (
    ("인사", "좋은 하루 보내세요. 감사합니다."),
    ("직접 모욕", "너는 정말 멍청한 바보야."),
    ("차별 반대", "외국인이라는 이유로 차별하면 안 됩니다."),
    ("인용 설명", "그 표현은 인용한 것이고 상대를 모욕하려는 뜻이 아닙니다."),
    ("공백·이모지", "🙂 너는\t\n정말  멍청한 바보야."),
    ("반복 표현", "멍청한 바보야 멍청한 바보야"),
    ("따옴표 인용", "그가 '멍청한 바보야'라고 말한 것은 잘못입니다."),
    ("대상 단어 포함", "외국인 친구와 함께 한국어를 공부합니다."),
    ("미등록 특징", "힣힣힣"),
    ("HTML 형태", "<script>alert('안녕')</script> 좋은 하루 보내세요."),
)


class DemoResult(BaseModel):
    """Authored input and observed inference, without an invented gold label."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    case: str
    authored_input: str
    predict_seconds: float
    prediction: KoreanPrediction


class DemoReport(BaseModel):
    """Smoke observations are separate from held-out accuracy measurements."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    purpose: str = (
        "Authored demo inputs; not a labeled benchmark or span accuracy test."
    )
    load_seconds: float
    cases: tuple[DemoResult, ...]


def run_demo(report_root: Path = ROOT / "reports/korean_v2") -> Path:
    """Save only built-in authored examples using the actual verified model."""
    started = time.perf_counter()
    service = load_korean_model()
    load_seconds = time.perf_counter() - started
    results: list[DemoResult] = []
    for case, text in CASES:
        started = time.perf_counter()
        prediction = service.predict(KoreanRequest(text=text))
        results.append(
            DemoResult(
                case=case,
                authored_input=text,
                predict_seconds=time.perf_counter() - started,
                prediction=prediction,
            )
        )
    report_root.mkdir(parents=True, exist_ok=True)
    output = report_root / "model_demo.json"
    _ = output.write_text(
        DemoReport(load_seconds=load_seconds, cases=tuple(results)).model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    typer.echo(str(run_demo()))
