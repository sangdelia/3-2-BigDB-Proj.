"""Frozen-model score diagnosis using authored examples and validation aggregates."""

from __future__ import annotations

import csv
import json
import re
import time
from statistics import fmean, quantiles
from typing import TYPE_CHECKING, ClassVar, Final

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.experiments.classification_contracts import ExperimentError, Row
from bigdb_mvp.korean_model import ROOT, KoreanRequest, load_korean_model

if TYPE_CHECKING:
    from pathlib import Path

LOW_SCORE: Final = 5
VERY_LOW_SCORE: Final = 10

KEYWORDS: Final = re.compile(r"시발|씨발|ㅅㅂ|ㅆㅂ|병신|지랄|새끼|좆|꺼져|멍청|바보")
CASES: Final = (
    (
        "짧은 욕설·모욕",
        (
            "시발",
            "씨발",
            "시발롬아",
            "시발놈아",
            "씨발놈아",
            "병신",
            "병신아",
            "지랄",
            "지랄하지마",
            "개새끼",
            "개새끼야",
            "좆같네",
            "꺼져",
            "닥쳐",
            "바보",
            "바보야",
            "멍청이",
            "ㅅㅂ",
            "ㅆㅂ",
        ),
    ),
    (
        "문장형 욕설·모욕",
        (
            "너는 정말 멍청한 바보야.",
            "너 같은 병신은 꺼져라.",
            "씨발 진짜 짜증나네.",
            "시발놈아 다시는 오지 마.",
            "이 개새끼야 당장 꺼져.",
            "닥쳐, 너랑 말하기 싫어.",
            "너는 왜 이렇게 멍청하냐.",
            "지랄하지 말고 조용히 해.",
        ),
    ),
    (
        "공백·오타 변형",
        (
            " 시발롬아",
            "시발 롬아",
            "시발롬아 ",
            " 씨발놈아",
            "씨 발 놈 아",
            "ㅅㅣ발",
            "시1발",
            "씨@발",
            "시발!!",
            " 바보",
            "바보 ",
            "병 신 아",
        ),
    ),
    (
        "평상문",
        (
            "안녕하세요",
            "감사합니다",
            "좋은 하루 보내세요. 감사합니다.",
            "안녕",
            "네",
            "응",
            "오늘 날씨가 좋네요.",
            "점심 먹고 산책하러 갈까요?",
            "내일 오전 열 시에 만나요.",
            "회의 자료를 확인해 주세요.",
            "수고하셨습니다.",
            "도와주셔서 감사합니다.",
            "오늘도 행복한 하루 되세요.",
            "정말 친절하시네요.",
            "저녁 메뉴는 무엇인가요?",
            "잠시 후 다시 연락드리겠습니다.",
            "책을 읽고 음악을 들었습니다.",
            "정말 맛있네요.",
            "잘 지내고 계신가요?",
            "괜찮아요. 걱정하지 마세요.",
            "죄송합니다. 제가 착각했어요.",
            "주문한 상품이 도착했습니다.",
            "한국어를 함께 공부해요.",
            "외국인 친구와 함께 한국어를 공부합니다.",
            "여성과 남성 모두 존중받아야 합니다.",
            "종교가 달라도 서로 존중해야 합니다.",
        ),
    ),
    (
        "인용·부정·다른 문맥",
        (
            "외국인이라는 이유로 차별하면 안 됩니다.",
            "그가 '멍청한 바보야'라고 말한 것은 잘못입니다.",
            "욕설을 사용하지 마세요.",
            "'씨발'은 상대에게 쓰면 안 되는 욕설입니다.",
            "너는 바보가 아니야.",
            "나는 그 사람을 멍청하다고 생각하지 않는다.",
            "그 사람이 나에게 '병신아'라고 해서 속상했어요.",
            "강아지가 새끼를 낳았습니다.",
        ),
    ),
    ("미등록 특징 대조", ("힣힣힣",)),
)


class Example(BaseModel):
    """Authored categories describe intent and are not human-annotated gold labels."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    category: str
    text: str
    score: float
    label: str
    recognized_features: int
    highlighted_candidates: int


class ScoredRow(BaseModel):
    """Keep raw validation text out of all exported diagnostic reports."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    source: str
    label: str
    text_sha256: str
    score: float
    recognized_features: int
    profanity_annotation: bool
    keyword_match: bool


class ProbeRow(Row):
    """Preserve source expression annotations for distinct diagnostic subsets."""

    expression_labels: tuple[str, ...]


class Distribution(BaseModel):
    """Ranges and quantiles measure only the named finite collection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: str
    count: int
    minimum: float
    p05: float
    p25: float
    median: float
    p75: float
    p95: float
    maximum: float
    mean: float
    zero_features: int
    below_1: int
    below_5: int
    below_10: int
    harmful_predictions: int


def distribution(
    name: str,
    scores: tuple[float, ...],
    feature_counts: tuple[int, ...],
    threshold: float,
) -> Distribution:
    """Compute diagnostics without choosing a new threshold or changing weights."""
    if not scores or len(scores) != len(feature_counts):
        reason = "invalid_score_distribution_input"
        raise ExperimentError(reason)
    percentiles = (
        quantiles(scores, n=100, method="inclusive")
        if len(scores) > 1
        else [fmean(scores)] * 99
    )
    return Distribution(
        name=name,
        count=len(scores),
        minimum=min(scores),
        maximum=max(scores),
        p05=percentiles[4],
        p25=percentiles[24],
        median=percentiles[49],
        p75=percentiles[74],
        p95=percentiles[94],
        mean=fmean(scores),
        zero_features=sum(n == 0 for n in feature_counts),
        below_1=sum(s < 1 for s in scores),
        below_5=sum(s < LOW_SCORE for s in scores),
        below_10=sum(s < VERY_LOW_SCORE for s in scores),
        harmful_predictions=sum(s >= threshold for s in scores),
    )


def run_probe(report_root: Path = ROOT / "reports/korean_v2/score_probe") -> Path:
    """Export authored cases and text-free validation scores of the frozen model."""
    started = time.perf_counter()
    service = load_korean_model()
    threshold = service.model.threshold * 100
    examples: list[Example] = []
    for category, texts in CASES:
        for text in texts:
            request = KoreanRequest(text=text)
            matrix = service.model.vectorizer.transform((request.text,))
            recognized = int(matrix.getnnz(axis=1)[0])
            prediction = service.predict(request)
            examples.append(
                Example(
                    category=category,
                    text=text,
                    score=prediction.harmful_score,
                    label=prediction.label,
                    recognized_features=recognized,
                    highlighted_candidates=sum(
                        c.highlight_as_harmful_candidate for c in prediction.candidates
                    ),
                )
            )
    raw = pl.read_parquet(ROOT / "data/curated/korean_v2/corpus.parquet").filter(
        pl.col("split") == "validation"
    )
    rows = tuple(ProbeRow.model_validate(row) for row in raw.iter_rows(named=True))
    scored: list[ScoredRow] = []
    for start in range(0, len(rows), 512):
        batch = rows[start : start + 512]
        matrix = service.model.vectorizer.transform(tuple(r.text for r in batch))
        scores = service.model.classifier.predict_proba(matrix)[:, 1]
        counts = matrix.getnnz(axis=1)
        scored.extend(
            ScoredRow(
                source=row.source,
                label=row.label,
                text_sha256=row.text_sha256,
                score=float(score) * 100,
                recognized_features=int(count),
                profanity_annotation="unsmile:악플/욕설" in row.expression_labels,
                keyword_match=KEYWORDS.search(row.text) is not None,
            )
            for row, score, count in zip(batch, scores, counts, strict=True)
        )
    distributions: list[Distribution] = []
    for category, _texts in CASES:
        selected = tuple(e for e in examples if e.category == category)
        distributions.append(
            distribution(
                "작성 예시:" + category,
                tuple(e.score for e in selected),
                tuple(e.recognized_features for e in selected),
                threshold,
            )
        )
    for name, subset in (
        ("validation:clean", tuple(r for r in scored if r.label == "clean")),
        ("validation:harmful", tuple(r for r in scored if r.label == "harmful")),
        (
            "validation:UnSmile 악플/욕설 주석",
            tuple(r for r in scored if r.profanity_annotation),
        ),
        (
            "validation:harmful+진단 키워드",
            tuple(r for r in scored if r.label == "harmful" and r.keyword_match),
        ),
        (
            "validation:clean+진단 키워드",
            tuple(r for r in scored if r.label == "clean" and r.keyword_match),
        ),
    ):
        if subset:
            distributions.append(
                distribution(
                    name,
                    tuple(r.score for r in subset),
                    tuple(r.recognized_features for r in subset),
                    threshold,
                )
            )
    report_root.mkdir(parents=True, exist_ok=True)
    for filename, records in (
        ("authored_examples.csv", examples),
        ("distributions.csv", distributions),
    ):
        with (report_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].model_dump()))
            writer.writeheader()
            writer.writerows(record.model_dump() for record in records)
    pl.DataFrame([r.model_dump() for r in scored]).write_parquet(
        report_root / "validation_scores.parquet"
    )
    result = {
        "model_sha256": service.model_sha256,
        "dataset_sha256": service.dataset_sha256,
        "threshold_score": threshold,
        "authored_count": len(examples),
        "validation_count": len(rows),
        "seconds": time.perf_counter() - started,
        "note": (
            "Frozen-model diagnostic; authored categories and keyword matching "
            "are not gold labels. Source clean/harmful semantics differ. "
            "No test tuning, retraining or runtime model change."
        ),
        "examples": [e.model_dump() for e in examples],
        "distributions": [d.model_dump() for d in distributions],
    }
    output = report_root / "results.json"
    _ = output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    typer.echo(str(run_probe()))
