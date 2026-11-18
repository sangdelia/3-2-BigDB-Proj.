"""Compute source-aware EDA without exporting raw comments or annotations."""

from __future__ import annotations

from typing import Final

import polars as pl

from bigdb_mvp.data.models import SourceName
from bigdb_mvp.experiments.dataset_types import DatasetError
from bigdb_mvp.experiments.eda_contracts import (
    CorpusManifest,
    Count,
    LengthProfile,
    LengthStats,
    Method,
    MissingSummary,
    SourceLength,
    SourceSummary,
    SplitSummary,
)

SOURCE_NAMES: Final = {
    "nikl": "NIKL",
    "unsmile": "UnSmile",
    "kold": "KOLD",
    "beep": "BEEP",
    "kmhas": "K-MHaS",
    "khaters": "K-HATERS",
}
LENGTH_BOUNDS: Final = (10, 20, 50, 100, 200, 500, 1000)


def overlap_count(frame: pl.DataFrame, field: str) -> int:
    """Count keys shared across frozen partitions, not repeated rows."""
    return (
        frame.group_by(field)
        .agg(pl.col("split").n_unique().alias("splits"))
        .filter(pl.col("splits") > 1)
        .height
    )


def validate_corpus(frame: pl.DataFrame, manifest: CorpusManifest) -> None:
    """Fail the reporting boundary on incompatible or unsafe corpus state."""
    if frame.height != manifest.accepted_rows or frame.height == 0:
        raise DatasetError(reason="eda_row_count_mismatch")
    expected_sources = {source.source for source in manifest.sources}
    if set(frame["source"].to_list()) != expected_sources:
        raise DatasetError(reason="eda_source_mismatch")
    if frame["text_sha256"].n_unique() != frame.height:
        raise DatasetError(reason="eda_duplicate_final_text")
    if any(overlap_count(frame, key) for key in ("text_sha256", "group_id")):
        raise DatasetError(reason="eda_split_overlap")
    if frame.filter(~pl.col("label").is_in(["clean", "harmful"])).height:
        raise DatasetError(reason="eda_unknown_label")


def source_summaries(
    frame: pl.DataFrame,
    manifest: CorpusManifest,
) -> tuple[SourceSummary, ...]:
    """Keep source namespaces and annotation availability separate."""
    output: list[SourceSummary] = []
    for receipt in manifest.sources:
        part = frame.filter(pl.col("source") == receipt.source)
        output.append(
            SourceSummary(
                source=SourceName(receipt.source).value,
                name=SOURCE_NAMES[receipt.source],
                original_rows=receipt.raw_rows,
                parsed_rows=receipt.accepted_rows,
                final_rows=part.height,
                clean=part.filter(pl.col("label") == "clean").height,
                harmful=part.filter(pl.col("label") == "harmful").height,
                context_rows=part.filter(
                    pl.col("context_text").str.strip_chars() != ""
                ).height,
                target_annotated_rows=part.filter(
                    pl.col("target_labels").list.len() > 0
                ).height,
                expression_annotated_rows=part.filter(
                    pl.col("expression_labels").list.len() > 0
                ).height,
            )
        )
    return tuple(sorted(output, key=lambda item: item.final_rows, reverse=True))


def split_summaries(frame: pl.DataFrame) -> tuple[SplitSummary, ...]:
    """Use the existing frozen split, never repartition for presentation."""
    summaries: list[SplitSummary] = []
    for name in ("train", "validation", "test"):
        part = frame.filter(pl.col("split") == name)
        summaries.append(
            SplitSummary(
                split=name,
                count=part.height,
                clean=part.filter(pl.col("label") == "clean").height,
                harmful=part.filter(pl.col("label") == "harmful").height,
            )
        )
    return tuple(summaries)


def missing_profile(frame: pl.DataFrame) -> tuple[MissingSummary, ...]:
    """Empty source annotation lists are unavailable, not negative labels."""
    definitions = (
        (
            "text",
            "문장",
            "required",
            pl.col("text").is_null() | (pl.col("text").str.strip_chars() == ""),
            "빈 문장은 필수값 오류",
        ),
        (
            "label",
            "이진 라벨",
            "required",
            pl.col("label").is_null() | ~pl.col("label").is_in(["clean", "harmful"]),
            "라벨을 임의 추정하거나 대체하지 않음",
        ),
        (
            "source",
            "출처",
            "required",
            pl.col("source").is_null() | (pl.col("source") == ""),
            "출처 없는 행은 승인하지 않음",
        ),
        (
            "context_text",
            "제공된 문맥",
            "optional",
            pl.col("context_text").is_null()
            | (pl.col("context_text").str.strip_chars() == ""),
            "원천에 문맥이 없으면 그대로 보존",
        ),
        (
            "target_labels",
            "원천 대상 주석",
            "optional",
            pl.col("target_labels").is_null()
            | (pl.col("target_labels").list.len() == 0),
            "미제공 또는 비해당을 임의 대상 라벨로 채우지 않음",
        ),
        (
            "expression_labels",
            "원천 표현 주석",
            "optional",
            pl.col("expression_labels").is_null()
            | (pl.col("expression_labels").list.len() == 0),
            "빈 목록은 유해 표현이 없다는 정답으로 해석하지 않음",
        ),
    )
    result: list[MissingSummary] = []
    for field, label, kind, predicate, meaning in definitions:
        count = frame.filter(predicate).height
        result.append(
            MissingSummary.model_validate(
                {
                    "field": field,
                    "label": label,
                    "kind": kind,
                    "meaning": meaning,
                    "missing_rows": count,
                    "total_rows": frame.height,
                    "rate": count / frame.height,
                }
            )
        )
    return tuple(result)


def length_profile(frame: pl.DataFrame) -> LengthProfile:
    """Count Unicode code points of stored original strings, not tokens."""
    measured = frame.with_columns(pl.col("text").str.len_chars().alias("length"))
    stats = LengthStats.model_validate(
        measured.select(
            pl.len().alias("count"),
            pl.col("length").min().alias("min"),
            pl.col("length").median().alias("median"),
            pl.col("length").quantile(0.95, interpolation="nearest").alias("p95"),
            pl.col("length").max().alias("max"),
            pl.col("length").mean().alias("mean"),
        ).row(0, named=True)
    )
    histogram: list[Count] = []
    lower = 0
    for upper in LENGTH_BOUNDS:
        histogram.append(
            Count(
                label=f"{lower + 1}-{upper}자",
                count=measured.filter(
                    pl.col("length").is_between(lower + 1, upper)
                ).height,
            )
        )
        lower = upper
    histogram.append(
        Count(
            label=f"{lower}자 초과",
            count=measured.filter(pl.col("length") > lower).height,
        )
    )
    grouped = (
        measured.group_by("source")
        .agg(
            pl.len().alias("count"),
            pl.col("length").median().alias("median"),
            pl.col("length").quantile(0.95, interpolation="nearest").alias("p95"),
            pl.col("length").max().alias("max"),
        )
        .sort("count", descending=True)
    )
    return LengthProfile(
        unit=(
            "저장된 원문 Unicode code point 수(공백·구두점 포함), p95는 nearest 분위수"
        ),
        stats=stats,
        histogram=tuple(histogram),
        by_source=tuple(
            SourceLength.model_validate(row) for row in grouped.iter_rows(named=True)
        ),
    )


def category_counts(frame: pl.DataFrame, field: str) -> tuple[Count, ...]:
    """Count source-prefixed list entries without exporting source text."""
    values = frame.select(field).explode(field).drop_nulls().filter(pl.col(field) != "")
    grouped = (
        values.group_by(field)
        .len()
        .sort(["len", field], descending=[True, False])
        .head(20)
    )
    return tuple(
        Count.model_validate({"label": row[field], "count": row["len"]})
        for row in grouped.iter_rows(named=True)
    )


def retention_methods(frame: pl.DataFrame) -> tuple[Method, ...]:
    """Measure complete-case deletion only as a comparison, never apply it."""
    complete = frame.filter(
        (pl.col("context_text").str.strip_chars() != "")
        & (pl.col("target_labels").list.len() > 0)
        & (pl.col("expression_labels").list.len() > 0)
    ).height
    return (
        Method(
            id="preserve",
            title="선택 주석 보존",
            selected=True,
            start_rows=frame.height,
            retained_rows=frame.height,
            removed_rows=0,
            reason="이미 필수값·충돌·중복 처리를 통과한 문장과 원래 주석을 보존",
        ),
        Method(
            id="complete_case",
            title="선택 필드까지 완전 삭제",
            selected=False,
            start_rows=frame.height,
            retained_rows=complete,
            removed_rows=frame.height - complete,
            reason=(
                "문맥·대상·표현 주석을 모두 요구하는 비교용 정책. "
                "원천 구조 차이를 결측 오류로 취급하므로 미채택"
            ),
        ),
    )
