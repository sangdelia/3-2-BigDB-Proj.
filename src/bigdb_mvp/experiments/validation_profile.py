"""Aggregate service-aware validation errors without exposing source sentences."""

from __future__ import annotations

from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.experiments.classification_metrics import metrics

LONG_TEXT_START: Final = 201


class Observation(BaseModel):
    """Retain only prediction metadata, never original text or source annotations."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    source: str = Field(min_length=1)
    label: Literal["clean", "harmful"]
    length: int = Field(ge=1)
    score: float = Field(ge=0, le=1)
    feature_count: int = Field(ge=0)
    accepted_input: bool


class SliceKey(BaseModel):
    """Identify a population by one fixed grouping dimension."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    dimension: Literal["overall", "source", "length"]
    name: str


class DecisionStats(BaseModel):
    """Expose denominators and leave absent-class error rates undefined."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    count: int
    clean_count: int
    harmful_count: int
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    macro_f1: float
    clean_false_positive_rate: float | None
    harmful_miss_rate: float | None
    harmful_recall: float | None


class SliceAudit(BaseModel):
    """Keep rejected input, review and decided populations separate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    key: SliceKey
    total: int
    rejected_input: int
    accepted_input: int
    review: int
    review_clean: int
    review_harmful: int
    decided: int
    coverage: float | None
    decision_stats: DecisionStats | None


def decision_stats(
    rows: tuple[Observation, ...], threshold: float
) -> DecisionStats | None:
    """Measure only accepted inputs with recognized features."""
    if not rows:
        return None
    result = metrics(
        tuple(int(row.label == "harmful") for row in rows),
        tuple(row.score for row in rows),
        threshold,
    )
    clean = result.tn + result.fp
    harmful = result.tp + result.fn
    return DecisionStats(
        count=result.count,
        clean_count=clean,
        harmful_count=harmful,
        tn=result.tn,
        fp=result.fp,
        fn=result.fn,
        tp=result.tp,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        clean_false_positive_rate=result.fp / clean if clean else None,
        harmful_miss_rate=result.fn / harmful if harmful else None,
        harmful_recall=result.tp / harmful if harmful else None,
    )


def summarize(
    rows: tuple[Observation, ...], threshold: float, key: SliceKey
) -> SliceAudit:
    """Apply the live input and zero-feature review policies before scoring."""
    accepted = tuple(row for row in rows if row.accepted_input)
    review = tuple(row for row in accepted if row.feature_count == 0)
    decided = tuple(row for row in accepted if row.feature_count > 0)
    return SliceAudit(
        key=key,
        total=len(rows),
        rejected_input=len(rows) - len(accepted),
        accepted_input=len(accepted),
        review=len(review),
        review_clean=sum(row.label == "clean" for row in review),
        review_harmful=sum(row.label == "harmful" for row in review),
        decided=len(decided),
        coverage=len(decided) / len(accepted) if accepted else None,
        decision_stats=decision_stats(decided, threshold),
    )


def profile(rows: tuple[Observation, ...], threshold: float) -> tuple[SliceAudit, ...]:
    """Partition the same observations by source and fixed character ranges."""
    slices = [summarize(rows, threshold, SliceKey(dimension="overall", name="all"))]
    slices.extend(
        summarize(
            tuple(row for row in rows if row.source == source),
            threshold,
            SliceKey(dimension="source", name=source),
        )
        for source in sorted({row.source for row in rows})
    )
    for lower, upper in ((1, 10), (11, 20), (21, 50), (51, 100), (101, 200)):
        slices.append(
            summarize(
                tuple(row for row in rows if lower <= row.length <= upper),
                threshold,
                SliceKey(dimension="length", name=f"{lower}-{upper}"),
            )
        )
    slices.append(
        summarize(
            tuple(row for row in rows if row.length >= LONG_TEXT_START),
            threshold,
            SliceKey(dimension="length", name=f"{LONG_TEXT_START}+"),
        )
    )
    return tuple(slices)
