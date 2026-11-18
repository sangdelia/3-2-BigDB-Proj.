"""Aggregate-only contracts for the Korean preprocessing presentation."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class Aggregate(BaseModel):
    """Immutable presentation data; never contains comment text."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class SourceReceipt(BaseModel):
    """Read only the inventory fields used in the existing manifest."""

    source: str
    raw_rows: int
    accepted_rows: int


class CorpusManifest(BaseModel):
    """Parse the existing corpus manifest without changing its contract."""

    dataset_sha256: str
    raw_rows: int
    parsed_rows: int
    accepted_rows: int
    sources: tuple[SourceReceipt, ...]
    dedup_exclusions: dict[str, int]


class Count(Aggregate):
    """A frequency with its original category name."""

    label: str
    count: int = Field(ge=0)


class SourceSummary(Aggregate):
    """Counts of representative rows, distinct from merged provenance."""

    source: str
    name: str
    original_rows: int
    parsed_rows: int
    final_rows: int
    clean: int
    harmful: int
    context_rows: int
    target_annotated_rows: int
    expression_annotated_rows: int


class SplitSummary(Aggregate):
    """Class composition of one frozen internal partition."""

    split: str
    count: int
    clean: int
    harmful: int


class MissingSummary(Aggregate):
    """Separate required missing values from unavailable source annotations."""

    field: str
    label: str
    missing_rows: int
    total_rows: int
    rate: float = Field(ge=0, le=1)
    kind: Literal["required", "optional"]
    meaning: str


class LengthStats(Aggregate):
    """Unicode code-point length statistics with nearest quantiles."""

    count: int
    min: int
    median: float
    p95: float
    max: int
    mean: float


class SourceLength(Aggregate):
    """Comparable lengths for one representative source."""

    source: str
    count: int
    median: float
    p95: float
    max: int


class LengthProfile(Aggregate):
    """A histogram and its precise length definition."""

    unit: str
    stats: LengthStats
    histogram: tuple[Count, ...]
    by_source: tuple[SourceLength, ...]


class Method(Aggregate):
    """A measured retention comparison on the same final corpus."""

    id: str
    title: str
    selected: bool
    start_rows: int
    retained_rows: int
    removed_rows: int
    reason: str


class Stage(Aggregate):
    """One existing preprocessing decision with an explicit denominator."""

    id: str
    label: str
    input_rows: int
    removed_rows: int
    output_rows: int
    description: str


class Summary(Aggregate):
    """Observed corpus counts and current leakage checks."""

    raw_rows: int
    parsed_rows: int
    invalid_rows: int
    conflicting_rows: int
    duplicate_rows: int
    final_rows: int
    text_overlap: int
    group_overlap: int


class Provenance(Aggregate):
    """Bind new aggregates to the exact local artifacts."""

    dataset_sha256: str
    raw_dataset_sha256: str
    scope: str


class HadoopRuntime(Aggregate):
    """Execution status may only be upgraded using an actual receipt."""

    status: str
    label: str
    reason: str


class SparkRuntime(Aggregate):
    """Historical workload timings, independent of this EDA run."""

    status: str
    label: str
    scope: str
    python_seconds: float
    spark_seconds: float
    startup_seconds: float


class Runtime(Aggregate):
    """Explicitly separate completed and pending engines."""

    hadoop: HadoopRuntime
    spark: SparkRuntime


class EdaReport(Aggregate):
    """The complete browser-safe reporting interface."""

    schema_version: int = 1
    generated_at: str
    provenance: Provenance
    summary: Summary
    pipeline: tuple[Stage, ...]
    sources: tuple[SourceSummary, ...]
    labels: tuple[Count, ...]
    splits: tuple[SplitSummary, ...]
    missing: tuple[MissingSummary, ...]
    length: LengthProfile
    targets: tuple[Count, ...]
    expressions: tuple[Count, ...]
    methods: tuple[Method, ...]
    runtime: Runtime
    notes: tuple[str, ...]
