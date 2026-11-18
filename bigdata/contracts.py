"""Typed receipts for the single-host Hadoop data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

SOURCES: Final = ("beep", "khaters", "kmhas", "kold", "nikl", "unsmile")
BOUNDS: Final = (10, 20, 50, 100, 200, 500, 1000)


class RawRow(BaseModel):
    """Accepted-source input; unusable values are counted before exclusion."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    text: str | None
    label: str | None
    source: str | None


class SparkSettings(BaseModel):
    """Local task concurrency, independent of input partition count."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    workers: int = Field(default=2, ge=1, le=8)
    partitions: int = Field(default=4, ge=1, le=64)


class CleaningAudit(BaseModel):
    """Mutually exclusive row losses; together they reconcile with Bronze."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    bronze_rows: int = Field(ge=0)
    invalid_rows: int = Field(ge=0)
    conflict_rows: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    silver_rows: int = Field(ge=0)


class HdfsPaths(BaseModel):
    """The three storage layers of one immutable pipeline run."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    bronze: str
    silver: str
    mr_input: str
    gold: str


class AggregateRow(BaseModel):
    """One persisted Spark Gold row, parsed before numeric comparison."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    key: str
    n: int = Field(ge=0)


class SilverKey(BaseModel):
    """Persisted Silver identity used by the independent key digest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    digest: str
    label: str
    source: str


class SparkReceipt(BaseModel):
    """Spark stage output consumed by the independent verifier."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["complete"] = "complete"
    spark_version: str
    default_fs: str
    paths: HdfsPaths
    bronze_rows: int = Field(ge=0)
    silver_rows: int = Field(ge=0)
    output_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregates: dict[str, int]
    startup_seconds: float = Field(ge=0)
    job_seconds: float = Field(ge=0)
    settings: SparkSettings = Field(default_factory=SparkSettings)
    audit: CleaningAudit | None = None
    mode: str = "single host; Spark local threads; MapReduce LocalJobRunner"


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """Independent Python result, including the row-loss audit."""

    output_keys_sha256: str
    aggregates: dict[str, int]
    audit: CleaningAudit


class VerificationReceipt(BaseModel):
    """Only a passed receipt may be published to HDFS as verified."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    status: Literal["passed", "failed"]
    checks: dict[str, bool]
    bronze_rows: int
    silver_rows: int
    aggregate_keys: int
    input_sha256: str
    python_output_keys_sha256: str
    python_aggregates: dict[str, int]
    audit: CleaningAudit
    mismatched_keys: list[str]
