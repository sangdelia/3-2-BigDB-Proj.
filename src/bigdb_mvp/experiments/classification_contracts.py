"""Validated Korean experiment inputs and safe aggregate outputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from bigdb_mvp.data.staging import sha256_file

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentError(ValueError):
    """Reject an unsafe experiment boundary with a stable reason code."""

    reason_code: str

    @override
    def __str__(self) -> str:
        return self.reason_code


class Row(BaseModel):
    """Only permitted text and audit fields; gold metadata is never a feature."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    text: str = Field(min_length=1)
    label: Literal["clean", "harmful"]
    language: Literal["ko"]
    source: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split: Literal["train", "validation", "test"]
    context_text: str
    context_kind: str


class DatasetManifest(BaseModel):
    """Required immutable dataset provenance."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Metrics(BaseModel):
    """Harmful is the positive class in every binary confusion matrix."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    count: int
    accuracy: float
    macro_f1: float
    harmful_precision: float
    harmful_recall: float
    harmful_f1: float
    false_positive_rate: float
    tn: int
    fp: int
    fn: int
    tp: int


class MethodResult(BaseModel):
    """Text-free results with validation separated from final held-out test."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    method: str
    train_count: int
    threshold: float
    validation: Metrics
    test: Metrics
    test_by_source: dict[str, Metrics]
    fit_seconds: float
    test_predict_seconds: float
    peak_rss_mib: float
    model_sha256: str


class ClassificationReport(BaseModel):
    """Parse model results and dataset provenance before loading local artifacts."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    configuration: DatasetManifest
    results: tuple[MethodResult, ...]


def load_rows(root: Path) -> tuple[tuple[Row, ...], str]:
    """Fail closed on drift, language mixing, and group or text split leakage."""
    manifest = DatasetManifest.model_validate_json(
        (root / "manifest.json").read_bytes()
    )
    dataset = root / "corpus.parquet"
    if sha256_file(dataset) != manifest.dataset_sha256:
        reason = "dataset_digest_mismatch"
        raise ExperimentError(reason)
    rows = tuple(
        Row.model_validate(row)
        for row in pl.read_parquet(dataset).iter_rows(named=True)
    )
    groups: dict[str, str] = {}
    texts: dict[str, str] = {}
    for row in rows:
        if (
            hashlib.sha256(" ".join(row.text.split()).encode()).hexdigest()
            != row.text_sha256
        ):
            reason = "text_digest_mismatch"
            raise ExperimentError(reason)
        for key, mapping in ((row.group_id, groups), (row.text_sha256, texts)):
            if key in mapping and mapping[key] != row.split:
                reason = "cross_split_leakage"
                raise ExperimentError(reason)
            mapping[key] = row.split
    require_splits(rows)
    return rows, manifest.dataset_sha256


def require_splits(rows: tuple[Row, ...]) -> None:
    """Require both labels in each partition before fitting or reporting."""
    for split in ("train", "validation", "test"):
        if {row.label for row in rows if row.split == split} != {"clean", "harmful"}:
            reason = f"missing_class_or_split:{split}"
            raise ExperimentError(reason)


def feature_text(row: Row, *, context: bool = False) -> str:
    """Construct features exclusively from user text and genuine optional context."""
    return f"제목: {row.context_text}\n댓글: {row.text}" if context else row.text
