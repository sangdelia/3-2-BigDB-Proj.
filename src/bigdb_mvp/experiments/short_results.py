"""Freeze validation selection before inspecting authored short-input probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.classification_contracts import (
    ClassificationReport,
    MethodResult,
    Metrics,
    Row,
)
from bigdb_mvp.experiments.classification_metrics import metrics
from bigdb_mvp.experiments.score_probe import CASES
from bigdb_mvp.experiments.short_features import ARTIFACTS, REPORTS, VARIANTS
from bigdb_mvp.korean_model import KoreanRequest, load_korean_model


class _PolicyResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    method: str
    review_count: int
    harmful_review_count: int
    automatic_metrics: Metrics


def main() -> None:
    """Select on validation, export probes, and retain original artifacts."""
    baseline = next(
        result
        for result in ClassificationReport.model_validate_json(
            Path("reports/korean_v2/metrics.json").read_bytes()
        ).results
        if result.method == "char_tfidf"
    )
    variants = tuple(
        MethodResult.model_validate_json((REPORTS / f"{name}.json").read_bytes())
        for name, _ngrams, _limit in VARIANTS
    )
    selected = max((baseline, *variants), key=lambda result: result.validation.macro_f1)
    selection = {
        "method": selected.method,
        "artifact_root": str(ARTIFACTS)
        if selected != baseline
        else "artifacts/korean_v2",
        "report_path": str(REPORTS / "metrics.json")
        if selected != baseline
        else "reports/korean_v2/metrics.json",
        "criterion": "validation macro_f1 only; baseline wins exact ties",
        "baseline": baseline.model_dump(),
        "variants": [result.model_dump() for result in variants],
    }
    _ = (REPORTS / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    services = (
        (baseline.method, load_korean_model()),
        *(
            (
                result.method,
                load_korean_model(
                    ARTIFACTS, REPORTS / "metrics.json", method=result.method
                ),
            )
            for result in variants
        ),
    )
    validation = pl.read_parquet("data/curated/korean_v2/corpus.parquet").filter(
        pl.col("split") == "validation"
    )
    rows = tuple(Row.model_validate(row) for row in validation.iter_rows(named=True))
    texts = tuple(row.text for row in rows)
    labels = tuple(int(row.label == "harmful") for row in rows)
    policies: list[_PolicyResult] = []
    for name, service in services:
        matrix = service.model.vectorizer.transform(texts)
        counts = matrix.getnnz(axis=1)
        scores = tuple(
            float(score)
            for score in service.model.classifier.predict_proba(matrix)[:, 1]
        )
        known = tuple(i for i, count in enumerate(counts) if count > 0)
        policies.append(
            _PolicyResult(
                method=name,
                review_count=sum(count == 0 for count in counts),
                harmful_review_count=sum(
                    label == 1 and count == 0
                    for label, count in zip(labels, counts, strict=True)
                ),
                automatic_metrics=metrics(
                    tuple(labels[i] for i in known),
                    tuple(scores[i] for i in known),
                    service.model.threshold,
                ),
            )
        )
    _ = (REPORTS / "review_policy.json").write_text(
        json.dumps([result.model_dump() for result in policies], indent=2),
        encoding="utf-8",
    )
    with (REPORTS / "authored_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("method", "category", "text", "score", "label", "features", "highlighted")
        )
        for name, service in services:
            for category, texts in CASES:
                for text in texts:
                    prediction = service.predict(KoreanRequest(text=text))
                    writer.writerow(
                        (
                            name,
                            category,
                            text,
                            prediction.harmful_score,
                            prediction.label,
                            prediction.recognized_feature_count,
                            sum(
                                item.highlight_as_harmful_candidate
                                for item in prediction.candidates
                            ),
                        )
                    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
