"""Audit the frozen deployed classifier on validation only, without refitting."""

from __future__ import annotations

import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Final, Literal

import typer
from pydantic import BaseModel, ConfigDict, ValidationError
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.classification_contracts import (
    ClassificationReport,
    ExperimentError,
    Metrics,
    Row,
    load_rows,
)
from bigdb_mvp.experiments.classification_metrics import metrics
from bigdb_mvp.experiments.validation_profile import Observation, SliceAudit, profile
from bigdb_mvp.korean_model import (
    KoreanModelService,
    KoreanRequest,
    load_korean_model,
)

DATA_ROOT: Final = Path("data/curated/korean_v2")
REPORT_ROOT: Final = Path("reports/validation_audit_20261029")
MODEL_REPORT: Final = Path("reports/korean_short_v1/metrics.json")
BATCH_SIZE: Final = 2048


class AuditReport(BaseModel):
    """Bind diagnostic results to the exact deployed model and existing split."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    generated_at: str
    elapsed_seconds: float
    dataset_sha256: str
    model_sha256: str
    method: Literal["char_2_5_200000"] = "char_2_5_200000"
    split: Literal["validation"] = "validation"
    threshold: float
    raw_binary_metrics: Metrics
    reference_confusion_verified: Literal[True] = True
    policy: str = (
        "KoreanRequest input validation, then review when feature count is zero; "
        "binary metrics are conditional on decided inputs. "
        "No refit, threshold selection, test inference or model publication. "
        "This validation split was previously used for model selection."
    )
    slices: tuple[SliceAudit, ...]


def accepts(text: str) -> bool:
    """Use the service's actual boundary rather than duplicating its rules."""
    try:
        _ = KoreanRequest(text=text)
    except ValidationError:
        return False
    return True


def observe(
    rows: tuple[Row, ...], service: KoreanModelService
) -> tuple[Observation, ...]:
    """Predict in bounded batches with the persisted vocabulary and threshold."""
    observations: list[Observation] = []
    for offset in range(0, len(rows), BATCH_SIZE):
        batch = rows[offset : offset + BATCH_SIZE]
        matrix = service.model.vectorizer.transform(tuple(row.text for row in batch))
        scores = service.model.classifier.predict_proba(matrix)[:, 1]
        counts = matrix.getnnz(axis=1)
        observations.extend(
            Observation(
                source=row.source,
                label=row.label,
                length=len(row.text),
                score=float(score),
                feature_count=int(count),
                accepted_input=accepts(row.text),
            )
            for row, score, count in zip(batch, scores, counts, strict=True)
        )
    return tuple(observations)


def run() -> AuditReport:
    """Verify data and model provenance, then isolate existing validation rows."""
    started = time.perf_counter()
    rows, digest = load_rows(DATA_ROOT)
    service = load_korean_model(
        Path("artifacts/korean_short_v1"),
        MODEL_REPORT,
        method="char_2_5_200000",
    )
    if digest != service.dataset_sha256:
        raise ExperimentError(reason_code="audit_dataset_mismatch")
    validation = tuple(row for row in rows if row.split == "validation")
    with threadpool_limits(limits=2):
        observations = observe(validation, service)
    raw = metrics(
        tuple(int(row.label == "harmful") for row in observations),
        tuple(row.score for row in observations),
        service.model.threshold,
    )
    saved = ClassificationReport.model_validate_json(MODEL_REPORT.read_bytes())
    reference = next(
        result.validation
        for result in saved.results
        if result.method == "char_2_5_200000"
    )
    if (raw.count, raw.tn, raw.fp, raw.fn, raw.tp) != (
        reference.count,
        reference.tn,
        reference.fp,
        reference.fn,
        reference.tp,
    ):
        raise ExperimentError(reason_code="audit_reference_confusion_mismatch")
    return AuditReport(
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        dataset_sha256=digest,
        model_sha256=service.model_sha256,
        threshold=service.model.threshold,
        raw_binary_metrics=raw,
        slices=profile(observations, service.model.threshold),
    )


def main() -> None:
    """Publish aggregate-only JSON and CSV without touching training artifacts."""
    report = run()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _ = (REPORT_ROOT / "report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    flattened: list[dict[str, str | int | float | None]] = []
    for result in report.slices:
        row = result.model_dump(exclude={"key", "decision_stats"})
        row.update(result.key.model_dump())
        if result.decision_stats is not None:
            row.update(result.decision_stats.model_dump())
        flattened.append(row)
    fields = tuple(dict.fromkeys(key for row in flattened for key in row))
    with (REPORT_ROOT / "slices.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    typer.echo(f"Validation audit: {report.slices[0].total} rows -> {REPORT_ROOT}")


if __name__ == "__main__":
    main()
