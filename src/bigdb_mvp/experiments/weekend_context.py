"""Compare one longer-character candidate before any test-based diagnostics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Final

import joblib
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification_contracts import Metrics, Row, load_rows
from bigdb_mvp.experiments.classification_metrics import (
    metrics,
    threshold_on_validation,
)
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.experiments.hypothesis_cases import cases
from bigdb_mvp.korean_model import load_korean_model

REPORTS: Final = Path("reports/weekend_20261101")
ARTIFACTS: Final = Path("artifacts/weekend_20261101")
NAME: Final = "char_2_8_200000"


def eligible(candidate: Metrics, baseline: Metrics) -> bool:
    """Require improved F1 with neither higher clean FPR nor lower recall."""
    return (
        candidate.macro_f1 > baseline.macro_f1
        and candidate.false_positive_rate <= baseline.false_positive_rate
        and candidate.harmful_recall >= baseline.harmful_recall
    )


def scores(model: Model, texts: tuple[str, ...]) -> tuple[float, ...]:
    """Use a persisted model's own vocabulary without refitting."""
    matrix = model.vectorizer.transform(texts)
    return tuple(float(p) for p in model.classifier.predict_proba(matrix)[:, 1])


def evaluate(model: Model, rows: tuple[Row, ...]) -> Metrics:
    """Evaluate corpus labels using the frozen threshold."""
    return metrics(
        tuple(int(row.label == "harmful") for row in rows),
        scores(model, tuple(row.text for row in rows)),
        model.threshold,
    )


def diagnostics(baseline: Model, candidate: Model) -> None:
    """Keep authored intent labels and zero-feature cases out of selection."""
    authored = cases()
    texts = tuple(case.text for case in authored)
    with (REPORTS / "authored.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("model", "category", "text", "expected", "score", "label"))
        for name, model in (("baseline", baseline), (NAME, candidate)):
            values = scores(model, texts)
            counts = model.vectorizer.transform(texts).getnnz(axis=1)
            for case, score, count in zip(authored, values, counts, strict=True):
                label = (
                    "review"
                    if count == 0
                    else "harmful"
                    if score >= model.threshold
                    else "clean"
                )
                writer.writerow(
                    (name, case.category, case.text, case.expected, score * 100, label)
                )


def main() -> None:
    """Freeze validation selection before evaluating the already-seen test."""
    REPORTS.mkdir(parents=True, exist_ok=False)
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    rows, digest = load_rows(Path("data/curated/korean_v2"))
    train, validation, test = (
        tuple(row for row in rows if row.split == split)
        for split in ("train", "validation", "test")
    )
    baseline = load_korean_model(
        Path("artifacts/korean_short_v1"),
        Path("reports/korean_short_v1/metrics.json"),
        method="char_2_5_200000",
    )
    configuration = {
        "dataset_sha256": digest,
        "baseline_sha256": baseline.model_sha256,
        "candidate": NAME,
        "ngram_range": [2, 8],
        "max_features": 200000,
        "seed": 20261010,
        "split_counts": [len(train), len(validation), len(test)],
        "selection": "validation F1 higher; clean FPR <= baseline; recall >= baseline",
        "test_role": "previously seen follow-up only; no retuning",
        "authored_role": "71 previously seen diagnostic cases; not selection",
    }
    _ = (REPORTS / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    with threadpool_limits(limits=2):
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 8),
            sublinear_tf=True,
            max_features=200000,
        )
        classifier = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=20261010,
        )
        matrix = vectorizer.fit_transform(tuple(row.text for row in train))
        _ = classifier.fit(matrix, tuple(int(row.label == "harmful") for row in train))
        fit_seconds = time.perf_counter() - started
        temporary = Model(vectorizer, classifier, None, 0.5, context=False)
        validation_scores = scores(temporary, tuple(row.text for row in validation))
        threshold = threshold_on_validation(
            tuple(int(row.label == "harmful") for row in validation),
            validation_scores,
        )
        candidate = Model(vectorizer, classifier, None, threshold, context=False)
        artifact = ARTIFACTS / f"{NAME}.joblib"
        _ = joblib.dump(candidate, artifact)
        baseline_validation = evaluate(baseline.model, validation)
        candidate_validation = metrics(
            tuple(int(row.label == "harmful") for row in validation),
            validation_scores,
            threshold,
        )
        selection = {
            "configuration": configuration,
            "candidate_eligible": eligible(candidate_validation, baseline_validation),
            "baseline_validation": baseline_validation.model_dump(),
            "candidate_validation": candidate_validation.model_dump(),
            "baseline_threshold": baseline.model.threshold,
            "candidate_threshold": threshold,
            "candidate_sha256": sha256_file(artifact),
            "fit_seconds": fit_seconds,
            "deployed": False,
        }
        selection_path = REPORTS / "selection.json"
        _ = selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        typer.echo(json.dumps(selection))
        followup = {
            "selection_sha256": sha256_file(selection_path),
            "baseline_test": evaluate(baseline.model, test).model_dump(),
            "candidate_test": evaluate(candidate, test).model_dump(),
        }
        _ = (REPORTS / "followup.json").write_text(
            json.dumps(followup, indent=2), encoding="utf-8"
        )
        diagnostics(baseline.model, candidate)
        typer.echo(json.dumps(followup))


if __name__ == "__main__":
    main()
