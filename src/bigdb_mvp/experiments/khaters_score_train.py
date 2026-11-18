"""Train, validate and independently evaluate original-rating score heads."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import joblib
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.hypothesis_cases import cases
from bigdb_mvp.experiments.khaters_ratings import (
    KINDS,
    REVISION,
    Rating,
    Ratings,
    distribution,
    load_dataset,
)
from bigdb_mvp.khaters_score import (
    Head,
    Probabilities,
    ScoreModel,
    gold_probability,
    predict,
    probabilities,
)
from bigdb_mvp.khaters_score_artifact import load_score_model

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import FeatureMatrix

ROOT: Final = Path("data/raw/khaters_ratings") / REVISION
ARTIFACT: Final = Path("artifacts/khaters_score_v1/model.joblib")
REPORT: Final = Path("reports/khaters_score_v1")
TEMPERATURES: Final = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
app: Final = typer.Typer()


def nll(values: tuple[Probabilities, ...], gold: tuple[Rating, ...]) -> float:
    """Evaluate gold likelihood with a finite floor for missing class support."""
    return -sum(
        math.log(max(gold_probability(p, y), 1e-15))
        for p, y in zip(values, gold, strict=True)
    ) / len(gold)


def calibrated(head: Head, matrix: FeatureMatrix, gold: tuple[Rating, ...]) -> Head:
    """Select a frozen temperature by validation NLL only."""
    candidates = tuple(replace(head, temperature=t) for t in TEMPERATURES)
    return min(
        candidates,
        key=lambda candidate: (
            nll(probabilities(candidate, matrix), gold),
            abs(candidate.temperature - 1),
        ),
    )


def evaluate(
    head: Head,
    matrix: FeatureMatrix,
    rows: tuple[Ratings, ...],
    training: tuple[Ratings, ...],
) -> str:
    """Report ordinal error, grade F1 and a train-mean score comparator."""
    gold: tuple[Rating, ...] = tuple(row.grade(head.kind) for row in rows)
    values = probabilities(head, matrix)
    scores = tuple(p.score for p in values)
    labels = tuple(
        max(range(3), key=lambda i: (p.zero, p.one, p.two)[i]) for p in values
    )
    confusion = [
        [
            sum(y == i and p == j for y, p in zip(gold, labels, strict=True))
            for j in range(3)
        ]
        for i in range(3)
    ]
    f1 = tuple(
        2
        * confusion[i][i]
        / max(sum(confusion[i]) + sum(row[i] for row in confusion), 1)
        for i in range(3)
    )
    prior = 50 * sum(row.grade(head.kind) for row in training) / len(training)
    return json.dumps(
        {
            "count": len(gold),
            "grade_counts": distribution(rows, head.kind),
            "mae_points": sum(
                abs(score - 50 * y) for score, y in zip(scores, gold, strict=True)
            )
            / len(gold),
            "rmse_points": math.sqrt(
                sum(
                    (score - 50 * y) ** 2 for score, y in zip(scores, gold, strict=True)
                )
                / len(gold)
            ),
            "train_mean_mae_points": sum(abs(prior - 50 * y) for y in gold) / len(gold),
            "grade_macro_f1": sum(f1) / 3,
            "confusion": confusion,
            "nll": nll(values, gold),
            "temperature": head.temperature,
        }
    )


@app.command()
def train() -> None:
    """Freeze three ordinal rating heads and retain source hashes and diagnostics."""
    started = time.monotonic()
    dataset = load_dataset(ROOT)
    training, validation, testing = dataset.splits
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), sublinear_tf=True, max_features=100000
    )
    train_matrix = vectorizer.fit_transform(tuple(row.text for row in training))
    val_matrix = vectorizer.transform(tuple(row.text for row in validation))
    test_matrix = vectorizer.transform(tuple(row.text for row in testing))
    heads: list[Head] = []
    results: dict[str, dict[str, str]] = {}
    for kind in KINDS:
        typer.echo(f"Training {kind}: {len(training)} rows")
        with threadpool_limits(limits=2):
            classifier = LogisticRegression(
                class_weight=None, solver="lbfgs", max_iter=1000, random_state=20261014
            )
            _ = classifier.fit(train_matrix, tuple(row.grade(kind) for row in training))
        head = calibrated(
            Head(kind, classifier, 1.0),
            val_matrix,
            tuple(row.grade(kind) for row in validation),
        )
        heads.append(head)
        results[kind] = {
            "validation": evaluate(head, val_matrix, validation, training),
            "test": evaluate(head, test_matrix, testing, training),
        }
    digest = hashlib.sha256("".join(dataset.file_sha256).encode()).hexdigest()
    model = ScoreModel(vectorizer, tuple(heads), digest, REVISION)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    _ = joblib.dump(model, ARTIFACT)
    artifact_digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    report = {
        "dataset": "humane-lab/K-HATERS-Ratings",
        "revision": REVISION,
        "file_sha256": dataset.file_sha256,
        "dataset_digest": digest,
        "original_counts": dataset.original_counts,
        "retained_counts": tuple(map(len, dataset.splits)),
        "excluded_rows": dataset.excluded_rows,
        "empty_rows": dataset.empty_rows,
        "cross_split_text_keys": dataset.cross_split_keys,
        "conflicting_rating_keys": dataset.conflicting_keys,
        "model_sha256": artifact_digest,
        "formula": "50 * (P(grade=1) + 2 * P(grade=2))",
        "gold_hate": "max(gender,age,disability,job,race,religion,politics)",
        "gold_abuse": "max(all 13 original ratings)",
        "gold_swear": "swear",
        "features": "raw char 2-5, train-only vocabulary, max 100000",
        "class_weight": None,
        "seed": 20261014,
        "temperature_selection": "validation NLL, grid 0.5/0.75/1/1.5/2/3",
        "zero_feature_rows": tuple(
            sum(int(n) == 0 for n in matrix.getnnz(axis=1))
            for matrix in (train_matrix, val_matrix, test_matrix)
        ),
        "results": {
            kind: {split: json.loads(data) for split, data in result.items()}
            for kind, result in results.items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Ordinal spacing is a convention, not physical severity",
            "Validation calibration does not guarantee new-text calibration",
            "Dedup does not detect all article or semantic overlap",
            "Six-source model saw some K-HATERS test texts; no ensemble",
        ],
    }
    _ = (REPORT / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (REPORT / "authored_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case_id",
                "category",
                "text",
                "recognized_features",
                "hate_score",
                "abuse_score",
                "swear_score",
            )
        )
        for case in cases():
            prediction = predict(model, case.text)
            writer.writerow(
                (
                    case.case_id,
                    case.category,
                    case.text,
                    prediction.recognized_features,
                    prediction.hate_score,
                    prediction.abuse_score,
                    prediction.swear_score,
                )
            )
    typer.echo(json.dumps(report, indent=2))


@app.command()
def score(text: str) -> None:
    """Load the verified local artifact and expose numerical predictions as JSON."""
    typer.echo(predict(load_score_model(Path.cwd()), text).model_dump_json(indent=2))


if __name__ == "__main__":
    app()
