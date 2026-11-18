"""Matched small-data ablations; production artifacts are never overwritten."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Final

import typer
from kiwipiepy import Kiwi
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.classification_contracts import Metrics, Row, load_rows
from bigdb_mvp.experiments.classification_metrics import (
    metrics,
    threshold_on_validation,
)
from bigdb_mvp.experiments.hypothesis_cases import Case, cases, normalize
from bigdb_mvp.experiments.hypothesis_features import ViewKind, branch, matrix, views
from bigdb_mvp.experiments.hypothesis_samples import sample
from bigdb_mvp.korean_model import load_korean_model

REPORT: Final = Path("reports/hypothesis_pilot_20261014")
METHODS: Final[tuple[tuple[str, tuple[ViewKind, ...]], ...]] = (
    ("raw_char", ("raw",)),
    ("raw_normalized", ("raw", "normalized")),
    ("raw_morph", ("raw", "morph")),
    ("raw_normalized_morph", ("raw", "normalized", "morph")),
)


class _Result(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    method: str
    threshold: float
    validation: Metrics
    test: Metrics
    test_review_count: int
    fit_seconds: float


def _export_cases(
    name: str,
    challenge: tuple[Case, ...],
    scores: tuple[float, ...],
    counts: tuple[int, ...],
    threshold: float,
) -> None:
    with (REPORT / f"{name}_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case_id",
                "family",
                "category",
                "text",
                "normalized",
                "expected_authored",
                "score",
                "features",
                "label",
                "matches_intent",
            )
        )
        for case, score, count in zip(challenge, scores, counts, strict=True):
            label = (
                "review" if count == 0 else "harmful" if score >= threshold else "clean"
            )
            writer.writerow(
                (
                    case.case_id,
                    case.family,
                    case.category,
                    case.text,
                    normalize(case.text),
                    case.expected,
                    score * 100,
                    count,
                    label,
                    label == case.expected,
                )
            )


def main() -> None:
    """Freeze splits and expectations, then compare four feature pipelines."""
    started = time.perf_counter()
    rows, digest = load_rows(Path("data/curated/korean_v2"))
    partitions = tuple(
        sample(rows, split, cap)
        for split, cap in (("train", 1000), ("validation", 200), ("test", 200))
    )
    original_counts = tuple(map(len, partitions))
    seen: set[str] = set()
    disjoint: list[tuple[Row, ...]] = []
    for partition in partitions:
        retained = tuple(row for row in partition if normalize(row.text) not in seen)
        disjoint.append(retained)
        seen.update(normalize(row.text) for row in retained)
    partitions = tuple(disjoint)
    challenge = cases()
    REPORT.mkdir(parents=True, exist_ok=True)
    configuration = {
        "dataset_sha256": digest,
        "seed": 20261014,
        "counts": dict(
            zip(("train", "validation", "test"), map(len, partitions), strict=True)
        ),
        "normalized_overlap_removed": tuple(
            before - len(after)
            for before, after in zip(original_counts, partitions, strict=True)
        ),
        "sampling": (
            "sha256(seed|text_sha256) sort; per source/label cap1000 train,"
            "200 validation/test; same original splits"
        ),
        "feature_budget": "60000 total max; divided equally among branches",
        "methods": METHODS,
        "kiwipiepy": importlib.metadata.version("kiwipiepy"),
        "morphology": (
            "default Kiwi, form/POS unigram/bigram; retain negation; "
            "no typo correction or custom dictionary"
        ),
        "normalization": (
            "NFC; limited mixed-jamo replacement; "
            "separator removal only inside declared profanity patterns"
        ),
        "normalization_words": (
            "시발",
            "씨발",
            "병신",
            "지랄",
            "개새끼",
            "꺼져",
            "닥쳐",
            "좆같",
        ),
        "limits": (
            "Pilot source-balanced sample, not population accuracy. "
            "Existing test previously inspected. Authored intentions not gold. "
            "No severity targets or production replacement."
        ),
    }
    _ = (REPORT / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _ = (REPORT / "authored_cases.json").write_text(
        json.dumps([asdict(case) for case in challenge], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (REPORT / "sample_ids.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "source", "label", "text_sha256", "group_id"))
        for partition in partitions:
            writer.writerows(
                (row.split, row.source, row.label, row.text_sha256, row.group_id)
                for row in partition
            )
    kiwi = Kiwi(num_workers=2)
    prepared = tuple(
        views(tuple(row.text for row in partition), kiwi) for partition in partitions
    )
    challenge_views = views(tuple(case.text for case in challenge), kiwi)
    labels = tuple(
        tuple(int(row.label == "harmful") for row in partition)
        for partition in partitions
    )
    service = load_korean_model(
        Path("artifacts/korean_short_v1"),
        Path("reports/korean_short_v1/metrics.json"),
        method="char_2_5_200000",
    )
    for name, texts in (
        ("production_raw", challenge_views["raw"]),
        ("production_normalized_input_only", challenge_views["normalized"]),
    ):
        production_matrix = service.model.vectorizer.transform(texts)
        _export_cases(
            name,
            challenge,
            tuple(
                float(score)
                for score in service.model.classifier.predict_proba(production_matrix)[
                    :, 1
                ]
            ),
            tuple(int(count) for count in production_matrix.getnnz(axis=1)),
            service.model.threshold,
        )
    results: list[_Result] = []
    for name, kinds in METHODS:
        typer.echo(f"Training pilot {name}")
        fit_start = time.perf_counter()
        branches = tuple(branch(kind, 60000 // len(kinds)) for kind in kinds)
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=20261014,
            solver="liblinear",
        )
        _ = classifier.fit(matrix(branches, prepared[0], fit=True), labels[0])
        fit_seconds = time.perf_counter() - fit_start
        validation_scores = tuple(
            float(score)
            for score in classifier.predict_proba(
                matrix(branches, prepared[1], fit=False)
            )[:, 1]
        )
        threshold = threshold_on_validation(labels[1], validation_scores)
        test_matrix = matrix(branches, prepared[2], fit=False)
        test_scores = tuple(
            float(score) for score in classifier.predict_proba(test_matrix)[:, 1]
        )
        result = _Result(
            method=name,
            threshold=threshold,
            validation=metrics(labels[1], validation_scores, threshold),
            test=metrics(labels[2], test_scores, threshold),
            test_review_count=sum(count == 0 for count in test_matrix.getnnz(axis=1)),
            fit_seconds=fit_seconds,
        )
        results.append(result)
        challenge_matrix = matrix(branches, challenge_views, fit=False)
        _export_cases(
            name,
            challenge,
            tuple(
                float(score)
                for score in classifier.predict_proba(challenge_matrix)[:, 1]
            ),
            tuple(int(count) for count in challenge_matrix.getnnz(axis=1)),
            threshold,
        )
        typer.echo(result.model_dump_json())
    _ = (REPORT / "results.json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "results": [result.model_dump() for result in results],
                "total_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
