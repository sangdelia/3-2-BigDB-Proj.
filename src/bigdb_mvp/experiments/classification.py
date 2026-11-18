"""Run frozen Korean classifier comparisons with honest context ablations."""

from __future__ import annotations

import csv
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Final

import joblib
import typer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bigdb_mvp.data.staging import sha256_file
from bigdb_mvp.experiments.classification_contracts import (
    MethodResult,
    Metrics,
    Row,
    feature_text,
    load_rows,
    require_splits,
)
from bigdb_mvp.experiments.classification_metrics import (
    metrics,
    threshold_on_validation,
)
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.experiments.lexicon import Lexicon, learn_lexicon, lexicon_features

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import FeatureMatrix

DEFAULT_DATASET: Final = Path("data/curated/korean_v2")
DEFAULT_ARTIFACT: Final = Path("artifacts/korean_v2")
DEFAULT_REPORT: Final = Path("reports/korean_v2")

METHODS: Final = (
    "char_tfidf",
    "char_tfidf_dictionary",
    "context_comment",
    "context_title_comment",
)


def _matrix(
    vectorizer: TfidfVectorizer,
    texts: tuple[str, ...],
    lexicon: Lexicon | None,
    *,
    fit: bool,
) -> FeatureMatrix:
    matrix = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return (
        matrix
        if lexicon is None
        else hstack((matrix, lexicon_features(texts, lexicon)), format="csr")
    )


def _labels(rows: tuple[Row, ...]) -> tuple[int, ...]:
    return tuple(int(row.label == "harmful") for row in rows)


def run_method(
    name: str,
    rows: tuple[Row, ...],
    artifact_root: Path,
    max_features: int,
    *,
    ngram_range: tuple[int, int] = (3, 5),
) -> MethodResult:
    """Fit once, select policy on validation, and predict final test once."""
    context = name == "context_title_comment"
    selected = (
        tuple(row for row in rows if row.context_text.strip())
        if name.startswith("context_")
        else rows
    )
    require_splits(selected)
    train, validation, test = (
        tuple(row for row in selected if row.split == split)
        for split in ("train", "validation", "test")
    )
    started = time.perf_counter()
    lexicon = learn_lexicon(train) if name == "char_tfidf_dictionary" else None
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        sublinear_tf=True,
        max_features=max_features,
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=20261010,
        solver="liblinear",
    )
    train_matrix = _matrix(
        vectorizer,
        tuple(feature_text(row, context=context) for row in train),
        lexicon,
        fit=True,
    )
    _ = classifier.fit(train_matrix, _labels(train))
    fit_seconds = time.perf_counter() - started
    validation_matrix = _matrix(
        vectorizer,
        tuple(feature_text(row, context=context) for row in validation),
        lexicon,
        fit=False,
    )
    validation_scores = tuple(
        float(value) for value in classifier.predict_proba(validation_matrix)[:, 1]
    )
    threshold = threshold_on_validation(_labels(validation), validation_scores)
    model = Model(vectorizer, classifier, lexicon, threshold, context)
    model_path = artifact_root / f"{name}.joblib"
    _ = joblib.dump(model, model_path)
    if lexicon is not None:
        _ = (artifact_root / "train_lexicon.json").write_text(
            json.dumps(asdict(lexicon), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    started = time.perf_counter()
    test_matrix = _matrix(
        vectorizer,
        tuple(feature_text(row, context=context) for row in test),
        lexicon,
        fit=False,
    )
    test_scores = tuple(
        float(value) for value in classifier.predict_proba(test_matrix)[:, 1]
    )
    predict_seconds = time.perf_counter() - started
    slices: dict[str, Metrics] = {}
    for source in sorted({row.source for row in test}):
        indices = tuple(i for i, row in enumerate(test) if row.source == source)
        slices[source] = metrics(
            tuple(int(test[i].label == "harmful") for i in indices),
            tuple(test_scores[i] for i in indices),
            threshold,
        )
    return MethodResult(
        method=name,
        train_count=len(train),
        threshold=threshold,
        validation=metrics(_labels(validation), validation_scores, threshold),
        test=metrics(_labels(test), test_scores, threshold),
        test_by_source=slices,
        fit_seconds=fit_seconds,
        test_predict_seconds=predict_seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        model_sha256=sha256_file(model_path),
    )


def run(
    dataset_root: Path,
    artifact_root: Path,
    report_root: Path,
    *,
    max_features: int = 60000,
    threads: int = 2,
) -> tuple[MethodResult, ...]:
    """Execute the predeclared four methods without test-driven selection."""
    seed = 20261010
    rows, digest = load_rows(dataset_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "dataset_sha256": digest,
        "methods": METHODS,
        "seed": seed,
        "max_features": max_features,
        "threads": threads,
        "label_positive": "harmful",
        "feature_columns": ["text", "context_text (context_title_comment only)"],
        "threshold_policy": (
            "validation macro_f1; ties nearest 0.5 then lower; "
            "grid 0.05..0.95 step 0.01"
        ),
        "memory_note": (
            "Linux shared-process cumulative peak RSS; not isolated per-method memory"
        ),
        "embedding_status": "pending: pretrained sentence embedding comparison not run",
        "context_policy": (
            "both context methods train and evaluate on identical "
            "actual nonempty-context subset"
        ),
    }
    _ = (artifact_root / "manifest.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    with threadpool_limits(limits=threads):
        results = tuple(
            run_method(name, rows, artifact_root, max_features) for name in METHODS
        )
    _ = (report_root / "metrics.json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "results": [result.model_dump() for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (report_root / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "partition",
                "count",
                "accuracy",
                "macro_f1",
                "harmful_precision",
                "harmful_recall",
                "harmful_f1",
                "false_positive_rate",
                "fit_seconds",
                "test_predict_seconds",
                "peak_rss_mib",
            ]
        )
        for result in results:
            for partition, value in (
                ("validation", result.validation),
                ("test", result.test),
            ):
                writer.writerow(
                    [
                        result.method,
                        partition,
                        value.count,
                        value.accuracy,
                        value.macro_f1,
                        value.harmful_precision,
                        value.harmful_recall,
                        value.harmful_f1,
                        value.false_positive_rate,
                        result.fit_seconds,
                        result.test_predict_seconds,
                        result.peak_rss_mib,
                    ]
                )
    return results


def main(
    dataset_root: Path = DEFAULT_DATASET,
    artifact_root: Path = DEFAULT_ARTIFACT,
    report_root: Path = DEFAULT_REPORT,
    threads: int = 2,
) -> None:
    """Run reproducible experiments and write aggregate reports."""
    results = run(dataset_root, artifact_root, report_root, threads=threads)
    typer.echo(f"Completed {len(results)} methods; reports: {report_root}")


if __name__ == "__main__":
    typer.run(main)
