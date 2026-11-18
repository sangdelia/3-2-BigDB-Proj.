"""Train fixed candidates and freeze their selection before test evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import ClassVar, Final, assert_never

import joblib
import typer
from pydantic import BaseModel, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.khaters_ratings import (
    KINDS,
    REVISION,
    ScoreKind,
    load_dataset,
)
from bigdb_mvp.experiments.khaters_score_train import ROOT, calibrated, evaluate
from bigdb_mvp.experiments.midterm_contracts import (
    Audit,
    CommonData,
    Metrics,
    TrialName,
    common_rows,
    eligible,
    normalize_features,
)
from bigdb_mvp.khaters_score import Head, ScoreModel

REPORT: Final = Path("reports/midterm_20261104")
ARTIFACT: Final = Path("artifacts/midterm_20261104")
TRIALS: Final[tuple[TrialName, ...]] = ("baseline", "normalized", "balanced")


class Trial(BaseModel):
    """Bind validation evidence to the exact persisted estimator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: TrialName
    model_sha256: str
    dataset_digest: str
    revision: str
    validation: dict[ScoreKind, Metrics]
    elapsed_seconds: float


class Selection(BaseModel):
    """Persist the decision using only validation metrics and common-row hashes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    selected: TrialName
    eligible_candidates: tuple[TrialName, ...]
    audit: Audit
    trials: tuple[Trial, ...]
    rule: str
    evaluation_status: str


def fit_trial(name: TrialName, data: CommonData) -> Trial:
    """Fit one predeclared treatment on train and calibrate on validation."""
    started = time.monotonic()
    training, validation, _testing = data.splits
    match name:
        case "baseline":
            preprocessor, weight = None, None
        case "normalized":
            preprocessor, weight = normalize_features, None
        case "balanced":
            preprocessor, weight = None, "balanced"
        case unreachable:
            assert_never(unreachable)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        max_features=100000,
        preprocessor=preprocessor,
    )
    train_matrix = vectorizer.fit_transform(tuple(row.text for row in training))
    val_matrix = vectorizer.transform(tuple(row.text for row in validation))
    heads: list[Head] = []
    scores: dict[ScoreKind, Metrics] = {}
    for kind in KINDS:
        typer.echo(f"Fitting {name}/{kind}: {len(training)} train rows")
        classifier = LogisticRegression(
            class_weight=weight, solver="lbfgs", max_iter=1000, random_state=20261014
        )
        with threadpool_limits(limits=2):
            _ = classifier.fit(train_matrix, tuple(row.grade(kind) for row in training))
        head = calibrated(
            Head(kind, classifier, 1),
            val_matrix,
            tuple(row.grade(kind) for row in validation),
        )
        heads.append(head)
        scores[kind] = Metrics.model_validate_json(
            evaluate(head, val_matrix, validation, training)
        )
    digest = hashlib.sha256("".join(data.audit.split_sha256).encode()).hexdigest()
    model = ScoreModel(vectorizer, tuple(heads), digest, REVISION)
    path = ARTIFACT / f"{name}.joblib"
    _ = joblib.dump(model, path)
    trial = Trial(
        name=name,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dataset_digest=digest,
        revision=REVISION,
        validation=scores,
        elapsed_seconds=time.monotonic() - started,
    )
    _ = (REPORT / f"{name}_validation.json").write_text(
        trial.model_dump_json(indent=2), encoding="utf-8"
    )
    return trial


def main() -> None:
    """Run the fixed comparison without computing any test prediction."""
    dataset = load_dataset(ROOT)
    data = common_rows(dataset)
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    _ = (REPORT / "data_audit.json").write_text(
        data.audit.model_dump_json(indent=2), encoding="utf-8"
    )
    typer.echo(data.audit.model_dump_json(indent=2))
    trials = tuple(fit_trial(name, data) for name in TRIALS)
    accepted = tuple(
        trial
        for trial in trials[1:]
        if eligible(trial.validation["hate"], trials[0].validation["hate"])
    )
    chosen = min(
        accepted or (trials[0],), key=lambda trial: trial.validation["hate"].mae_points
    )
    selection = Selection(
        selected=chosen.name,
        eligible_candidates=tuple(t.name for t in accepted),
        audit=data.audit,
        trials=trials,
        rule="Hate MAE lower AND macro-F1 >= baseline AND grade-0 FPR <= baseline",
        evaluation_status="validation selection frozen; test predictions not computed",
    )
    _ = (REPORT / "selection.json").write_text(
        selection.model_dump_json(indent=2), encoding="utf-8"
    )
    typer.echo(
        json.dumps({"selected": chosen.name, "eligible": [t.name for t in accepted]})
    )


if __name__ == "__main__":
    main()
