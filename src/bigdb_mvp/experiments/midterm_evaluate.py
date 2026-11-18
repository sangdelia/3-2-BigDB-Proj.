"""Evaluate frozen midterm candidates without reopening model selection."""

from __future__ import annotations

import csv
import hashlib
import json
from typing import assert_never

import joblib
import typer

from bigdb_mvp.baseline import BaselineModel
from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.experiments.hypothesis_cases import cases
from bigdb_mvp.experiments.khaters_ratings import load_dataset, text_key
from bigdb_mvp.experiments.khaters_score_train import ROOT, evaluate
from bigdb_mvp.experiments.midterm_contracts import CommonData, Metrics, common_rows
from bigdb_mvp.experiments.midterm_train import ARTIFACT, REPORT, Selection, Trial
from bigdb_mvp.khaters_score import ScoreModel, predict, probabilities


def load_trial(trial: Trial) -> ScoreModel:
    """Require the frozen checksum and row provenance before trusted deserialization."""
    path = ARTIFACT / f"{trial.name}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != trial.model_sha256:
        reason = "midterm_artifact_checksum_mismatch"
        raise ExperimentError(reason)
    loaded = joblib.load(path, mmap_mode="c")
    match loaded:
        case ScoreModel():
            if (loaded.dataset_digest, loaded.revision) != (
                trial.dataset_digest,
                trial.revision,
            ):
                reason = "midterm_artifact_provenance_mismatch"
                raise ExperimentError(reason)
            return loaded
        case BaselineModel() | Model():
            reason = "midterm_artifact_type_mismatch"
            raise ExperimentError(reason)
        case unreachable:
            assert_never(unreachable)


def evaluate_trial(trial: Trial, data: CommonData) -> None:
    """Export hashed errors and previously seen test metrics for each fixed model."""
    model = load_trial(trial)
    training, validation, testing = data.splits
    test_matrix = model.vectorizer.transform(tuple(row.text for row in testing))
    test_metrics = {
        head.kind: Metrics.model_validate_json(
            evaluate(head, test_matrix, testing, training)
        )
        for head in model.heads
    }
    _ = (REPORT / f"{trial.name}_followup.json").write_text(
        json.dumps(
            {
                "evaluation_status": "previously seen test; follow-up comparison only",
                "model_sha256": trial.model_sha256,
                "test": {
                    kind: value.model_dump() for kind, value in test_metrics.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (REPORT / f"{trial.name}_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "split",
                "text_sha256",
                "kind",
                "gold",
                "predicted",
                "score",
                "recognized_features",
                "characters",
            )
        )
        for split, rows in (("validation", validation), ("test", testing)):
            matrix = model.vectorizer.transform(tuple(row.text for row in rows))
            counts = matrix.getnnz(axis=1)
            for head in model.heads:
                values = probabilities(head, matrix)
                for row, value, count in zip(rows, values, counts, strict=True):
                    label = max(
                        range(3), key=lambda i: (value.zero, value.one, value.two)[i]
                    )
                    writer.writerow(
                        (
                            split,
                            text_key(row.text),
                            head.kind,
                            row.grade(head.kind),
                            label,
                            value.score,
                            int(count),
                            len(row.text),
                        )
                    )
    with (REPORT / f"{trial.name}_authored.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "case_id",
                "category",
                "text",
                "intended_binary",
                "recognized_features",
                "hate_score",
                "abuse_score",
                "swear_score",
            )
        )
        for case in cases():
            result = predict(model, case.text)
            writer.writerow(
                (
                    case.case_id,
                    case.category,
                    case.text,
                    case.expected,
                    result.recognized_features,
                    result.hate_score,
                    result.abuse_score,
                    result.swear_score,
                )
            )
    typer.echo(f"Follow-up evaluation saved: {trial.name}")


def main() -> None:
    """Require an existing validation decision and identical row hashes."""
    path = REPORT / "selection.json"
    selection_bytes = path.read_bytes()
    selection = Selection.model_validate_json(selection_bytes)
    data = common_rows(load_dataset(ROOT))
    if data.audit != selection.audit:
        reason = "midterm_common_rows_changed"
        raise ExperimentError(reason)
    for trial in selection.trials:
        evaluate_trial(trial, data)
    _ = (REPORT / "evaluation_receipt.json").write_text(
        json.dumps(
            {
                "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                "selected": selection.selected,
                "test_role": "follow-up only; no retuning",
                "diagnostics": "71 authored cases; no independent ordinal gold labels",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Frozen selection retained: {selection.selected}")


if __name__ == "__main__":
    main()
