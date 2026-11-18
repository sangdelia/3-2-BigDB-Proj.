"""Publish the validation-selected artifact under a separate release directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from bigdb_mvp.experiments.classification_contracts import ExperimentError
from bigdb_mvp.experiments.midterm_contracts import eligible
from bigdb_mvp.experiments.midterm_evaluate import load_trial
from bigdb_mvp.experiments.midterm_materials import Followup
from bigdb_mvp.experiments.midterm_train import ARTIFACT, REPORT, Selection


def main() -> None:
    """Verify the frozen selection, retain v1, and publish a reversible v2 pointer."""
    selected = Selection.model_validate_json((REPORT / "selection.json").read_bytes())
    trial = next(t for t in selected.trials if t.name == selected.selected)
    if not eligible(trial.validation["hate"], selected.trials[0].validation["hate"]):
        reason = "midterm_candidate_does_not_meet_validation_gate"
        raise ExperimentError(reason)
    _ = load_trial(trial)
    followup = Followup.model_validate_json(
        (REPORT / f"{trial.name}_followup.json").read_bytes()
    )
    if followup.model_sha256 != trial.model_sha256:
        reason = "midterm_followup_artifact_mismatch"
        raise ExperimentError(reason)
    artifact = Path("artifacts/khaters_score_v2/model.joblib")
    report = Path("reports/khaters_score_v2/metrics.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(ARTIFACT / f"{trial.name}.joblib", artifact)
    _ = report.write_text(
        json.dumps(
            {
                "model_sha256": trial.model_sha256,
                "dataset_digest": trial.dataset_digest,
                "revision": trial.revision,
                "method": trial.name,
                "selection_report": str(REPORT / "selection.json"),
                "formula": "50 * (P(grade=1) + 2 * P(grade=2))",
                "retained_counts": selected.audit.retained,
                "results": {
                    kind: {
                        "validation": value.model_dump(),
                        "test": followup.test[kind].model_dump(),
                    }
                    for kind, value in trial.validation.items()
                },
                "test_role": "previously seen follow-up evaluation; no model selection",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _ = Path("reports/khaters_score_selection.json").write_text(
        json.dumps({"version": "khaters_score_v2"}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
