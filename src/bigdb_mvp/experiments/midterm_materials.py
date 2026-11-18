"""Publish comparable metrics and labeled figures from frozen experiment reports."""

from __future__ import annotations

import csv
from typing import ClassVar, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.experiments.khaters_ratings import ScoreKind
from bigdb_mvp.experiments.midterm_contracts import Metrics, TrialName
from bigdb_mvp.experiments.midterm_train import REPORT, Selection

LABELS: Final[dict[TrialName, str]] = {
    "baseline": "기준 모델",
    "normalized": "제한 정규화",
    "balanced": "불균형 보정",
}


class Followup(BaseModel):
    """Require explicit test provenance alongside every follow-up metric."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    evaluation_status: str
    model_sha256: str
    test: dict[ScoreKind, Metrics]


def main() -> None:
    """Generate the comparison table and two CJK figures from measured outputs."""
    selection = Selection.model_validate_json((REPORT / "selection.json").read_bytes())
    with (REPORT / "comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "trial",
                "split",
                "kind",
                "count",
                "mae_points",
                "grade_macro_f1",
                "grade_zero_fpr",
                "grade_one_recall",
                "temperature",
            )
        )
        for trial in selection.trials:
            test = Followup.model_validate_json(
                (REPORT / f"{trial.name}_followup.json").read_bytes()
            )
            for split, scores in (
                ("validation", trial.validation),
                ("test", test.test),
            ):
                for kind, value in scores.items():
                    writer.writerow(
                        (
                            trial.name,
                            split,
                            kind,
                            value.count,
                            value.mae_points,
                            value.grade_macro_f1,
                            value.grade_zero_fpr,
                            value.grade_one_recall,
                            value.temperature,
                        )
                    )
    mpl.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 14
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    names = [LABELS[trial.name] for trial in selection.trials]
    metrics = [trial.validation["hate"] for trial in selection.trials]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, values, title, limit in zip(
        axes,
        ([m.mae_points for m in metrics], [100 * m.grade_zero_fpr for m in metrics]),
        (
            "Hate 평점 평균 절대 오차 (점, 낮을수록 좋음)",
            "등급 0 오탐률 (%, 낮을수록 좋음)",
        ),
        (35.0, 25.0),
        strict=True,
    ):
        bars = ax.bar(names, values, color=["#263F32", "#738A7B", "#9F2F2D"])
        _ = ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5)
        _ = ax.set_title(title)
        _ = ax.set_ylim(0, max(limit, max(values) * 1.25))
    _ = fig.suptitle("동일 validation 9,997건에서 후보를 비교하고 선택")
    fig.savefig(figures / "validation_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    values = [100 * m.grade_one_recall for m in metrics]
    bars = ax.bar(names, values, color=["#263F32", "#738A7B", "#9F2F2D"])
    _ = ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=5)
    _ = ax.set_ylim(0, 100)
    _ = ax.set_ylabel("정답 등급 1을 등급 1로 예측한 비율 (%)")
    _ = ax.set_title("약한 혐오 재현율의 변화와 오탐을 함께 해석")
    fig.savefig(figures / "mild_recall.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
