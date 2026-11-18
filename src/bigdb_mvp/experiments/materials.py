"""Export aggregate-only CSV tables and presentation figures from measured results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import typer
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import BaseModel, ConfigDict

from bigdb_mvp.experiments.classification_contracts import MethodResult
from bigdb_mvp.experiments.preprocessing_benchmark import Timing

REPORT_ROOT: Final = Path("reports/korean_v2")
DATASET_ROOT: Final = Path("data/curated/korean_v2")


class Source(BaseModel):
    """Original input inventory counts; no text is included."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    source: str
    raw_rows: int
    accepted_rows: int


class Inventory(BaseModel):
    """Safe source and partition aggregates from the dataset manifest."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    sources: tuple[Source, ...]
    source_counts: dict[str, int]
    split_counts: dict[str, int]


class Comparison(BaseModel):
    """Measured validation and test results with configuration stored separately."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    results: tuple[MethodResult, ...]


class Processing(BaseModel):
    """Measured wall times of identical preprocessing actions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    timings: tuple[Timing, ...]
    output_equivalence: bool


def main(
    dataset_root: Path = DATASET_ROOT,
    report_root: Path = REPORT_ROOT,
) -> None:
    """Generate PPT-ready figures and PDF after both real experiment runs finish."""
    inventory = Inventory.model_validate_json(
        (dataset_root / "manifest.json").read_bytes()
    )
    comparison = Comparison.model_validate_json(
        (report_root / "metrics.json").read_bytes()
    )
    processing = Processing.model_validate_json(
        (report_root / "preprocessing.json").read_bytes()
    )
    if not processing.output_equivalence:
        message = "unverified_preprocessing_comparison"
        raise RuntimeError(message)
    mpl.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = 13
    figures = report_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_tables(report_root, inventory, comparison)
    _write_figures(report_root, figures, inventory, comparison, processing)
    typer.echo(str(report_root / "experiment_figures.pdf"))


def _write_tables(
    report_root: Path, inventory: Inventory, comparison: Comparison
) -> None:
    with (report_root / "dataset_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("source", "original_rows", "parsed_rows", "final_representative_rows")
        )
        writer.writerows(
            (s.source, s.raw_rows, s.accepted_rows, inventory.source_counts[s.source])
            for s in inventory.sources
        )
    with (report_root / "test_by_source.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "method",
                "source",
                "count",
                "macro_f1",
                "harmful_precision",
                "harmful_recall",
                "false_positive_rate",
            )
        )
        writer.writerows(
            (
                r.method,
                source,
                m.count,
                m.macro_f1,
                m.harmful_precision,
                m.harmful_recall,
                m.false_positive_rate,
            )
            for r in comparison.results
            for source, m in r.test_by_source.items()
        )


def _write_figures(
    report_root: Path,
    figures: Path,
    inventory: Inventory,
    comparison: Comparison,
    processing: Processing,
) -> None:
    with PdfPages(report_root / "experiment_figures.pdf") as pdf:
        source_order = sorted(
            inventory.source_counts, key=inventory.source_counts.__getitem__
        )
        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        bars = ax.barh(
            source_order,
            [inventory.source_counts[s] for s in source_order],
            color="#3566a0",
        )
        _ = ax.bar_label(bars, fmt="%d", padding=5)
        _ = ax.set_xlim(0, max(inventory.source_counts.values()) * 1.18)
        _ = ax.set_xlabel("충돌·중복 처리 후 대표 문장 수 (건)")
        _ = ax.set_title("한국어 6종 통합 데이터: 출처별 규모")
        fig.savefig(figures / "01_source_counts.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
        pairs = (
            (comparison.results[:2], ("기본 분류기", "사전 특징 추가"), "전체 평가셋"),
            (
                comparison.results[2:],
                ("댓글만 입력", "기사 제목+댓글"),
                "동일 문맥 보유 부분집합",
            ),
        )
        for ax, (results, names, title) in zip(axes, pairs, strict=True):
            values = [r.test.macro_f1 for r in results]
            bars = ax.bar(names, values, color=["#3566a0", "#429b8b"])
            _ = ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5)
            _ = ax.set_ylim(0, 1)
            _ = ax.set_ylabel("최종 test Macro-F1")
            _ = ax.set_title(f"{title} · n={results[0].test.count:,}")
        _ = fig.suptitle("사전 특징과 실제 기사 제목의 효과를 별도 대조군으로 비교")
        fig.savefig(figures / "02_classifier_comparison.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        for engine, label in (
            ("python_sequential", "Python 순차"),
            ("spark_local4", "Spark local[4]"),
        ):
            items = [t for t in processing.timings if t.engine == engine]
            _ = ax.plot(
                [t.input_rows for t in items],
                [t.median_seconds for t in items],
                marker="o",
                label=label,
            )
        ax.set_yscale("log")
        _ = ax.set_xlabel("실제 입력 문장 수 (건)")
        _ = ax.set_ylabel("읽기·처리·결과 수집 시간: 3회 중앙값 (초, 로그 축)")
        _ = ax.set_title("동일한 전처리 결과를 만드는 순차 처리와 로컬 Spark 비교")
        _ = ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(figures / "03_preprocessing_time.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        parts = ["train", "validation", "test"]
        bars = ax.bar(
            parts, [inventory.split_counts[p] for p in parts], color="#3566a0"
        )
        _ = ax.bar_label(bars, fmt="%d", padding=5)
        _ = ax.set_ylim(0, max(inventory.split_counts.values()) * 1.15)
        _ = ax.set_ylabel("대표 문장 수 (건)")
        _ = ax.set_title("문장·기사·대화 그룹을 분리한 내부 학습·검증·평가셋")
        fig.savefig(figures / "04_split_counts.png", dpi=180)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
