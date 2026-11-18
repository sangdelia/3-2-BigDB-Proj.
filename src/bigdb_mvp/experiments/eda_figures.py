"""Export presentation figures from the same aggregate contract as the web UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import typer
from matplotlib.backends.backend_pdf import PdfPages

from bigdb_mvp.experiments.eda_contracts import EdaReport

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

INK: Final = "#242822"
ACCENT: Final = "#263f32"
MUTED: Final = "#62665e"
CANVAS: Final = "#f7f6f3"
SECONDARY: Final = "#b7c8b1"
REPORT_ROOT: Final = Path("reports/eda_20261029")


def chart(title: str, xlabel: str) -> tuple[Figure, Axes]:
    """Apply the existing BigDB presentation palette and Korean typography."""
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    fig.set_facecolor(CANVAS)
    ax.set_facecolor(CANVAS)
    _ = ax.set_title(title, loc="left", fontsize=21, fontweight="bold", pad=20)
    _ = ax.set_xlabel(xlabel, labelpad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=13)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    return fig, ax


def sources_figure(report: EdaReport) -> Figure:
    """Expose source concentration together with class imbalance."""
    fig, ax = chart("원천별 규모와 라벨 구성이 다르다", "최종 대표 문장 수 (건)")
    rows = tuple(reversed(report.sources))
    names = [row.name for row in rows]
    _ = ax.barh(names, [row.clean for row in rows], color=SECONDARY, label="clean")
    _ = ax.barh(
        names,
        [row.harmful for row in rows],
        left=[row.clean for row in rows],
        color=ACCENT,
        label="harmful",
    )
    _ = ax.legend(frameon=False, loc="lower right")
    ax.xaxis.set_major_formatter("{x:,.0f}")
    _ = fig.supxlabel(
        "출처는 대표 행 기준이며 병합된 원주석의 출처 수와 다름",
        fontsize=11,
        color=MUTED,
    )
    return fig


def missing_figure(report: EdaReport) -> Figure:
    """Explain optional annotation availability without calling it label error."""
    fig, ax = chart(
        "필수값은 충족했지만 선택 주석 제공 범위는 다르다", "빈 값·빈 목록 비율 (%)"
    )
    rows = tuple(reversed(report.missing))
    bars = ax.barh(
        [row.label for row in rows], [row.rate * 100 for row in rows], color=ACCENT
    )
    _ = ax.bar_label(bars, labels=[f"{row.rate * 100:.1f}%" for row in rows], padding=5)
    _ = ax.set_xlim(0, 100)
    _ = fig.supxlabel(
        "최종 311,661건 기준. 선택 주석 미제공·비해당은 정답의 결측 오류와 구분",
        fontsize=11,
        color=MUTED,
    )
    return fig


def lengths_figure(report: EdaReport) -> Figure:
    """Show the observed original-string length distribution."""
    stats = report.length.stats
    fig, ax = chart(
        f"원문 길이 중앙값 {stats.median:g}자 · 95백분위 {stats.p95:g}자",
        "문장 수 (건)",
    )
    rows = tuple(reversed(report.length.histogram))
    bars = ax.barh(
        [row.label for row in rows], [row.count for row in rows], color=ACCENT
    )
    _ = ax.bar_label(bars, labels=[f"{row.count:,}" for row in rows], padding=5)
    _ = ax.set_xlim(0, max(row.count for row in rows) * 1.2)
    ax.xaxis.set_major_formatter("{x:,.0f}")
    _ = fig.supxlabel(
        "원문 Unicode code point 수, 공백·구두점 포함. 토큰 수·바이트 수와 구분",
        fontsize=11,
        color=MUTED,
    )
    return fig


def retention_figure(report: EdaReport) -> Figure:
    """Compare measured retention with one fixed final-corpus denominator."""
    selected, comparison = report.methods
    removed_rate = comparison.removed_rows / comparison.start_rows * 100
    fig, ax = chart(
        f"선택 필드 완전 삭제는 문장의 {removed_rate:.1f}%를 제외한다",
        "동일 최종 데이터에서 남는 문장 수 (건)",
    )
    bars = ax.barh(
        ["선택 필드까지 완전 삭제\n비교용·미채택", "선택 주석 보존\n채택"],
        [comparison.retained_rows, selected.retained_rows],
        color=[SECONDARY, ACCENT],
    )
    _ = ax.bar_label(
        bars,
        labels=[f"{comparison.retained_rows:,}", f"{selected.retained_rows:,}"],
        padding=6,
    )
    _ = ax.set_xlim(0, selected.start_rows * 1.2)
    ax.xaxis.set_major_formatter("{x:,.0f}")
    _ = fig.supxlabel(
        (
            "문맥·대상·표현 주석을 모두 요구한 비교. "
            "실제 데이터 삭제·라벨 대체·재학습 없음"
        ),
        fontsize=11,
        color=MUTED,
    )
    return fig


def main(report_root: Path = REPORT_ROOT) -> None:
    """Write four standalone PNGs and a four-page shareable PDF."""
    report = EdaReport.model_validate_json((report_root / "report.json").read_bytes())
    mpl.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 13,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    output = report_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_root / "EDA_전처리_그래프.pdf") as pdf:
        for name, builder in (
            ("01_source_labels", sources_figure),
            ("02_missing", missing_figure),
            ("03_lengths", lengths_figure),
            ("04_retention", retention_figure),
        ):
            fig = builder(report)
            fig.savefig(output / f"{name}.png", dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    typer.echo(str(output))


if __name__ == "__main__":
    typer.run(main)
