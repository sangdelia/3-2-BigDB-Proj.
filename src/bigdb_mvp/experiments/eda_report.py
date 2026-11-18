"""Publish fresh aggregate-only EDA, preserving existing training artifacts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import typer
from pydantic import BaseModel

from bigdb_mvp.data.models import SourceName
from bigdb_mvp.experiments.dataset_types import DatasetError
from bigdb_mvp.experiments.eda_contracts import (
    CorpusManifest,
    Count,
    EdaReport,
    HadoopRuntime,
    Provenance,
    Runtime,
    SparkRuntime,
    Stage,
    Summary,
)
from bigdb_mvp.experiments.eda_profile import (
    category_counts,
    length_profile,
    missing_profile,
    overlap_count,
    retention_methods,
    source_summaries,
    split_summaries,
    validate_corpus,
)
from bigdb_mvp.experiments.preprocessing_benchmark import Timing

DATA_ROOT: Final = Path("data/curated/korean_v2")
REPORT_ROOT: Final = Path("reports/eda_20261029")
DASHBOARD_ROOT: Final = Path("eda_dashboard")


class HistoricalBenchmark(BaseModel):
    """Read measurements from the original executed Spark comparison."""

    timings: tuple[Timing, ...]
    startup_seconds: float


def preprocessing_stages(manifest: CorpusManifest) -> tuple[Stage, ...]:
    """Describe the existing run; never replay or mutate preprocessing."""
    conflict = manifest.dedup_exclusions["conflicting_binary_rows"]
    duplicates = manifest.dedup_exclusions["consistent_duplicate_rows"]
    after_conflicts = manifest.parsed_rows - conflict
    return (
        Stage(
            id="required",
            label="필수값·라벨 검사",
            input_rows=manifest.raw_rows,
            removed_rows=manifest.raw_rows - manifest.parsed_rows,
            output_rows=manifest.parsed_rows,
            description="K-HATERS 빈 문장 1건, UnSmile 라벨 누락·충돌 2건 제외",
        ),
        Stage(
            id="conflict",
            label="정답 충돌 제외",
            input_rows=manifest.parsed_rows,
            removed_rows=conflict,
            output_rows=after_conflicts,
            description=(
                "같은 공백 정규화 문장에 서로 다른 이진 정답이 붙은 묶음 전체 제외"
            ),
        ),
        Stage(
            id="dedup",
            label="일치 중복 병합",
            input_rows=after_conflicts,
            removed_rows=duplicates,
            output_rows=manifest.accepted_rows,
            description=(
                "같은 정답의 중복을 대표 문장으로 병합하고 원래 주석·출처는 보존"
            ),
        ),
    )


def runtime_status(report_root: Path) -> Runtime:
    """Use an execution receipt for Hadoop and original timings for Spark."""
    receipt = report_root / "hadoop" / "receipt.json"
    hadoop = (
        HadoopRuntime.model_validate_json(receipt.read_bytes())
        if receipt.exists()
        else HadoopRuntime(
            status="pending",
            label="실행 준비 중",
            reason=(
                "Docker 소켓 접근 불가. 별도 사용자 권한 Hadoop 환경을 준비하며 "
                "성공 기록은 아직 없음"
            ),
        )
    )
    saved = HistoricalBenchmark.model_validate_json(
        Path("reports/korean_v2/preprocessing.json").read_bytes()
    )
    largest = max(item.input_rows for item in saved.timings)
    times = {
        item.engine: item.median_seconds
        for item in saved.timings
        if item.input_rows == largest
    }
    return Runtime(
        hadoop=hadoop,
        spark=SparkRuntime(
            status="historical",
            label="2026-10-10 실제 실행 기록",
            scope=(
                "단일 서버 local[4], 공백 정리·충돌 제외·중복 제거. "
                "전체 전처리 및 다중 서버 성능이 아님"
            ),
            python_seconds=times["python_sequential"],
            spark_seconds=times["spark_local4"],
            startup_seconds=saved.startup_seconds,
        ),
    )


def build_report(
    data_root: Path = DATA_ROOT, report_root: Path = REPORT_ROOT
) -> EdaReport:
    """Measure the final corpus after binding it to its immutable manifest."""
    manifest = CorpusManifest.model_validate_json(
        (data_root / "manifest.json").read_bytes()
    )
    path = data_root / "corpus.parquet"
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != manifest.dataset_sha256:
        raise DatasetError(reason="eda_dataset_checksum_mismatch")
    with (data_root / "raw_input.parquet").open("rb") as handle:
        raw_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    frame = pl.scan_parquet(path).collect()
    validate_corpus(frame, manifest)
    missing = missing_profile(frame)
    if any(item.missing_rows for item in missing if item.kind == "required"):
        raise DatasetError(reason="eda_required_value_missing")
    if any(
        source.source == SourceName.CIVIL_COMMENTS.value for source in manifest.sources
    ):
        raise DatasetError(reason="eda_english_source_forbidden")
    summary = Summary(
        raw_rows=manifest.raw_rows,
        parsed_rows=manifest.parsed_rows,
        invalid_rows=manifest.raw_rows - manifest.parsed_rows,
        conflicting_rows=manifest.dedup_exclusions["conflicting_binary_rows"],
        duplicate_rows=manifest.dedup_exclusions["consistent_duplicate_rows"],
        final_rows=frame.height,
        text_overlap=overlap_count(frame, "text_sha256"),
        group_overlap=overlap_count(frame, "group_id"),
    )
    if (
        summary.parsed_rows - summary.conflicting_rows - summary.duplicate_rows
        != summary.final_rows
    ):
        raise DatasetError(reason="eda_pipeline_count_mismatch")
    return EdaReport(
        generated_at=datetime.now(UTC).isoformat(),
        provenance=Provenance(
            dataset_sha256=digest,
            raw_dataset_sha256=raw_digest,
            scope=(
                "한국어 6종 기존 정제 corpus 전체. 분포·결측·길이는 이번 실측, "
                "처리 전후 건수는 기존 manifest"
            ),
        ),
        summary=summary,
        pipeline=preprocessing_stages(manifest),
        sources=source_summaries(frame, manifest),
        labels=tuple(
            Count(label=name, count=frame.filter(pl.col("label") == name).height)
            for name in ("clean", "harmful")
        ),
        splits=split_summaries(frame),
        missing=missing,
        length=length_profile(frame),
        targets=category_counts(frame, "target_labels"),
        expressions=category_counts(frame, "expression_labels"),
        methods=retention_methods(frame),
        runtime=runtime_status(report_root),
        notes=(
            (
                "선택 주석의 빈 목록은 원천 미제공·비해당을 포함한다. "
                "결측 오류나 유형의 부정 정답으로 일괄 해석하지 않는다."
            ),
            (
                "정규화는 중복 판별용 공백 정리이며 저장된 원문을 보존했다. "
                "형태소 분석·불용어 삭제·임의 라벨 대체를 하지 않았다."
            ),
            (
                "출처 건수는 최종 대표 문장 기준이다. "
                "병합된 다른 출처의 원래 주석도 로컬 데이터에 보존된다."
            ),
            (
                "한국어 6종 내부 그룹 분할과 K-HATERS Ratings 공식 분할은 별도 경로다. "
                "이 화면은 6종 데이터만 분석한다."
            ),
            (
                "문장·그룹 교집합 0은 정확 중복·문서 연결 검사 결과이며 "
                "모든 의미·근사 중복 제거를 입증하지 않는다."
            ),
            (
                "선택 필드 완전 삭제는 같은 최종 corpus에 적용해 본 보존량 비교다. "
                "운영 데이터를 수정하거나 재학습하지 않았다."
            ),
            (
                "대상·표현 빈도는 원천 접두사를 보존한 상위 20개 주석이며 "
                "서로 배타적이거나 통일된 유형이 아니다."
            ),
        ),
    )


def main(
    data_root: Path = DATA_ROOT,
    report_root: Path = REPORT_ROOT,
    dashboard_root: Path = DASHBOARD_ROOT,
) -> None:
    """Write safe JSON/CSV plus a local-only Hadoop input without raw text."""
    report = build_report(data_root, report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    dashboard_root.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2) + "\n"
    _ = (report_root / "report.json").write_text(payload, encoding="utf-8")
    _ = (dashboard_root / "report.json").write_text(payload, encoding="utf-8")
    for name, records in (
        ("sources", report.sources),
        ("missing", report.missing),
        ("splits", report.splits),
        ("methods", report.methods),
    ):
        pl.DataFrame([record.model_dump() for record in records]).write_csv(
            report_root / f"{name}.csv"
        )
    local_root = Path("data/processed/eda_20261029")
    local_root.mkdir(parents=True, exist_ok=True)
    pl.scan_parquet(data_root / "corpus.parquet").select(
        "source",
        "label",
        pl.col("text").str.len_chars().alias("length"),
    ).sink_ndjson(local_root / "hadoop_input.jsonl")
    typer.echo(f"rows={report.summary.final_rows} report={report_root / 'report.json'}")


if __name__ == "__main__":
    typer.run(main)
