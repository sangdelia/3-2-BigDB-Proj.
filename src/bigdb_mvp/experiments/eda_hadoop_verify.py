"""Verify real Hadoop results against independently computed Polars aggregates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import typer
from pydantic import TypeAdapter

from bigdb_mvp.experiments.dataset_types import DatasetError
from bigdb_mvp.experiments.eda_contracts import EdaReport, HadoopRuntime
from bigdb_mvp.experiments.eda_report import REPORT_ROOT

LENGTH_KEYS: Final = ("10", "20", "50", "100", "200", "500", "1000", "overflow")


def main(report_root: Path = REPORT_ROOT) -> None:
    """Publish a success receipt only after exact aggregate and HDFS checks."""
    root = Path(report_root, "hadoop")
    report = EdaReport.model_validate_json((report_root / "report.json").read_bytes())
    expected = {"total": report.summary.final_rows}
    for source in report.sources:
        expected[f"source:{source.source}"] = source.final_rows
        expected[f"source_label:{source.source}:clean"] = source.clean
        expected[f"source_label:{source.source}:harmful"] = source.harmful
    for item in report.labels:
        expected[f"label:{item.label}"] = item.count
    for key, item in zip(LENGTH_KEYS, report.length.histogram, strict=True):
        expected[f"length:{key}"] = item.count
    expected = {key: value for key, value in expected.items() if value}
    observed: dict[str, int] = {}
    adapter = TypeAdapter(tuple[str, int])
    for line in (root / "aggregates.tsv").read_text().splitlines():
        key, value = adapter.validate_python(line.split("\t"))
        if key in observed:
            raise DatasetError(reason="hadoop_duplicate_output_key")
        observed[key] = value
    if observed != expected:
        raise DatasetError(reason="hadoop_eda_aggregate_mismatch")
    digest = (root / "hdfs-corpus.sha256").read_text().split()[0]
    if digest != report.provenance.dataset_sha256:
        raise DatasetError(reason="hadoop_hdfs_checksum_mismatch")
    evidence = {
        "status": "verified",
        "rows": report.summary.final_rows,
        "matched_aggregate_keys": len(expected),
        "hdfs_corpus_sha256": digest,
        "checks": ["row_count", "source", "label", "source_label", "length_histogram"],
        "scope": (
            "정제 후 전체 corpus의 EDA 집계. "
            "원본 정제·분할을 Hadoop으로 재구현한 결과는 아님"
        ),
    }
    _ = (root / "verification.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = HadoopRuntime(
        status="completed",
        label="HDFS 저장·MapReduce EDA 실행 완료",
        reason=(
            f"Hadoop 3.4.2 단일 서버 HDFS + MapReduce LocalJobRunner. "
            f"{report.summary.final_rows:,}건, {len(expected)}개 집계와 "
            "파일 SHA-256 일치. "
            "YARN·다중 서버 성능 실험은 수행하지 않음"
        ),
    )
    _ = (root / "receipt.json").write_text(
        receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"verified rows={report.summary.final_rows} keys={len(expected)}")


if __name__ == "__main__":
    typer.run(main)
