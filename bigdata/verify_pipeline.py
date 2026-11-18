#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["polars>=1.44,<2", "pydantic>=2.12,<3", "typer>=0.21,<1"]
# ///
# How to run: python -m bigdata.verify_pipeline --raw INPUT --report RUN_REPORT.
"""Independently verify the stored Hadoop pipeline results with Python."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Final

import polars as pl
import typer

from bigdata.contracts import (
    BOUNDS,
    SOURCES,
    CleaningAudit,
    RawRow,
    ReferenceResult,
    SparkReceipt,
    VerificationReceipt,
)

SPACE: Final = re.compile(r"[ \t\r\n\f\v]+")
DEFAULT_INPUT: Final = Path("data/curated/korean_v2/raw_input.parquet")
DEFAULT_REPORT: Final = Path("reports/bigdata_pipeline")


class AggregateFileError(ValueError):
    """Reject malformed counts instead of hiding duplicate aggregate keys."""

    def __init__(self, path: Path) -> None:
        """Identify the rejected result file without including input text."""
        super().__init__(f"invalid or duplicate aggregate entry in {path}")


def python_reference(raw: Path) -> ReferenceResult:
    """Apply the documented rules without Spark or its SQL implementation."""
    labels: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    texts: dict[str, str] = {}
    occurrences: Counter[str] = Counter()
    rows = valid_rows = 0
    frame = pl.scan_parquet(raw).select("text", "label", "source").collect()
    for values in frame.iter_rows(named=True):
        rows += 1
        row = RawRow.model_validate(values)
        text = SPACE.sub(" ", row.text or "").strip(" ")
        if not text or row.label not in {"clean", "harmful"}:
            continue
        if row.source not in SOURCES or row.source is None or row.label is None:
            continue
        valid_rows += 1
        digest = hashlib.sha256(text.encode()).hexdigest()
        labels.setdefault(digest, set()).add(row.label)
        sources.setdefault(digest, set()).add(row.source)
        texts[digest] = text
        occurrences[digest] += 1
    keys: list[str] = []
    aggregates: Counter[str] = Counter()
    conflict_rows = 0
    for digest, observed_labels in labels.items():
        if len(observed_labels) != 1:
            conflict_rows += occurrences[digest]
            continue
        label = next(iter(observed_labels))
        source = min(sources[digest])
        length = len(texts[digest])
        keys.append(f"{digest}|{label}|{source}")
        bucket = next((str(bound) for bound in BOUNDS if length <= bound), "overflow")
        for key in (
            "total",
            f"source:{source}",
            f"label:{label}",
            f"source_label:{source}:{label}",
            f"length:{bucket}",
        ):
            aggregates[key] += 1
    _ = aggregates.setdefault("total", 0)
    audit = CleaningAudit(
        bronze_rows=rows,
        invalid_rows=rows - valid_rows,
        conflict_rows=conflict_rows,
        duplicate_rows=valid_rows - conflict_rows - len(keys),
        silver_rows=len(keys),
    )
    return ReferenceResult(
        hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
        dict(aggregates),
        audit,
    )


def read_tsv(path: Path) -> dict[str, int]:
    """Read concatenated reducer parts and reject duplicated keys."""
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("\t")
        if not separator or not key or key in counts or not value.isdecimal():
            raise AggregateFileError(path)
        counts[key] = int(value)
    return counts


def verify(raw: Path, report: Path) -> VerificationReceipt:
    """Compare persisted stage receipts with independently computed results."""
    reference = python_reference(raw)
    spark = SparkReceipt.model_validate_json((report / "spark.json").read_text())
    mr = read_tsv(report / "mapreduce_aggregates.tsv")
    hdfs_bronze = (report / "hdfs-bronze.sha256").read_text().split()[0]
    with raw.open("rb") as stream:
        input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    checks = {
        "bronze_hdfs_sha256_equals_local": hdfs_bronze == input_sha256,
        "bronze_rows_equal": spark.bronze_rows == reference.audit.bronze_rows,
        "spark_output_keys_equal_python": spark.output_keys_sha256
        == reference.output_keys_sha256,
        "spark_aggregates_equal_python": spark.aggregates == reference.aggregates,
        "mapreduce_aggregates_equal_python": mr == reference.aggregates,
        "mapreduce_aggregates_equal_spark": mr == spark.aggregates,
    }
    if spark.audit is not None:
        checks["cleaning_audit_equal_python"] = spark.audit == reference.audit
    all_keys = set(reference.aggregates) | set(mr) | set(spark.aggregates)
    return VerificationReceipt(
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        bronze_rows=reference.audit.bronze_rows,
        silver_rows=reference.audit.silver_rows,
        aggregate_keys=len(reference.aggregates),
        input_sha256=input_sha256,
        python_output_keys_sha256=reference.output_keys_sha256,
        python_aggregates=dict(sorted(reference.aggregates.items())),
        audit=reference.audit,
        mismatched_keys=sorted(
            key
            for key in all_keys
            if not (
                reference.aggregates.get(key)
                == mr.get(key)
                == spark.aggregates.get(key)
            )
        ),
    )


def main(
    raw: Path = DEFAULT_INPUT,
    report: Path = DEFAULT_REPORT,
) -> None:
    """Write a receipt and exit nonzero when a checked result differs."""
    receipt = verify(raw, report)
    _ = (report / "verification.json").write_text(
        receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(receipt.model_dump_json())
    raise typer.Exit(code=0 if receipt.status == "passed" else 1)


if __name__ == "__main__":
    typer.run(main)
