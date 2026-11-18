"""Compare one identical preprocessing workload on sequential Python and Spark."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import resource
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar, Final

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

SPACE: Final = re.compile(r"[ \t\r\n\f\v]+")
REPEATS: Final = 3
DATASET_ROOT: Final = Path("data/curated/korean_v2")
REPORT_ROOT: Final = Path("reports/korean_v2")


class InputRow(BaseModel):
    """Parse only the three fields used by both processing engines."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    text: str
    label: str
    source: str


class Timing(BaseModel):
    """One natural-size input and its repeated wall-time measurements."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    engine: str
    input_rows: int
    output_rows: int
    output_sha256: str
    seconds: tuple[float, ...]
    median_seconds: float
    includes_parquet_read_and_collection: bool = True


def sequential_keys(path: Path) -> tuple[str, ...]:
    """Normalize, reject conflicts, deduplicate, and choose a stable source."""
    labels: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for raw in pl.read_parquet(path).iter_rows(named=True):
        row = InputRow.model_validate(raw)
        text = SPACE.sub(" ", row.text).strip(" ")
        if not text or row.label not in {"clean", "harmful"}:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        labels.setdefault(digest, set()).add(row.label)
        sources.setdefault(digest, set()).add(row.source)
    return tuple(
        sorted(
            f"{digest}|{next(iter(values))}|{min(sources[digest])}"
            for digest, values in labels.items()
            if len(values) == 1
        )
    )


def spark_keys(spark: SparkSession, path: Path) -> tuple[str, ...]:
    """Execute equivalent transformations using Spark SQL without Python UDFs."""
    spark.read.parquet(str(path.resolve())).createOrReplaceTempView("benchmark_input")
    rows = spark.sql(
        r"""
        WITH normalized AS (
          SELECT trim(regexp_replace(text, '[ \\t\\r\\n\\f\\x0B]+', ' ')) AS text,
                 label, source FROM benchmark_input
        ), valid AS (
          SELECT sha2(text,256) AS digest,label,source FROM normalized
          WHERE text <> '' AND label IN ('clean','harmful')
        ), collapsed AS (
          SELECT digest,min(label) AS label,min(source) AS source
          FROM valid GROUP BY digest HAVING count(DISTINCT label)=1
        )
        SELECT concat_ws('|',digest,label,source) AS output_key
        FROM collapsed ORDER BY output_key
        """
    ).collect()
    return tuple(row["output_key"] for row in rows)


def output_digest(keys: tuple[str, ...]) -> str:
    """Hash complete ordered results, not just aggregate counts."""
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def measure(path: Path, spark: SparkSession | None) -> Timing:
    """Measure three full actions with a stable output contract."""
    times: list[float] = []
    digests: set[str] = set()
    count = 0
    for _ in range(REPEATS):
        start = perf_counter()
        keys = sequential_keys(path) if spark is None else spark_keys(spark, path)
        times.append(perf_counter() - start)
        digests.add(output_digest(keys))
        count = len(keys)
    if len(digests) != 1:
        message = "benchmark_nondeterministic_output"
        raise RuntimeError(message)
    return Timing(
        engine="python_sequential" if spark is None else "spark_local4",
        input_rows=pl.read_parquet(path, columns=["label"]).height,
        output_rows=count,
        output_sha256=next(iter(digests)),
        seconds=tuple(times),
        median_seconds=median(times),
    )


def main(
    dataset_root: Path = DATASET_ROOT,
    report_root: Path = REPORT_ROOT,
) -> None:
    """Run real local-mode Spark comparison and write safe timing artifacts."""
    from pyspark.sql import SparkSession  # noqa: PLC0415

    raw_path = dataset_root / "raw_input.parquet"
    raw = pl.read_parquet(raw_path).select("text", "label", "source")
    report_root.mkdir(parents=True, exist_ok=True)
    temp_root = dataset_root / "benchmark_inputs"
    temp_root.mkdir(parents=True, exist_ok=True)
    startup = perf_counter()
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("BigDB-Korean-preprocessing-v2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    startup_seconds = perf_counter() - startup
    timings: list[Timing] = []
    try:
        for size in sorted(
            {min(10_000, raw.height), min(100_000, raw.height), raw.height}
        ):
            path = temp_root / f"input-{size}.parquet"
            raw.head(size).write_parquet(path)
            python = measure(path, None)
            # A distinct warm-up is excluded; startup is reported separately.
            warmup = spark_keys(spark, path)
            measured = measure(path, spark)
            if (
                output_digest(warmup) != python.output_sha256
                or measured.output_sha256 != python.output_sha256
            ):
                message = "python_spark_output_mismatch"
                raise RuntimeError(message)
            timings.extend((python, measured))
        payload = {
            "status": "complete",
            "workload": "whitespace_normalize_conflict_filter_exact_dedup",
            "scope": "accepted_source_rows_not_full_ingest_or_group_split",
            "cluster_mode": "single_host_local4_not_multinode",
            "startup_seconds": startup_seconds,
            "spark_version": spark.version,
            "python_version": platform.python_version(),
            "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "output_equivalence": True,
            "python_process_cumulative_peak_rss_mib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / 1024,
            "memory_limitation": "cumulative Python RSS; JVM memory not measured",
            "timings": [item.model_dump() for item in timings],
        }
        _ = (report_root / "preprocessing.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (report_root / "preprocessing.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "engine",
                    "input_rows",
                    "output_rows",
                    "median_seconds",
                    "output_sha256",
                )
            )
            writer.writerows(
                (
                    item.engine,
                    item.input_rows,
                    item.output_rows,
                    item.median_seconds,
                    item.output_sha256,
                )
                for item in timings
            )
        typer.echo(json.dumps(payload, ensure_ascii=False))
    finally:
        spark.stop()


if __name__ == "__main__":
    typer.run(main)
