#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pyspark==3.5.7", "pydantic>=2.12,<3", "typer>=0.21,<1"]
# ///
# How to run: bash bigdata/run_pipeline.sh (sets the runtime and HDFS paths).
"""Clean HDFS Bronze with Spark and persist Silver and Gold in HDFS."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import typer
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from bigdata.contracts import (
    SOURCES,
    AggregateRow,
    CleaningAudit,
    HdfsPaths,
    SilverKey,
    SparkReceipt,
    SparkSettings,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.readwriter import DataFrameReader

VALID_SQL: Final = r"""
WITH normalized AS (
  SELECT trim(regexp_replace(text, '[ \\t\\r\\n\\f\\x0B]+', ' ')) AS text,
         label, source FROM bronze
)
SELECT text, sha2(text, 256) AS digest, label, source FROM normalized
WHERE text <> '' AND label IN ('clean', 'harmful')
"""
GROUP_SQL: Final = """
SELECT digest, min(text) AS text, min(label) AS label, min(source) AS source,
       count(*) AS occurrences, count(DISTINCT label) AS labels
FROM valid GROUP BY digest
"""
SILVER_SQL: Final = """
SELECT digest, text, label, source, char_length(text) AS length
FROM grouped WHERE labels = 1
"""
BUCKET: Final = """CASE
  WHEN length <= 10 THEN '10' WHEN length <= 20 THEN '20'
  WHEN length <= 50 THEN '50' WHEN length <= 100 THEN '100'
  WHEN length <= 200 THEN '200' WHEN length <= 500 THEN '500'
  WHEN length <= 1000 THEN '1000' ELSE 'overflow' END"""
GOLD_SQL: Final = """
SELECT 'total' AS key, count(*) AS n FROM silver
UNION ALL SELECT concat('source:', source), count(*) FROM silver GROUP BY source
UNION ALL SELECT concat('label:', label), count(*) FROM silver GROUP BY label
UNION ALL SELECT concat('source_label:', source, ':', label), count(*)
  FROM silver GROUP BY source, label
UNION ALL SELECT concat('length:', bucket), count(*) FROM silver GROUP BY bucket
"""
AUDIT_SQL: Final = """
SELECT :bronze_rows AS bronze_rows,
  :bronze_rows - coalesce(sum(occurrences), 0) AS invalid_rows,
  coalesce(sum(CASE WHEN labels > 1 THEN occurrences ELSE 0 END), 0) AS conflict_rows,
  coalesce(sum(CASE WHEN labels = 1 THEN occurrences - 1 ELSE 0 END), 0)
    AS duplicate_rows,
  count(CASE WHEN labels = 1 THEN 1 END) AS silver_rows
FROM grouped
"""
DEFAULT_REPORT: Final = Path("reports/bigdata_pipeline")


class SparkEngine(Protocol):
    """Narrow the SQL boundary left partially untyped by PySpark 3.5."""

    def sql(self, query: str, /, *, args: dict[str, int] | None = None) -> DataFrame:
        """Execute fixed SQL with separately bound integer parameters."""
        ...

    @property
    def read(self) -> DataFrameReader:
        """Expose Spark's typed HDFS input reader."""
        ...

    @property
    def version(self) -> str:
        """Record the engine version in the run receipt."""
        ...

    def stop(self) -> None:
        """Release the local Spark runtime at the CLI boundary."""
        ...


def session(fs: str, settings: SparkSettings) -> SparkEngine:
    """Use bounded local threads without changing host resource limits."""
    jvm = (
        f"-XX:ActiveProcessorCount={settings.workers} -XX:+UseSerialGC "
        "-XX:CICompilerCount=2 -Xss512k"
    )
    return (
        SparkSession.Builder()
        .master(f"local[{settings.workers}]")
        .appName("BigDB-HDFS-Spark-pipeline")
        .config("spark.hadoop.fs.defaultFS", fs)
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.extraJavaOptions", jvm)
        .config("spark.sql.shuffle.partitions", str(settings.partitions))
        .config("spark.default.parallelism", str(settings.partitions))
        .config("spark.rpc.netty.dispatcher.numThreads", "2")
        .config("spark.rpc.io.serverThreads", "1")
        .config("spark.shuffle.io.serverThreads", "1")
        .config("spark.shuffle.io.clientThreads", "1")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def main(
    fs: str = "hdfs://127.0.0.1:19310",
    ns: str = "/bigdb/pipeline",
    report: Path = DEFAULT_REPORT,
) -> None:
    """Read Bronze, clean globally, and write run-scoped HDFS datasets."""
    settings = SparkSettings.model_validate(
        {
            "workers": os.environ.get("BIGDB_SPARK_WORKERS", "2"),
            "partitions": os.environ.get("BIGDB_PARTITIONS", "4"),
        }
    )
    paths = HdfsPaths(
        bronze=f"{ns}/bronze/raw_input.parquet",
        silver=f"{ns}/silver/corpus",
        mr_input=f"{ns}/silver/mr_input",
        gold=f"{ns}/gold/spark_aggregates",
    )
    report.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spark = session(fs, settings)
    startup = time.perf_counter() - started
    try:
        job_started = time.perf_counter()
        bronze = spark.read.parquet(paths.bronze).select("text", "label", "source")
        bronze.createOrReplaceTempView("bronze")
        bronze_rows = bronze.count()
        spark.sql(VALID_SQL).filter(
            sf.col("source").isin(list(SOURCES))
        ).createOrReplaceTempView("valid")
        grouped = spark.sql(GROUP_SQL).cache()
        grouped.createOrReplaceTempView("grouped")
        audit = CleaningAudit.model_validate_json(
            spark.sql(AUDIT_SQL, args={"bronze_rows": bronze_rows})
            .toJSON()
            .collect()[0]
        )
        spark.sql(SILVER_SQL).repartition(settings.partitions).write.parquet(
            paths.silver, mode="errorifexists"
        )
        _ = grouped.unpersist()
        silver = spark.read.parquet(paths.silver)
        silver.withColumn("bucket", sf.expr(BUCKET)).createOrReplaceTempView("silver")
        silver.select("source", "label", "length").write.json(
            paths.mr_input, mode="errorifexists"
        )
        spark.sql(GOLD_SQL).coalesce(1).write.option("sep", "\t").csv(
            paths.gold, mode="errorifexists"
        )
        # Gold is read from storage so verification covers HDFS output.
        gold = (
            spark.read.schema("key STRING, n LONG").option("sep", "\t").csv(paths.gold)
        )
        aggregate_rows = (
            AggregateRow.model_validate_json(raw)
            for raw in gold.toJSON().toLocalIterator()
        )
        aggregates = {row.key: row.n for row in aggregate_rows}
        silver_keys = (
            SilverKey.model_validate_json(raw)
            for raw in silver.select("digest", "label", "source")
            .toJSON()
            .toLocalIterator()
        )
        keys = sorted(f"{row.digest}|{row.label}|{row.source}" for row in silver_keys)
        receipt = SparkReceipt(
            spark_version=spark.version,
            default_fs=fs,
            paths=paths,
            bronze_rows=bronze_rows,
            silver_rows=aggregates.get("total", 0),
            output_keys_sha256=hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            aggregates=dict(sorted(aggregates.items())),
            startup_seconds=startup,
            job_seconds=time.perf_counter() - job_started,
            settings=settings,
            audit=audit,
        )
        _ = (report / "spark.json").write_text(
            receipt.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        typer.echo(receipt.model_dump_json())
    finally:
        spark.stop()


if __name__ == "__main__":
    typer.run(main)
