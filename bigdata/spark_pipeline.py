"""Spark stage of the HDFS-integrated pipeline: read Bronze from HDFS, write Silver/Gold back to HDFS.

Bronze  hdfs://…/bronze/raw_input.parquet       accepted source rows (text, label, source)
Silver  hdfs://…/silver/corpus/                 whitespace-normalized, conflict-free, exact-dedup rows (+ length)
        hdfs://…/silver/mr_input/               metadata only (source, label, length) for MapReduce; no text
Gold    hdfs://…/gold/spark_aggregates/         27-key counts computed by Spark
Run: python bigdata/spark_pipeline.py --fs hdfs://127.0.0.1:19310 --ns /bigdb/pipeline --report reports/bigdata_pipeline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession

SILVER_SQL = r"""
WITH normalized AS (
  SELECT trim(regexp_replace(text, '[ \\t\\r\\n\\f\\x0B]+', ' ')) AS text, label, source FROM bronze
), valid AS (
  SELECT text, sha2(text, 256) AS digest, label, source FROM normalized
  WHERE text <> '' AND label IN ('clean', 'harmful')
)
SELECT digest, min(text) AS text, min(label) AS label, min(source) AS source, char_length(min(text)) AS length
FROM valid GROUP BY digest HAVING count(DISTINCT label) = 1
"""
BUCKET = """CASE WHEN length <= 10 THEN '10' WHEN length <= 20 THEN '20' WHEN length <= 50 THEN '50'
  WHEN length <= 100 THEN '100' WHEN length <= 200 THEN '200' WHEN length <= 500 THEN '500'
  WHEN length <= 1000 THEN '1000' ELSE 'overflow' END"""
GOLD_SQL = f"""
SELECT 'total' AS key, count(*) AS n FROM silver
UNION ALL SELECT concat('source:', source), count(*) FROM silver GROUP BY source
UNION ALL SELECT concat('label:', label), count(*) FROM silver GROUP BY label
UNION ALL SELECT concat('source_label:', source, ':', label), count(*) FROM silver GROUP BY source, label
UNION ALL SELECT concat('length:', {BUCKET}), count(*) FROM silver GROUP BY {BUCKET}
"""


def session(fs: str) -> SparkSession:
    """Single-host Spark kept to few threads; HDFS is the default filesystem."""
    jvm = "-XX:ActiveProcessorCount=2 -XX:+UseSerialGC -XX:CICompilerCount=2 -Xss512k"
    return (
        SparkSession.builder.master("local[2]")
        .appName("BigDB-HDFS-Spark-pipeline")
        .config("spark.hadoop.fs.defaultFS", fs)
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.extraJavaOptions", jvm)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "2")
        .config("spark.rpc.netty.dispatcher.numThreads", "2")
        .config("spark.rpc.io.serverThreads", "1")
        .config("spark.shuffle.io.serverThreads", "1")
        .config("spark.shuffle.io.clientThreads", "1")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fs", default="hdfs://127.0.0.1:19310")
    p.add_argument("--ns", default="/bigdb/pipeline")
    p.add_argument("--report", type=Path, default=Path("reports/bigdata_pipeline"))
    a = p.parse_args()
    a.report.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spark = session(a.fs)
    startup = time.perf_counter() - started
    try:
        t = time.perf_counter()
        bronze = spark.read.parquet(f"{a.ns}/bronze/raw_input.parquet").select("text", "label", "source")
        bronze.createOrReplaceTempView("bronze")
        bronze_rows = bronze.count()
        silver = spark.sql(SILVER_SQL)
        silver.write.mode("overwrite").parquet(f"{a.ns}/silver/corpus")
        silver = spark.read.parquet(f"{a.ns}/silver/corpus")  # read back from HDFS
        silver.createOrReplaceTempView("silver")
        silver.select("source", "label", "length").write.mode("overwrite").json(f"{a.ns}/silver/mr_input")
        gold = spark.sql(GOLD_SQL)
        gold.coalesce(1).write.mode("overwrite").option("sep", "\t").csv(f"{a.ns}/gold/spark_aggregates")
        aggregates = {r["key"]: int(r["n"]) for r in gold.collect()}
        keys = sorted(f"{r['digest']}|{r['label']}|{r['source']}" for r in silver.select("digest", "label", "source").toLocalIterator())
        seconds = time.perf_counter() - t
        payload = {
            "status": "complete",
            "spark_version": spark.version,
            "hadoop_client_version": spark.sparkContext._jvm.org.apache.hadoop.util.VersionInfo.getVersion(),
            "default_fs": a.fs,
            "paths": {
                "bronze": f"{a.ns}/bronze/raw_input.parquet",
                "silver": f"{a.ns}/silver/corpus",
                "mr_input": f"{a.ns}/silver/mr_input",
                "gold": f"{a.ns}/gold/spark_aggregates",
            },
            "bronze_rows": bronze_rows,
            "silver_rows": aggregates.get("total"),
            "output_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            "aggregates": dict(sorted(aggregates.items())),
            "startup_seconds": startup,
            "job_seconds": seconds,
            "mode": "single host, Spark local[2] reading/writing HDFS; not a multi-node cluster",
        }
        (a.report / "spark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("bronze_rows", "silver_rows", "output_keys_sha256", "job_seconds")}))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
