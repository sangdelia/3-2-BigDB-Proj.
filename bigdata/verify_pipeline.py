"""Cross-check the HDFS-integrated pipeline: Python reference vs Spark (HDFS) vs MapReduce (HDFS).

Run: python bigdata/verify_pipeline.py --raw data/curated/korean_v2/raw_input.parquet --report reports/bigdata_pipeline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

SPACE = re.compile(r"[ \t\r\n\f\v]+")
BOUNDS = (10, 20, 50, 100, 200, 500, 1000)


def python_reference(raw: Path) -> tuple[str, Counter[str], int]:
    labels: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    texts: dict[str, str] = {}
    rows = 0
    for r in pl.read_parquet(raw, columns=["text", "label", "source"]).iter_rows(named=True):
        rows += 1
        text = SPACE.sub(" ", r["text"]).strip(" ")
        if not text or r["label"] not in {"clean", "harmful"}:
            continue
        d = hashlib.sha256(text.encode()).hexdigest()
        labels.setdefault(d, set()).add(r["label"])
        sources.setdefault(d, set()).add(r["source"])
        texts[d] = text
    keys, agg = [], Counter()
    for d, ls in labels.items():
        if len(ls) != 1:
            continue
        label, source, n = next(iter(ls)), min(sources[d]), len(texts[d])
        keys.append(f"{d}|{label}|{source}")
        bucket = next((str(b) for b in BOUNDS if n <= b), "overflow")
        for k in ("total", f"source:{source}", f"label:{label}", f"source_label:{source}:{label}", f"length:{bucket}"):
            agg[k] += 1
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(), agg, rows


def read_tsv(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            k, v = line.split("\t")
            out[k] = int(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/curated/korean_v2/raw_input.parquet"))
    p.add_argument("--report", type=Path, default=Path("reports/bigdata_pipeline"))
    a = p.parse_args()
    digest, ref, rows = python_reference(a.raw)
    spark = json.loads((a.report / "spark.json").read_text(encoding="utf-8"))
    mr = read_tsv(a.report / "mapreduce_aggregates.tsv")
    hdfs_bronze = (a.report / "hdfs-bronze.sha256").read_text().split()[0]
    local_bronze = hashlib.sha256(a.raw.read_bytes()).hexdigest()
    ref = dict(ref)
    checks = {
        "bronze_hdfs_sha256_equals_local": hdfs_bronze == local_bronze,
        "bronze_rows_equal": spark["bronze_rows"] == rows,
        "spark_output_keys_equal_python": spark["output_keys_sha256"] == digest,
        "spark_aggregates_equal_python": spark["aggregates"] == ref,
        "mapreduce_aggregates_equal_python": mr == ref,
        "mapreduce_aggregates_equal_spark": mr == spark["aggregates"],
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "bronze_rows": rows,
        "silver_rows": ref.get("total"),
        "aggregate_keys": len(ref),
        "python_output_keys_sha256": digest,
        "python_aggregates": dict(sorted(ref.items())),
        "mismatched_keys": sorted(k for k in set(ref) | set(mr) | set(spark["aggregates"]) if not (ref.get(k) == mr.get(k) == spark["aggregates"].get(k))),
    }
    (a.report / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **checks, "keys": len(ref)}))
    raise SystemExit(0 if payload["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
