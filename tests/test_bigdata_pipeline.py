from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal, assert_never

import polars as pl
import pytest
import typer
from bigdata.contracts import CleaningAudit, HdfsPaths, SparkReceipt
from bigdata.verify_pipeline import (
    AggregateFileError,
    main,
    python_reference,
    read_tsv,
    verify,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def stored_pipeline(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw.parquet"
    pl.DataFrame(
        {
            "text": [
                "  정상\t문장 ",
                "정상 문장",
                "충돌",
                "충돌",
                "유해",
                " \t",
                None,
                "제외",
                "제외",
            ],
            "label": [
                "clean",
                "clean",
                "clean",
                "harmful",
                "harmful",
                "clean",
                "clean",
                "unknown",
                "clean",
            ],
            "source": [
                "kmhas",
                "beep",
                "beep",
                "kold",
                "khaters",
                "beep",
                "beep",
                "beep",
                "unknown",
            ],
        }
    ).write_parquet(raw)
    aggregates = {
        "total": 2,
        "source:beep": 1,
        "source:khaters": 1,
        "label:clean": 1,
        "label:harmful": 1,
        "source_label:beep:clean": 1,
        "source_label:khaters:harmful": 1,
        "length:10": 2,
    }
    expected_keys = sorted(
        f"{hashlib.sha256(text.encode()).hexdigest()}|{label}|{source}"
        for text, label, source in [
            ("정상 문장", "clean", "beep"),
            ("유해", "harmful", "khaters"),
        ]
    )
    receipt = SparkReceipt(
        spark_version="3.5.7",
        default_fs="hdfs://127.0.0.1:19310",
        paths=HdfsPaths(
            bronze="/bronze", silver="/silver", mr_input="/mr", gold="/gold"
        ),
        bronze_rows=9,
        silver_rows=2,
        output_keys_sha256=hashlib.sha256(
            "\n".join(expected_keys).encode()
        ).hexdigest(),
        aggregates=aggregates,
        startup_seconds=0,
        job_seconds=0,
        audit=CleaningAudit(
            bronze_rows=9,
            invalid_rows=4,
            conflict_rows=2,
            duplicate_rows=1,
            silver_rows=2,
        ),
    )
    report = tmp_path / "report"
    report.mkdir()
    _ = (report / "spark.json").write_text(receipt.model_dump_json())
    _ = (report / "mapreduce_aggregates.tsv").write_text(
        "".join(f"{key}\t{value}\n" for key, value in aggregates.items())
    )
    _ = (report / "hdfs-bronze.sha256").write_text(
        hashlib.sha256(raw.read_bytes()).hexdigest() + "  -\n"
    )
    return raw, report


def test_reference_accounts_for_global_conflicts_duplicates_and_invalid_rows(
    stored_pipeline: tuple[Path, Path],
) -> None:
    raw, _ = stored_pipeline
    result = python_reference(raw)
    assert result.audit == CleaningAudit(
        bronze_rows=9, invalid_rows=4, conflict_rows=2, duplicate_rows=1, silver_rows=2
    )
    assert result.aggregates == {
        "total": 2,
        "source:beep": 1,
        "source:khaters": 1,
        "label:clean": 1,
        "label:harmful": 1,
        "source_label:beep:clean": 1,
        "source_label:khaters:harmful": 1,
        "length:10": 2,
    }


@pytest.mark.parametrize("damage", ["input", "spark_keys", "mr_counts", "audit"])
def test_verification_entrypoint_fails_on_persisted_stage_corruption(
    stored_pipeline: tuple[Path, Path],
    damage: Literal["input", "spark_keys", "mr_counts", "audit"],
) -> None:
    raw, report = stored_pipeline
    assert verify(raw, report).status == "passed"
    receipt = SparkReceipt.model_validate_json((report / "spark.json").read_text())
    match damage:
        case "input":
            _ = (report / "hdfs-bronze.sha256").write_text("0" * 64)
        case "spark_keys":
            receipt = receipt.model_copy(update={"output_keys_sha256": "0" * 64})
        case "mr_counts":
            _ = (report / "mapreduce_aggregates.tsv").write_text("total\t3\n")
        case "audit":
            receipt = receipt.model_copy(
                update={
                    "audit": CleaningAudit(
                        bronze_rows=9,
                        invalid_rows=3,
                        conflict_rows=2,
                        duplicate_rows=2,
                        silver_rows=2,
                    )
                }
            )
        case unreachable:
            assert_never(unreachable)
    _ = (report / "spark.json").write_text(receipt.model_dump_json())
    with pytest.raises(typer.Exit) as result:
        main(raw, report)
    assert result.value.exit_code == 1
    assert json.loads((report / "verification.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("contents", ["total\t2\ntotal\t3\n", "total\t-1\n", "total\n"])
def test_reducer_output_rejects_duplicate_or_malformed_entries(
    tmp_path: Path,
    contents: str,
) -> None:
    path = tmp_path / "aggregates.tsv"
    _ = path.write_text(contents)
    with pytest.raises(AggregateFileError):
        _ = read_tsv(path)
