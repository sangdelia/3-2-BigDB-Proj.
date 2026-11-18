"""Regression checks for the sequential side of the shared workload."""

from pathlib import Path

import polars as pl

from bigdb_mvp.experiments.preprocessing_benchmark import sequential_keys


def test_conflicts_duplicates_and_whitespace(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pl.DataFrame(
        {
            "text": [" a\t b ", "a b", "conflict", "conflict", "", "not harmful"],
            "label": ["clean", "clean", "clean", "harmful", "clean", "clean"],
            "source": ["z", "a", "x", "x", "x", "x"],
        }
    ).write_parquet(path)
    keys = sequential_keys(path)
    assert len(keys) == 2
    assert any(key.endswith("|clean|a") for key in keys)
    assert sequential_keys(path) == keys
