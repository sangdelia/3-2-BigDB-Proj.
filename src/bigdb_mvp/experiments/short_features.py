"""Compare short-character coverage on the existing frozen Korean splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import typer
from threadpoolctl import threadpool_limits

from bigdb_mvp.experiments.classification import run_method
from bigdb_mvp.experiments.classification_contracts import (
    ClassificationReport,
    MethodResult,
    load_rows,
)

ARTIFACTS: Final = Path("artifacts/korean_short_v1")
REPORTS: Final = Path("reports/korean_short_v1")
VARIANTS: Final = (
    ("char_2_5_60000", (2, 5), 60000),
    ("char_3_5_200000", (3, 5), 200000),
    ("char_2_5_200000", (2, 5), 200000),
)


def main() -> None:
    """Keep training data fixed and select by validation macro F1 only."""
    rows, digest = load_rows(Path("data/curated/korean_v2"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    configuration = {
        "dataset_sha256": digest,
        "variants": VARIANTS,
        "selection": "validation macro_f1; tie: fewer features",
        "seed": 20261010,
        "test_note": (
            "Existing test has been inspected in prior diagnosis; "
            "these are follow-up comparisons, not a new untouched benchmark. "
            "Test scores and authored probes do not select the model."
        ),
    }
    _ = (ARTIFACTS / "manifest.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    results: list[MethodResult] = []
    with threadpool_limits(limits=2):
        for name, ngrams, limit in VARIANTS:
            typer.echo(f"Training {name}")
            result = run_method(name, rows, ARTIFACTS, limit, ngram_range=ngrams)
            results.append(result)
            _ = (REPORTS / f"{name}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            typer.echo(result.model_dump_json())
    original = ClassificationReport.model_validate_json(
        Path("reports/korean_v2/metrics.json").read_bytes()
    )
    baseline = next(
        result for result in original.results if result.method == "char_tfidf"
    )
    selected = max((baseline, *results), key=lambda result: result.validation.macro_f1)
    _ = (REPORTS / "metrics.json").write_text(
        json.dumps(
            {
                "configuration": configuration,
                "selected_method": selected.method,
                "results": [result.model_dump() for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Validation selection: {selected.method}")


if __name__ == "__main__":
    main()
