"""Build the independent Korean6 experiment corpus from pinned local originals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

import polars as pl
import typer
from pydantic import TypeAdapter

from bigdb_mvp.experiments.dataset_sources import load_sources
from bigdb_mvp.experiments.dataset_types import (
    Annotation,
    DatasetError,
    ExperimentRow,
    digest_text,
    normalize_text,
)

DEFAULT_ROOT: Final = Path("data/curated/korean_v2")
MIN_TITLE: Final = 8
TRAIN_LIMIT: Final = 0.8
VALIDATION_LIMIT: Final = 0.9


def group_rows(
    rows: tuple[ExperimentRow, ...], seed: int = 42
) -> tuple[ExperimentRow, ...]:
    """Union document, meaningful title and normalized-text links before dedup."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        keys = ["text:" + row.text_sha256]
        if row.group_id:
            keys.append("doc:" + row.group_id)
        title = normalize_text(row.context_text)
        if (
            row.context_kind == "article_title"
            and len(title) >= MIN_TITLE
            and any(char.isalnum() for char in title)
        ):
            keys.append("title:" + digest_text(title))
        for key in keys:
            if key in seen:
                left, right = find(index), find(seen[key])
                parents[max(left, right)] = min(left, right)
            else:
                seen[key] = index
    members: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row.text_sha256)
    ids = {
        root: digest_text("\n".join(sorted(set(hashes))))
        for root, hashes in members.items()
    }
    result: list[ExperimentRow] = []
    for index, row in enumerate(rows):
        group = ids[find(index)]
        bucket = int(digest_text(f"{seed}:{group}")[:16], 16) / 18446744073709551616.0
        split = (
            "train"
            if bucket < TRAIN_LIMIT
            else ("validation" if bucket < VALIDATION_LIMIT else "test")
        )
        result.append(row.model_copy(update={"group_id": group, "split": split}))
    return tuple(result)


def deduplicate(
    rows: tuple[ExperimentRow, ...],
) -> tuple[tuple[ExperimentRow, ...], Counter[str]]:
    """Drop entire conflicting text sets and merge all consistent annotations."""
    groups: dict[str, list[ExperimentRow]] = defaultdict(list)
    for row in rows:
        groups[row.text_sha256].append(row)
    kept: list[ExperimentRow] = []
    exclusions: Counter[str] = Counter()
    adapter = TypeAdapter(list[Annotation])
    for members in groups.values():
        if len({row.label for row in members}) > 1:
            exclusions["conflicting_binary_rows"] += len(members)
            exclusions["conflicting_binary_sets"] += 1
            continue
        annotations = [
            annotation
            for row in members
            for annotation in adapter.validate_json(row.source_labels_json)
        ]
        selected = max(members, key=lambda row: bool(row.context_text.strip()))
        kept.append(
            selected.model_copy(
                update={
                    "source_labels_json": adapter.dump_json(annotations).decode(),
                    "target_labels": sorted(
                        {target for row in members for target in row.target_labels}
                    ),
                    "expression_labels": sorted(
                        {label for row in members for label in row.expression_labels}
                    ),
                }
            )
        )
        exclusions["consistent_duplicate_rows"] += len(members) - 1
    return tuple(kept), exclusions


def validate_splits(rows: tuple[ExperimentRow, ...]) -> None:
    """Reject cross-split normalized text and transitive group overlap."""
    for attribute in ("group_id", "text_sha256"):
        seen: dict[str, str] = {}
        for row in rows:
            key = row.group_id if attribute == "group_id" else row.text_sha256
            if key in seen and seen[key] != row.split:
                raise DatasetError("split_overlap:" + attribute)
            seen[key] = row.split


def build_dataset(output_root: Path = DEFAULT_ROOT) -> Path:
    """Write independent corpus, raw benchmark input and privacy-safe manifest."""
    parsed = load_sources()
    grouped = group_rows(parsed.rows)
    rows, exclusions = deduplicate(grouped)
    validate_splits(grouped)
    validate_splits(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    for name, records in (("raw_input", grouped), ("corpus", rows)):
        pl.DataFrame([row.model_dump() for row in records]).write_parquet(
            output_root / f"{name}.parquet"
        )
    manifest = {
        "version": "korean6-v2",
        "seed": 42,
        "schema": ExperimentRow.model_json_schema(),
        "schema_sha256": digest_text(
            json.dumps(ExperimentRow.model_json_schema(), sort_keys=True)
        ),
        "dataset_sha256": hashlib.sha256(
            (output_root / "corpus.parquet").read_bytes()
        ).hexdigest(),
        "sources": [receipt.root for receipt in parsed.receipts],
        "raw_rows": sum(
            TypeAdapter(int).validate_python(receipt.root["raw_rows"])
            for receipt in parsed.receipts
        ),
        "parsed_rows": len(grouped),
        "accepted_rows": len(rows),
        "parse_exclusions": dict(parsed.exclusions),
        "dedup_exclusions": dict(exclusions),
        "label_counts": dict(Counter(row.label for row in rows)),
        "source_counts": dict(Counter(row.source for row in rows)),
        "split_counts": dict(Counter(row.split for row in rows)),
        "source_split_counts": dict(
            Counter(f"{row.source}/{row.split}" for row in rows)
        ),
        "normalized_text_split_overlap": 0,
        "group_split_overlap": 0,
        "normalization": (
            "Whitespace collapse only; punctuation, negation and case preserved. "
        ),
        "grouping": (
            "Transitive union before dedup of normalized text, source document "
            "id and shared article titles (>=8 characters with alphanumeric "
            "content). "
        ),
        "targets": (
            "Source-prefixed original labels; no unsupported cross-source "
            "ontology equivalence. BEEP bias is retained as bias, not inferred "
            "target identity. "
        ),
        "deduplication": (
            "Exact normalized duplicates only; no exhaustive near-duplicate "
            "claim. All consistent source annotations retained in "
            "source_labels_json. "
        ),
        "split_method": (
            "SHA256(seed:component_digest) thresholds 0.8/0.9; approximate "
            "80/10/10 by groups. "
        ),
    }
    path = output_root / "manifest.json"
    _ = path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main(output_root: Path = DEFAULT_ROOT) -> None:
    """Build without network access or changes to the legacy canonical pointer."""
    typer.echo(str(build_dataset(output_root)))


if __name__ == "__main__":
    typer.run(main)
