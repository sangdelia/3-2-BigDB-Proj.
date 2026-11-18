"""Offline ingestion from the six pinned original assets, never staging."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING, Final, assert_never

from pyarrow import ipc
from pydantic import TypeAdapter

from bigdb_mvp.data.arrow_sources import ARROW_SOURCES
from bigdb_mvp.data.local_sources import LOCAL_SOURCES
from bigdb_mvp.data.models import SourceName
from bigdb_mvp.data.normalize import KHATERS_LABELS
from bigdb_mvp.data.remote_registry import REMOTE_SOURCES
from bigdb_mvp.experiments.dataset_types import (
    Annotation,
    DatasetError,
    ExperimentRow,
    ParsedRows,
    RawRow,
    digest_text,
    normalize_text,
)

if TYPE_CHECKING:
    from pathlib import Path

    from bigdb_mvp.data.remote_registry import RemoteSourceSpec

UNSMILE_TARGETS: Final = (
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
    "기타 혐오",
    "개인지칭",
)


def verify_asset(path: Path, expected_sha256: str) -> None:
    """Reject missing or changed assets before parsing any source rows."""
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise DatasetError("asset_missing:" + path.name) from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DatasetError("checksum_mismatch:" + path.name)


def _kmhas_labels(raw: RawRow) -> tuple[str, list[str]]:
    """Reject clean/harmful co-labeling and preserve multilabel IDs."""
    labels = {part.strip() for part in raw.string("label").split(",")}
    if labels == {"8"}:
        label = "clean"
    elif labels and labels <= {str(i) for i in range(8)}:
        label = "harmful"
    else:
        raise DatasetError(reason="kmhas_conflicting_or_unknown_labels")
    targets = sorted(labels - {"8"})
    return label, targets


def _unsmile_labels(raw: RawRow) -> tuple[str, list[str], list[str]]:
    """Retain all active target flags and the abuse expression flag."""
    harmful = any(
        raw.root.get(key) == 1 for key in (*UNSMILE_TARGETS[:-1], "악플/욕설")
    )
    clean = raw.root.get("clean") == 1
    if clean == harmful:
        raise DatasetError(reason="unsmile_conflicting_or_missing_labels")
    label = "clean" if clean else "harmful"
    targets = [key for key in UNSMILE_TARGETS if raw.root.get(key) == 1]
    expressions = ["악플/욕설"] if raw.root.get("악플/욕설") == 1 else []
    return label, targets, expressions


def _khaters_targets(raw: RawRow) -> list[str]:
    """Preserve every original target entry without category inference."""
    values = raw.root.get("target_label", [])
    if not isinstance(values, list):
        raise DatasetError(reason="invalid_target_labels")
    return [str(value) for value in values]


def _kold_label(raw: RawRow) -> str:
    """Reject absent or non-boolean offensiveness before binary projection."""
    value = raw.root.get("OFF")
    if not isinstance(value, bool):
        raise DatasetError(reason="kold_invalid_off")
    return "harmful" if value else "clean"


def parse_row(source: SourceName, raw: RawRow, row_id: int) -> ExperimentRow:
    """Project the binary label while retaining all original annotations."""
    targets: list[str] = []
    expressions: list[str] = []
    context = ""
    context_kind = "unavailable"
    document = ""
    match source:
        case SourceName.NIKL:
            text = raw.string("target_form")
            label = {"appropriate": "clean", "inappropriate": "harmful"}.get(
                raw.string("label")
            )
            document = raw.string("doc_id")
            context = raw.string("context")
            context_kind = "available_context" if context.strip() else "unavailable"
        case SourceName.BEEP:
            text = raw.string("comments")
            label = {"none": "clean", "offensive": "harmful", "hate": "harmful"}.get(
                raw.string("hate")
            )
            targets = ["bias:" + raw.string("bias")]
            expressions = [raw.string("hate")]
            context = raw.string("title")
            context_kind = "article_title" if context.strip() else "unavailable"
        case SourceName.KMHAS:
            text = raw.string("document")
            label, targets = _kmhas_labels(raw)
        case SourceName.KHATERS:
            text = raw.string("text")
            mapped = KHATERS_LABELS.get(raw.string("label").lower())
            label = mapped.value if mapped is not None else None
            targets = _khaters_targets(raw)
            expressions = [raw.string("label")]
        case SourceName.UNSMILE:
            text = raw.string("문장")
            label, targets, expressions = _unsmile_labels(raw)
        case SourceName.KOLD:
            text = raw.string("comment")
            label = _kold_label(raw)
            targets = [
                key + ":" + raw.string(key) for key in ("TGT", "GRP") if raw.string(key)
            ]
            context = raw.string("title")
            context_kind = "article_title" if context.strip() else "unavailable"
        case SourceName.CIVIL_COMMENTS:
            raise DatasetError(reason="english_source_forbidden")
        case unreachable:
            assert_never(unreachable)
    if label not in {"clean", "harmful"}:
        raise DatasetError(reason="unknown_binary_label")
    normalized = normalize_text(text)
    if not normalized:
        raise DatasetError(reason="empty_text")
    annotation = Annotation(source=source.value, row_id=row_id, raw=raw)
    return ExperimentRow.model_validate(
        {
            "text": text,
            "label": label,
            "source": source.value,
            "group_id": digest_text(source.value + ":doc:" + document)
            if document
            else "",
            "text_sha256": digest_text(normalized),
            "source_labels_json": "[" + annotation.model_dump_json() + "]",
            "target_labels": [source.value + ":" + value for value in targets],
            "expression_labels": [source.value + ":" + value for value in expressions],
            "context_text": context,
            "context_kind": context_kind,
        }
    )


def _remote_rows(remote: RemoteSourceSpec) -> list[RawRow]:
    """Read verified TSV/JSONL and paired article titles."""
    path = remote.assets[0].target
    with path.open(encoding="utf-8", newline="") as handle:
        if path.suffix == ".jsonl":
            rows = [RawRow.model_validate_json(line) for line in handle if line.strip()]
        else:
            rows = [
                RawRow.model_validate(row)
                for row in csv.DictReader(handle, delimiter="\t")
            ]
    if len(remote.assets) > 1:
        titles = remote.assets[1].target.read_text(encoding="utf-8").splitlines()
        if len(titles) != len(rows):
            raise DatasetError(reason="title_row_mismatch")
        rows = [
            RawRow({**row.root, "title": title})
            for row, title in zip(rows, titles, strict=True)
        ]
    return rows


def load_sources() -> ParsedRows:
    """Read only original checksum-verified files and record every exclusion."""
    batches: list[tuple[SourceName, list[RawRow], int | None, RawRow]] = []
    for spec in LOCAL_SOURCES:
        verify_asset(spec.upstream, spec.expected_sha256)

        rows = TypeAdapter(list[RawRow]).validate_json(spec.upstream.read_bytes())
        batches.append(
            (
                spec.source,
                rows,
                spec.expected_rows,
                RawRow(
                    {
                        "source": spec.source.value,
                        "sha256": spec.expected_sha256,
                        "license": spec.license,
                    }
                ),
            )
        )
    for arrow in ARROW_SOURCES:
        verify_asset(arrow.upstream, arrow.expected_sha256)
        with arrow.upstream.open("rb") as handle:
            table = ipc.open_stream(handle).read_all()
        if tuple(table.schema.names) != arrow.expected_schema:
            raise DatasetError("schema_mismatch:" + arrow.source.value)
        rows = [
            RawRow.model_validate_json(json.dumps(row, default=str, ensure_ascii=False))
            for row in table.to_pylist()
        ]
        batches.append(
            (
                arrow.source,
                rows,
                arrow.expected_rows,
                RawRow(
                    {
                        "source": arrow.source.value,
                        "sha256": arrow.expected_sha256,
                        "license": arrow.license,
                    }
                ),
            )
        )
    for remote in REMOTE_SOURCES:
        for asset in remote.assets:
            verify_asset(asset.target, asset.expected_sha256)
        rows = _remote_rows(remote)
        batches.append(
            (
                remote.source,
                rows,
                remote.expected_rows,
                RawRow(
                    {
                        "source": remote.source.value,
                        "sha256": [asset.expected_sha256 for asset in remote.assets],
                        "license": remote.license,
                    }
                ),
            )
        )
    result: list[ExperimentRow] = []
    receipts: list[RawRow] = []
    exclusions: Counter[str] = Counter()
    for source, rows, expected, receipt in batches:
        if len(rows) != expected:
            raise DatasetError("row_count_mismatch:" + source.value)
        accepted = 0
        for index, row in enumerate(rows):
            try:
                result.append(parse_row(source, row, index))
                accepted += 1
            except DatasetError as error:
                exclusions[source.value + ":" + error.reason] += 1
        receipts.append(
            RawRow({**receipt.root, "raw_rows": len(rows), "accepted_rows": accepted})
        )
    return ParsedRows(tuple(result), tuple(receipts), tuple(sorted(exclusions.items())))
