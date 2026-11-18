"""Typed experiment rows with lossless source annotation envelopes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, JsonValue, RootModel


class DatasetError(Exception):
    """A stable integrity or label rejection."""

    reason: str

    def __init__(self, reason: str) -> None:
        """Keep reports free of source text."""
        self.reason = reason
        super().__init__(reason)


class RawRow(RootModel[dict[str, JsonValue]]):
    """Preserve every field of an original source row."""

    def string(self, key: str) -> str:
        """Read a string through a strict typed boundary."""
        value = self.root.get(key, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise DatasetError("expected_string:" + key)
        return value


class Annotation(BaseModel):
    """Provenance and all unmodified raw fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    source: str
    source_split: str = "train"
    row_id: int
    raw: RawRow


class ExperimentRow(BaseModel):
    """Corpus contract; labels retain their source-specific meaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    text: str
    label: Literal["clean", "harmful"]
    language: Literal["ko"] = "ko"
    source: str
    source_split: str = "train"
    group_id: str
    text_sha256: str
    split: Literal["train", "validation", "test"] = "train"
    source_labels_json: str
    target_labels: list[str]
    expression_labels: list[str]
    context_text: str = ""
    context_kind: Literal["article_title", "available_context", "unavailable"] = (
        "unavailable"
    )


@dataclass(frozen=True, slots=True)
class ParsedRows:
    """Rows and source verification metadata."""

    rows: tuple[ExperimentRow, ...]
    receipts: tuple[RawRow, ...]
    exclusions: tuple[tuple[str, int], ...]


def normalize_text(text: str) -> str:
    """Collapse whitespace only; retain punctuation, negation and case."""
    return " ".join(text.split())


def digest_text(text: str) -> str:
    """Hash an exact UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
