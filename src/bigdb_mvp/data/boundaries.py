"""Validated models for untrusted dataset rows."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class NiklInput(BaseModel):
    """NIKL train row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    doc_id: str | None = None
    target_form: str
    label: str


class ExternalMetadata(BaseModel):
    """Metadata shared by cached Korean external rows."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    split: str
    original_label: str | tuple[int, ...] | None = None
    title: str | None = None


class ExternalInput(BaseModel):
    """Cached Korean external row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    target_form: str
    label: str
    metadata: ExternalMetadata | None = None


class CivilInput(BaseModel):
    """Civil Comments Parquet row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    text: str
    toxicity: float | None = Field(default=None, allow_inf_nan=True)
