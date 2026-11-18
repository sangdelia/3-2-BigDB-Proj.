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


class UnSmileInput(BaseModel):
    """UnSmile row boundary for the documented binary projection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    text: str = Field(alias="문장")
    women_family: int = Field(alias="여성/가족", strict=True)
    men: int = Field(alias="남성", strict=True)
    sexual_minorities: int = Field(alias="성소수자", strict=True)
    race_nationality: int = Field(alias="인종/국적", strict=True)
    age: int = Field(alias="연령", strict=True)
    region: int = Field(alias="지역", strict=True)
    religion: int = Field(alias="종교", strict=True)
    other_hate: int = Field(alias="기타 혐오", strict=True)
    abuse: int = Field(alias="악플/욕설", strict=True)
    clean: int = Field(strict=True)

    @property
    def harmful_flags(self) -> tuple[int, ...]:
        """Return only the nine labels included in the harmful projection."""
        return (
            self.women_family,
            self.men,
            self.sexual_minorities,
            self.race_nationality,
            self.age,
            self.region,
            self.religion,
            self.other_hate,
            self.abuse,
        )


class KoldInput(BaseModel):
    """KOLD row boundary for article-grouped binary classification."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    guid: str
    source: str
    title: str
    comment: str
    off: bool = Field(alias="OFF", strict=True)


class KmhasInput(BaseModel):
    """K-MHaS TSV row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=True
    )

    document: str
    label: str


class BeepInput(BaseModel):
    """BEEP labeled TSV row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=True
    )

    comments: str
    hate: str


class KhatersInput(BaseModel):
    """K-HATERS JSONL row boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, strict=True
    )

    text: str
    label: str
