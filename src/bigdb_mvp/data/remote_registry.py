"""Immutable registry for official Korean train assets."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from bigdb_mvp.data.civil_download import DownloadArea, DownloadSpec
from bigdb_mvp.data.models import CanonicalRole, SourceName

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
DOWNLOAD_ROOT: Final = PROJECT_ROOT / "data/raw/downloads"

KMHAS_REVISION: Final = "ec7a7e775d650b825872f6f538fc717822cdfc1a"
BEEP_REVISION: Final = "f8d05dce2b22007bb149e5139c0060c68ad8f94b"
KHATERS_REVISION: Final = "67b979254f2f5874f3c0f649e0db8ab33d038e0d"


class RemoteSourceSpec(BaseModel):
    """Immutable provenance and assets for one official train source."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    source: SourceName
    revision: str
    license: str
    repository: str
    assets: tuple[DownloadSpec, ...]
    source_split: str = "train"
    role: CanonicalRole = CanonicalRole.RESEARCH_INDEX
    expected_rows: int | None = Field(default=None, ge=0)
    expected_schema: tuple[str, ...] = ()


def _asset(
    *, url: str, filename: str, expected_sha256: str, expected_bytes: int
) -> DownloadSpec:
    return DownloadSpec(
        url=url,
        target=DOWNLOAD_ROOT / filename,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        area=DownloadArea.RAW,
    )


def _github_raw(repository: str, revision: str, asset: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{asset}"


def _huggingface_asset(repository: str, revision: str, asset: str) -> str:
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{asset}"


REMOTE_SOURCES: Final = (
    RemoteSourceSpec(
        source=SourceName.BEEP,
        revision=BEEP_REVISION,
        license="CC-BY-SA-4.0",
        repository="https://github.com/kocohub/korean-hate-speech",
        assets=(
            _asset(
                url=_github_raw(
                    "kocohub/korean-hate-speech",
                    BEEP_REVISION,
                    "labeled/train.tsv",
                ),
                filename=f"beep-{BEEP_REVISION}-train.tsv",
                expected_sha256=(
                    "ebebacdcd023af2c4acc8c0a37695fb6433ac04fc009feff8f222724e303a5a9"
                ),
                expected_bytes=913_546,
            ),
            _asset(
                url=_github_raw(
                    "kocohub/korean-hate-speech",
                    BEEP_REVISION,
                    "news_title/train.news_title.txt",
                ),
                filename=f"beep-{BEEP_REVISION}-train.news_title.txt",
                expected_sha256=(
                    "80fcb349633ddf38062c17363f18e62b3e533640c2a31ba50468963ff5db1492"
                ),
                expected_bytes=742_357,
            ),
        ),
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=7_896,
        expected_schema=("comments", "contain_gender_bias", "bias", "hate"),
    ),
    RemoteSourceSpec(
        source=SourceName.KMHAS,
        revision=KMHAS_REVISION,
        license="CC-BY-SA-4.0",
        repository="https://github.com/adlnlp/K-MHaS",
        assets=(
            _asset(
                url=_github_raw(
                    "adlnlp/K-MHaS",
                    KMHAS_REVISION,
                    "data/kmhas_train.txt",
                ),
                filename=f"kmhas-{KMHAS_REVISION}-train.txt",
                expected_sha256=(
                    "ca9145afc5fd70a2222094daa427e57a08b309cb42dcce7986528b44eb0e7008"
                ),
                expected_bytes=6_845_463,
            ),
        ),
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=78_977,
        expected_schema=("document", "label"),
    ),
    RemoteSourceSpec(
        source=SourceName.KHATERS,
        revision=KHATERS_REVISION,
        license="CC-BY-4.0",
        repository="https://huggingface.co/datasets/humane-lab/K-HATERS",
        assets=(
            _asset(
                url=_huggingface_asset(
                    "humane-lab/K-HATERS",
                    KHATERS_REVISION,
                    "train.jsonl?download=true",
                ),
                filename=f"khaters-{KHATERS_REVISION}-train.jsonl",
                expected_sha256=(
                    "e1eb9a7e7a7693a1d0e03f206a72247b98b703fdda8db8d74f25ddc8679aa437"
                ),
                expected_bytes=47_417_162,
            ),
        ),
        source_split="train",
        role=CanonicalRole.RESEARCH_INDEX,
        expected_rows=172_158,
        expected_schema=("text", "label"),
    ),
)
