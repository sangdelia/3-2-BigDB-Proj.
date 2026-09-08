"""Typed runtime settings for the local MVP commands."""

from pathlib import Path
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

GIBIBYTE: Final = 1024**3
MODEL_REPO_ID: Final = "FacebookAI/xlm-roberta-base"


class DoctorSettings(BaseModel):
    """Validated limits and selected CUDA device for a live doctor probe."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    device: int = Field(default=0, ge=0)
    minimum_disk_free_bytes: int = Field(default=10 * GIBIBYTE, gt=0)
    minimum_free_vram_bytes: int = Field(default=8 * GIBIBYTE, gt=0)
    expected_gpu_name: str = Field(default="RTX A6000", min_length=1)
    model_cache_root: Path | None = None
