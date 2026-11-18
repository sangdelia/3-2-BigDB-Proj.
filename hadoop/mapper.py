# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.12,<3"]
# ///
# How to run: .venv/bin/python hadoop/mapper.py < local_metrics.jsonl
"""Map browser-safe row metadata to source, label and length frequencies."""

from __future__ import annotations

import sys
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

BOUNDS: Final = (10, 20, 50, 100, 200, 500, 1000)


class Row(BaseModel):
    """No comment, model weight or original annotation enters this job."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    source: Literal["nikl", "unsmile", "kold", "beep", "kmhas", "khaters"]
    label: Literal["clean", "harmful"]
    length: int = Field(ge=1)


def main() -> None:
    """Emit additive counts; Hadoop handles grouping and partitioning."""
    for line in sys.stdin:
        row = Row.model_validate_json(line)
        upper = next((bound for bound in BOUNDS if row.length <= bound), None)
        bucket = str(upper) if upper is not None else "overflow"
        for key in (
            "total",
            f"source:{row.source}",
            f"label:{row.label}",
            f"source_label:{row.source}:{row.label}",
            f"length:{bucket}",
        ):
            _ = sys.stdout.write(f"{key}\t1\n")


if __name__ == "__main__":
    main()
