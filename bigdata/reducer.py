# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: sort mapper_output.tsv | .venv/bin/python hadoop/reducer.py
"""Sum sorted Hadoop Streaming counts, also usable as a combiner."""

from __future__ import annotations

import sys
from itertools import groupby


def main() -> None:
    """Reduce one key at a time with memory independent of corpus size."""
    pairs = (line.rstrip("\n").split("\t", 1) for line in sys.stdin)
    for key, values in groupby(pairs, key=lambda pair: pair[0]):
        total = sum(int(pair[1]) for pair in values)
        _ = sys.stdout.write(f"{key}\t{total}\n")


if __name__ == "__main__":
    main()
