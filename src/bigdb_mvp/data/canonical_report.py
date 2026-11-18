"""Safe physical and aggregate facts used by canonical publication and audit."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from bigdb_mvp.data.manifest import CanonicalCount, CanonicalIntersection

if TYPE_CHECKING:
    from pathlib import Path

    from bigdb_mvp.data.models import CanonicalRecord


@dataclass(frozen=True, slots=True)
class ParquetFacts:
    """Physical Parquet metadata that contains no row values."""

    schema_fingerprint: str
    file_size_bytes: int
    row_group_rows: tuple[int, ...]
    compression: tuple[str, ...]

    @property
    def row_group_count(self) -> int:
        """Return the number of physical row groups."""
        return len(self.row_group_rows)


def parquet_facts(path: Path) -> ParquetFacts:
    """Read safe schema, size, compression, and row-group metadata."""
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    schema_text = "|".join(f"{name}:{schema.field(name).type}" for name in schema.names)
    metadata = parquet.metadata
    row_group_rows = tuple(
        metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
    )
    compression = tuple(
        sorted(
            {
                metadata.row_group(group).column(column).compression.upper()
                for group in range(metadata.num_row_groups)
                for column in range(metadata.row_group(group).num_columns)
            }
        )
    )
    return ParquetFacts(
        schema_fingerprint=hashlib.sha256(schema_text.encode()).hexdigest(),
        file_size_bytes=path.stat().st_size,
        row_group_rows=row_group_rows,
        compression=compression,
    )


def counts(
    records: tuple[CanonicalRecord, ...],
) -> dict[str, tuple[CanonicalCount, ...]]:
    """Count canonical records by each required aggregate dimension."""
    dimensions = {
        "source": Counter(row.source.value for row in records),
        "language": Counter(row.language.value for row in records),
        "label": Counter(row.label.value for row in records if row.label is not None),
        "split": Counter(row.source_split for row in records),
        "role": Counter(row.role.value for row in records),
    }
    return {
        name: tuple(
            CanonicalCount(value=value, count=counter[value])
            for value in sorted(counter)
        )
        for name, counter in dimensions.items()
    }


def group_intersections(
    records: tuple[CanonicalRecord, ...], field: str
) -> tuple[CanonicalIntersection, ...]:
    """Count groups shared by distinct role or source-split values."""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        value = row.role.value if field == "role" else row.source_split
        groups[row.group_id].add(value)
    pairs = Counter(
        (left, right)
        for values in groups.values()
        for left in sorted(values)
        for right in sorted(values)
        if left < right
    )
    return tuple(
        CanonicalIntersection(left=left, right=right, count=count)
        for (left, right), count in sorted(pairs.items())
    )
