from collections.abc import Iterator, Sequence
from os import PathLike
from types import TracebackType
from typing import Self

from pyarrow import RecordBatch, Schema, Table

def write_table(table: Table, where: str | PathLike[str]) -> None: ...

class ParquetFile:
    def __init__(self, source: str | PathLike[str]) -> None: ...
    @property
    def schema_arrow(self) -> Schema: ...
    def iter_batches(
        self,
        batch_size: int = ...,
        columns: Sequence[str] | None = ...,
    ) -> Iterator[RecordBatch]: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
