from os import PathLike
from types import TracebackType
from typing import BinaryIO, Self

from pyarrow import Schema, Table

class RecordBatchReader:
    def read_all(self) -> Table: ...

class RecordBatchStreamWriter:
    def write_table(self, table: Table) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

def open_stream(source: str | PathLike[str] | BinaryIO) -> RecordBatchReader: ...
def new_stream(sink: BinaryIO, schema: Schema) -> RecordBatchStreamWriter: ...
