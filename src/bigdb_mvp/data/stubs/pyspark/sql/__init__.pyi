from typing import ClassVar

class Row:
    def __getitem__(self, key: str) -> str: ...

class DataFrame:
    def createOrReplaceTempView(self, name: str) -> None: ...  # noqa: N802
    def collect(self) -> list[Row]: ...

class DataFrameReader:
    def parquet(self, path: str) -> DataFrame: ...

class Builder:
    def master(self, master: str) -> Builder: ...
    def appName(self, name: str) -> Builder: ...  # noqa: N802
    def config(self, key: str, value: str) -> Builder: ...
    def getOrCreate(self) -> SparkSession: ...  # noqa: N802

class SparkSession:
    builder: ClassVar[Builder]
    read: DataFrameReader
    version: str
    def sql(self, query: str, /) -> DataFrame: ...
    def stop(self) -> None: ...
