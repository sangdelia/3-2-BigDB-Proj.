from collections.abc import Iterator, Sequence
from typing import overload

class Token:
    form: str
    tag: str

class Kiwi:
    def __init__(self, *, num_workers: int) -> None: ...
    @overload
    def tokenize(self, text: str) -> list[Token]: ...
    @overload
    def tokenize(self, text: Sequence[str]) -> Iterator[list[Token]]: ...
