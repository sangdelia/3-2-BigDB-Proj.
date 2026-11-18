from pathlib import Path
from types import TracebackType
from typing import Self

from matplotlib.figure import Figure

class PdfPages:
    def __init__(self, filename: str | Path) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...
    def savefig(self, figure: Figure) -> None: ...
