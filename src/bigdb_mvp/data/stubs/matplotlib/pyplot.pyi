from collections.abc import Iterable
from typing import Literal, overload

from matplotlib.axes import Axes
from matplotlib.figure import Figure

@overload
def subplots(
    nrows: Literal[1] = 1,
    ncols: Literal[1] = 1,
    *,
    figsize: tuple[float, float],
    layout: str,
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    nrows: Literal[1], ncols: Literal[2], *, figsize: tuple[float, float], layout: str
) -> tuple[Figure, Iterable[Axes]]: ...
def close(fig: Figure) -> None: ...
