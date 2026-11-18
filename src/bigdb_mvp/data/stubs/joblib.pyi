from pathlib import Path
from typing import Literal, overload

from bigdb_mvp.baseline import BaselineModel
from bigdb_mvp.experiments.classification_model import Model
from bigdb_mvp.khaters_score import ScoreModel

def dump(value: BaselineModel | Model | ScoreModel, filename: Path) -> list[str]: ...
@overload
def load(filename: Path) -> BaselineModel: ...
@overload
def load(filename: Path, *, mmap_mode: Literal["r"]) -> BaselineModel | Model: ...
@overload
def load(filename: Path, *, mmap_mode: Literal["c"]) -> BaselineModel | Model | ScoreModel: ...
