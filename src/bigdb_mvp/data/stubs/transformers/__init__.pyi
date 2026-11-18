from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Self

from torch import Tensor

class BatchEncoding(dict[str, Tensor]):
    def to(self, device: str) -> Self: ...

class SequenceClassifierOutput:
    loss: Tensor | None
    logits: Tensor

class PreTrainedTokenizerBase:
    def __call__(
        self,
        text: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: Literal["pt"],
    ) -> BatchEncoding: ...
    def save_pretrained(self, save_directory: Path) -> tuple[str, ...]: ...

class PreTrainedModel:
    def __call__(self, **kwargs: Tensor) -> SequenceClassifierOutput: ...
    def to(self, device: str) -> Self: ...
    def parameters(self) -> Iterator[Tensor]: ...
    def train(self) -> Self: ...
    def eval(self) -> Self: ...
    def requires_grad_(self, requires_grad: bool) -> Self: ...
    def save_pretrained(
        self, save_directory: Path, *, safe_serialization: bool
    ) -> None: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *, revision: str | None = ...
    ) -> PreTrainedTokenizerBase: ...

class AutoModelForSequenceClassification:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str | None = ...,
        num_labels: int | None = ...,
    ) -> PreTrainedModel: ...
