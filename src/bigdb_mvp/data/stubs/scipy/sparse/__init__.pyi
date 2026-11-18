from collections.abc import Sequence
from numpy import float64
from numpy.typing import NDArray
from sklearn.feature_extraction.text import FeatureMatrix
class csr_matrix(FeatureMatrix):
    def __init__(self, arg1: NDArray[float64]) -> None: ...
def hstack(blocks: Sequence[FeatureMatrix], format: str) -> FeatureMatrix: ...
