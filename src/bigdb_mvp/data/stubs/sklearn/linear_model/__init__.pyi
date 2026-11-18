from collections.abc import Sequence

from sklearn.feature_extraction.text import FeatureMatrix, FloatVector

class ProbabilityMatrix:
    def __getitem__(self, index: tuple[slice, int]) -> Sequence[float]: ...

class LogisticRegression:
    coef_: tuple[FloatVector, ...]
    intercept_: FloatVector
    classes_: Sequence[int]
    def __init__(
        self, *, class_weight: str | None, max_iter: int, random_state: int, solver: str
    ) -> None: ...
    def fit(self, x: FeatureMatrix, y: Sequence[int]) -> LogisticRegression: ...
    def predict_proba(self, x: FeatureMatrix) -> ProbabilityMatrix: ...
