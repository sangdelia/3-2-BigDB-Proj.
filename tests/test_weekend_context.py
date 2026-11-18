from __future__ import annotations

import pytest

from bigdb_mvp.experiments.classification_contracts import Metrics
from bigdb_mvp.experiments.weekend_context import eligible


@pytest.mark.parametrize(
    ("f1", "fpr", "recall", "expected"),
    [
        (0.81, 0.19, 0.86, True),
        (0.79, 0.19, 0.86, False),
        (0.81, 0.21, 0.86, False),
        (0.81, 0.19, 0.84, False),
        (0.80, 0.20, 0.85, False),
    ],
)
def test_selection_requires_all_guards(
    f1: float, fpr: float, recall: float, *, expected: bool
) -> None:
    baseline = Metrics(
        count=100,
        accuracy=0.8,
        macro_f1=0.8,
        harmful_precision=0.8,
        harmful_recall=0.85,
        harmful_f1=0.8,
        false_positive_rate=0.2,
        tn=40,
        fp=10,
        fn=8,
        tp=42,
    )
    candidate = baseline.model_copy(
        update={
            "macro_f1": f1,
            "false_positive_rate": fpr,
            "harmful_recall": recall,
        }
    )
    selected = eligible(candidate, baseline)
    assert selected is expected
