from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bigdb_mvp.experiments.khaters_ratings import Ratings, load_dataset
from bigdb_mvp.experiments.midterm_contracts import Metrics, common_rows, eligible

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ordinary_rating() -> Ratings:
    return Ratings(
        text="일반 문장",
        gender=0,
        age=0,
        disability=0,
        job=0,
        race=0,
        religion=0,
        politics=0,
        individuals=0,
        others=0,
        insult=0,
        swear=0,
        threat=0,
        obscenity=0,
    )


def test_common_rows_remove_new_normalized_leaks_and_conflicts(
    tmp_path: Path,
    ordinary_rating: Ratings,
) -> None:
    # Given: previously distinct spellings collide across splits or disagree in train.
    raw = (
        ("시.발", "병_신", "병신", "감사합니다", "감사 합니다"),
        ("시발", "검증 문장"),
        ("평가 문장",),
    )
    for split, texts in zip(("train", "val", "test"), raw, strict=True):
        rows = tuple(
            ordinary_rating.model_copy(
                update={"text": text, "swear": int(text == "병신")}
            )
            for text in texts
        )
        _ = (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(row.model_dump_json() for row in rows), encoding="utf-8"
        )
    # When: common-row preparation runs before any model fitting.
    common = common_rows(load_dataset(tmp_path))
    # Then: both sides of the leak and both conflicting train rows are removed.
    assert tuple({row.text for row in rows} for rows in common.splits) == (
        {"감사합니다", "감사 합니다"},
        {"검증 문장"},
        {"평가 문장"},
    )
    assert common.audit.crossing_keys == common.audit.conflicting_keys == 1
    assert common.audit.removed == (3, 1, 0)


@pytest.mark.parametrize(
    ("mae", "f1", "false_positives", "accepted"),
    [
        (19, 0.6, 5, True),
        (20, 0.6, 5, False),
        (19, 0.59, 5, False),
        (19, 0.6, 6, False),
        (21, 0.7, 4, False),
    ],
)
def test_selection_requires_all_validation_conditions(
    mae: float,
    f1: float,
    false_positives: int,
    accepted: bool,
) -> None:
    # Given: fixed baseline and a candidate that can fail each independent criterion.
    baseline = Metrics(
        count=120,
        grade_counts=(100, 10, 10),
        mae_points=20,
        rmse_points=30,
        train_mean_mae_points=40,
        grade_macro_f1=0.6,
        confusion=((95, 3, 2), (5, 3, 2), (2, 3, 5)),
        nll=0.5,
        temperature=1,
    )
    candidate = baseline.model_copy(
        update={
            "mae_points": mae,
            "grade_macro_f1": f1,
            "confusion": (
                (100 - false_positives, false_positives, 0),
                (5, 3, 2),
                (2, 3, 5),
            ),
        }
    )
    # When: only validation evidence is supplied to selection.
    result = eligible(candidate, baseline)
    # Then: an MAE gain alone never overrides the two error-rate safeguards.
    assert result is accepted
