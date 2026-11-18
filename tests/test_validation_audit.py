"""Guard coverage and error-rate denominators against review/rejection leakage."""

import pytest
from pydantic import ValidationError

from bigdb_mvp.experiments.classification_metrics import metrics
from bigdb_mvp.experiments.validation_audit import AuditReport, accepts
from bigdb_mvp.experiments.validation_profile import Observation, profile


def test_review_and_rejected_rows_do_not_count_as_binary_predictions() -> None:
    # Given: one decided clean FP, one review harmful and one rejected clean.
    rows = (
        Observation(
            source="a",
            label="clean",
            length=2,
            score=0.5,
            feature_count=1,
            accepted_input=True,
        ),
        Observation(
            source="a",
            label="harmful",
            length=1,
            score=0.9,
            feature_count=0,
            accepted_input=True,
        ),
        Observation(
            source="a",
            label="clean",
            length=1001,
            score=0.1,
            feature_count=2,
            accepted_input=False,
        ),
    )
    # When: the exact service threshold is applied.
    overall = profile(rows, 0.5)[0]
    # Then: review/rejection do not inflate recall or binary accuracy.
    assert (
        overall.total,
        overall.rejected_input,
        overall.review,
        overall.decided,
        overall.coverage,
    ) == (3, 1, 1, 1, 0.5)
    assert overall.review_harmful == 1
    assert overall.decision_stats is not None
    assert (overall.decision_stats.fp, overall.decision_stats.tn) == (1, 0)
    assert overall.decision_stats.harmful_recall is None


def test_groupings_preserve_population_and_empty_slices_have_no_metrics() -> None:
    # Given: two different sources and lengths straddling a bucket boundary.
    rows = (
        Observation(
            source="a",
            label="clean",
            length=10,
            score=0.1,
            feature_count=2,
            accepted_input=True,
        ),
        Observation(
            source="b",
            label="harmful",
            length=11,
            score=0.9,
            feature_count=2,
            accepted_input=True,
        ),
    )
    # When: the same population is partitioned by source and character length.
    slices = profile(rows, 0.5)
    # Then: neither grouping loses or duplicates a row; absent metrics stay null.
    for dimension in ("source", "length"):
        assert sum(s.total for s in slices if s.key.dimension == dimension) == 2
    lengths = {s.key.name: s for s in slices if s.key.dimension == "length"}
    assert (lengths["1-10"].total, lengths["11-20"].total) == (1, 1)
    assert lengths["101-200"].decision_stats is None
    assert lengths["101-200"].coverage is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("한", True),
        ("한" * 1000, True),
        ("한" * 1001, False),
        ("hello", False),
        (" ", False),
    ],
)
def test_input_acceptance_matches_the_actual_api(text: str, *, expected: bool) -> None:
    # Given: API boundary cases, including exact maximum length.
    # When: the audit checks acceptance.
    accepted = accepts(text)
    # Then: the actual request contract decides inclusion.
    assert accepted is expected


def test_report_contract_cannot_claim_test_evaluation() -> None:
    # Given: a result document with a forbidden test population.
    invalid = {
        "generated_at": "2026-10-29T12:00:00Z",
        "elapsed_seconds": 1,
        "dataset_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "threshold": 0.5,
        "raw_binary_metrics": metrics((0, 1), (0.1, 0.9), 0.5).model_dump(),
        "slices": (),
        "split": "test",
    }
    # When / Then: parsing rejects a report that claims the wrong partition.
    with pytest.raises(ValidationError):
        _ = AuditReport.model_validate(invalid)
