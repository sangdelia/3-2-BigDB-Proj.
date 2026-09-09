"""Pure threshold selection and finite binary metric calculations."""

from __future__ import annotations

from dataclasses import dataclass

from bigdb_mvp.baseline_contracts import MetricSummary


@dataclass(frozen=True, slots=True)
class Counts:
    """Binary confusion counts with harmful encoded as one."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def _counts(
    labels: tuple[int, ...], scores: tuple[float, ...], threshold: float
) -> Counts:
    predictions = tuple(int(score >= threshold) for score in scores)
    pairs = tuple(zip(labels, predictions, strict=True))
    return Counts(
        true_positive=sum(
            actual == 1 and predicted == 1 for actual, predicted in pairs
        ),
        false_positive=sum(
            actual == 0 and predicted == 1 for actual, predicted in pairs
        ),
        false_negative=sum(
            actual == 1 and predicted == 0 for actual, predicted in pairs
        ),
        true_negative=sum(
            actual == 0 and predicted == 0 for actual, predicted in pairs
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def _per_class_f1(counts: Counts, positive: bool) -> float:
    if positive:
        precision = _ratio(
            counts.true_positive, counts.true_positive + counts.false_positive
        )
        recall = _ratio(
            counts.true_positive, counts.true_positive + counts.false_negative
        )
    else:
        precision = _ratio(
            counts.true_negative, counts.true_negative + counts.false_negative
        )
        recall = _ratio(
            counts.true_negative, counts.true_negative + counts.false_positive
        )
    return _f1(precision, recall)


def _roc_auc(labels: tuple[int, ...], scores: tuple[float, ...]) -> float:
    positives = tuple(
        score for label, score in zip(labels, scores, strict=True) if label == 1
    )
    negatives = tuple(
        score for label, score in zip(labels, scores, strict=True) if label == 0
    )
    pairs = len(positives) * len(negatives)
    if not pairs:
        return 0.5
    wins = sum(left > right for left in positives for right in negatives)
    ties = sum(left == right for left in positives for right in negatives)
    return (wins + 0.5 * ties) / pairs


def binary_metrics(
    labels: tuple[int, ...], scores: tuple[float, ...], threshold: float
) -> MetricSummary:
    """Compute held-out metrics using one already-frozen policy threshold."""
    counts = _counts(labels, scores, threshold)
    precision = _ratio(
        counts.true_positive, counts.true_positive + counts.false_positive
    )
    recall = _ratio(counts.true_positive, counts.true_positive + counts.false_negative)
    return MetricSummary(
        accuracy=_ratio(counts.true_positive + counts.true_negative, len(labels)),
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        roc_auc=_roc_auc(labels, scores),
        brier=sum(
            (score - label) ** 2 for label, score in zip(labels, scores, strict=True)
        )
        / len(labels),
        sample_count=len(labels),
    )


def select_threshold(labels: tuple[int, ...], scores: tuple[float, ...]) -> float:
    """Select macro-F1, then harmful recall, then lower threshold on calibration."""
    candidates = tuple(round(step / 100.0, 2) for step in range(5, 96))
    return max(
        candidates,
        key=lambda threshold: (
            (
                _per_class_f1(_counts(labels, scores, threshold), positive=True)
                + _per_class_f1(_counts(labels, scores, threshold), positive=False)
            )
            / 2.0,
            binary_metrics(labels, scores, threshold).recall,
            -threshold,
        ),
    )
