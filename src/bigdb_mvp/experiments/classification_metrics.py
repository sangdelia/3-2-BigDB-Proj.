"""Linear-time binary metrics and validation-only threshold selection."""

from __future__ import annotations

from bigdb_mvp.experiments.classification_contracts import Metrics


def metrics(
    labels: tuple[int, ...], scores: tuple[float, ...], threshold: float
) -> Metrics:
    """Compute safe finite aggregates without exposing individual texts."""
    pairs = tuple(zip(labels, (int(s >= threshold) for s in scores), strict=True))
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    positive_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    negative_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    return Metrics(
        count=len(labels),
        accuracy=(tp + tn) / max(len(labels), 1),
        macro_f1=(positive_f1 + negative_f1) / 2,
        harmful_precision=tp / max(tp + fp, 1),
        harmful_recall=tp / max(tp + fn, 1),
        harmful_f1=positive_f1,
        false_positive_rate=fp / max(fp + tn, 1),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def threshold_on_validation(
    labels: tuple[int, ...], scores: tuple[float, ...]
) -> float:
    """Freeze macro F1 optimum with ties nearest 0.5 then lower threshold."""
    return max(
        (i / 100 for i in range(5, 96)),
        key=lambda t: (metrics(labels, scores, t).macro_f1, -abs(t - 0.5), -t),
    )
