"""Deterministic pooled temperature calibration for binary logits."""

from __future__ import annotations

import math
from typing import Final

LOW_LOG_TEMPERATURE: Final = -6.0
HIGH_LOG_TEMPERATURE: Final = 6.0
GOLDEN_RATIO: Final = (math.sqrt(5.0) - 1.0) / 2.0
SEARCH_STEPS: Final = 128


def sigmoid(value: float) -> float:
    """Evaluate the stable logistic function."""
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def calibrated_probabilities(
    logits: tuple[float, ...], temperature: float
) -> tuple[float, ...]:
    """Scale logits by one pooled temperature before sigmoid."""
    return tuple(sigmoid(logit / temperature) for logit in logits)


def binary_nll(
    labels: tuple[int, ...], logits: tuple[float, ...], temperature: float
) -> float:
    """Compute finite mean binary negative log likelihood."""
    probabilities = calibrated_probabilities(logits, temperature)
    epsilon = 1e-12
    return -sum(
        label * math.log(max(probability, epsilon))
        + (1 - label) * math.log(max(1.0 - probability, epsilon))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)


def fit_temperature(labels: tuple[int, ...], logits: tuple[float, ...]) -> float:
    """Minimize pooled NLL with a deterministic bounded golden-section search."""
    left = LOW_LOG_TEMPERATURE
    right = HIGH_LOG_TEMPERATURE
    inner_left = right - GOLDEN_RATIO * (right - left)
    inner_right = left + GOLDEN_RATIO * (right - left)
    for _ in range(SEARCH_STEPS):
        left_loss = binary_nll(labels, logits, math.exp(inner_left))
        right_loss = binary_nll(labels, logits, math.exp(inner_right))
        if left_loss <= right_loss:
            right = inner_right
            inner_right = inner_left
            inner_left = right - GOLDEN_RATIO * (right - left)
        else:
            left = inner_left
            inner_left = inner_right
            inner_right = left + GOLDEN_RATIO * (right - left)
    return math.exp((left + right) / 2.0)
