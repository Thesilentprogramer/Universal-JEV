"""Numerically stable score processing independent of any model backend."""

import math
from collections.abc import Mapping


def softmax(log_scores: Mapping[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not log_scores:
        raise ValueError("at least one score is required")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    scaled = {key: value / temperature for key, value in log_scores.items()}
    maximum = max(scaled.values())
    weights = {key: math.exp(value - maximum) for key, value in scaled.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}
