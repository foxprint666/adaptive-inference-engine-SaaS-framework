"""
Small metric helpers for the multi-tenant worker.
"""

from typing import Iterable


def population_stability_index(expected: Iterable[float], actual: Iterable[float]) -> float:
    expected_values = list(expected)
    actual_values = list(actual)
    if not expected_values or not actual_values:
        return 0.0

    expected_mean = sum(expected_values) / len(expected_values)
    actual_mean = sum(actual_values) / len(actual_values)
    denominator = abs(expected_mean) + 1e-9
    return abs(actual_mean - expected_mean) / denominator
