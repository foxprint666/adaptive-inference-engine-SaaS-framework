"""
worker/metrics.py

Production-grade metric helpers for the multi-tenant drift detection worker.

Provides:
  - population_stability_index(): proper histogram-bin PSI (not mean-difference)
  - adversarial_auc(): binary classifier AUC to detect distribution shift
  - Prometheus instrumentation (optional — gracefully absent when prometheus_client
    is not installed so unit tests can run without the dependency)
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus setup (optional dependency)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram

    _DRIFT_PSI_GAUGE = Gauge(
        "mlops_drift_psi",
        "Current PSI drift score for a tenant model",
        ["tenant_id", "model_id"],
    )
    _DRIFT_AUC_GAUGE = Gauge(
        "mlops_drift_adversarial_auc",
        "Adversarial validation AUC for a tenant model",
        ["tenant_id", "model_id"],
    )
    _DRIFT_EVENT_COUNTER = Counter(
        "mlops_drift_events_total",
        "Number of drift threshold crossings detected",
        ["tenant_id", "model_id", "metric"],
    )
    _RETRAIN_TRIGGER_COUNTER = Counter(
        "mlops_retraining_triggers_total",
        "Number of retraining jobs triggered by the worker",
        ["tenant_id", "model_id", "reason"],
    )
    _TELEMETRY_RECORDS_GAUGE = Gauge(
        "mlops_telemetry_queue_depth",
        "Number of telemetry records in the tenant queue",
        ["tenant_id", "model_id"],
    )
    _INFERENCE_LATENCY_HISTOGRAM = Histogram(
        "mlops_worker_check_latency_seconds",
        "Time taken per drift check cycle",
        ["tenant_id", "model_id"],
    )

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available — metrics will be exported")

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed; Prometheus metrics will be skipped. "
        "Install with: pip install prometheus_client"
    )


# ---------------------------------------------------------------------------
# Fix 7: PSI — Population Stability Index (continuous + categorical)
# ---------------------------------------------------------------------------

_DEFAULT_BINS = 10
_EPSILON = 1e-8  # avoids log(0) and division by zero


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    bins: int = _DEFAULT_BINS,
) -> float:
    """
    Compute Population Stability Index for CONTINUOUS features using the
    standard histogram-bin formula.

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    A PSI < 0.10 indicates no significant drift.
    A PSI between 0.10 – 0.25 is a moderate shift worth monitoring.
    A PSI >= 0.25 indicates significant population shift — retraining is advised.

    Args:
        expected: Reference distribution (baseline telemetry, numeric values).
        actual:   Current distribution (recent telemetry, numeric values).
        bins:     Number of histogram bins (default: 10).

    Returns:
        PSI score (float ≥ 0). Returns 0.0 if either distribution is empty.

    Note:
        Do NOT call this function with categorical string values — use
        categorical_psi() instead, or let smart_psi() auto-dispatch.
    """
    expected_values = list(expected)
    actual_values = list(actual)

    if not expected_values or not actual_values:
        return 0.0

    # Build shared bin edges from the combined range
    combined = expected_values + actual_values
    min_val, max_val = min(combined), max(combined)

    if min_val == max_val:
        # All values identical — no drift measurable
        return 0.0

    bin_width = (max_val - min_val) / bins
    edges: List[float] = [min_val + i * bin_width for i in range(bins + 1)]
    edges[-1] = max_val + _EPSILON  # ensure the max value falls in the last bin

    def _bin_counts(values: List[float]) -> List[int]:
        counts = [0] * bins
        for v in values:
            for b in range(bins):
                if edges[b] <= v < edges[b + 1]:
                    counts[b] += 1
                    break
        return counts

    expected_counts = _bin_counts(expected_values)
    actual_counts = _bin_counts(actual_values)

    n_expected = len(expected_values)
    n_actual = len(actual_values)

    psi = 0.0
    for exp_c, act_c in zip(expected_counts, actual_counts):
        exp_pct = (exp_c / n_expected) if exp_c > 0 else _EPSILON
        act_pct = (act_c / n_actual) if act_c > 0 else _EPSILON
        psi += (act_pct - exp_pct) * math.log(act_pct / exp_pct)

    return max(0.0, psi)


def categorical_psi(
    expected: Iterable,
    actual: Iterable,
) -> float:
    """
    Compute Population Stability Index for CATEGORICAL features.

    Uses raw category frequency proportions instead of histogram bins.

    Laplace smoothing (+1 pseudo-count) handles zero-frequency categories:
        - A category that appears in ``expected`` but not in ``actual`` would
          produce log(0) without smoothing → returns invalid infinity.
        - Smoothing replaces 0-count categories with 1/n_categories instead of 0,
          keeping PSI finite and interpretable.

    PSI = Σ over all categories: (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    Args:
        expected: Iterable of category values from the baseline window.
                  Example: ["iOS", "Android", "iOS", "Web", ...]
        actual:   Iterable of category values from the current window.

    Returns:
        PSI score (float ≥ 0). Returns 0.0 if either sequence is empty.

    Example:
        >>> categorical_psi(["iOS","Android","iOS"], ["Android","Android","Web"])
        0.693...
    """
    exp_list = list(expected)
    act_list = list(actual)

    if not exp_list or not act_list:
        return 0.0

    # Collect all unique categories across both windows
    all_categories = set(exp_list) | set(act_list)
    n_cats = len(all_categories)

    if n_cats == 0:
        return 0.0

    n_exp = len(exp_list)
    n_act = len(act_list)

    # Count raw frequencies
    exp_counts: dict = {cat: 0 for cat in all_categories}
    act_counts: dict = {cat: 0 for cat in all_categories}
    for v in exp_list:
        exp_counts[v] = exp_counts.get(v, 0) + 1
    for v in act_list:
        act_counts[v] = act_counts.get(v, 0) + 1

    psi = 0.0
    for cat in all_categories:
        # Laplace smoothing: add 1 pseudo-count to both numerator and denominator
        # to prevent log(0) when a category is absent in one window.
        exp_pct = (exp_counts[cat] + 1) / (n_exp + n_cats)
        act_pct = (act_counts[cat] + 1) / (n_act + n_cats)
        psi += (act_pct - exp_pct) * math.log(act_pct / exp_pct)

    return max(0.0, psi)


def smart_psi(
    expected: Iterable,
    actual: Iterable,
    feature_type: str = "continuous",
    bins: int = _DEFAULT_BINS,
) -> float:
    """
    Dispatch to the correct PSI function based on feature type.

    This is the preferred entry point for all PSI calculations in the worker.
    It prevents the crash that occurs when population_stability_index() is
    called on categorical string values (can't compare strings with < / >).

    Args:
        expected:     Baseline feature values.
        actual:       Current feature values.
        feature_type: ``"continuous"`` (default) or ``"categorical"``.
                      Pass the value from the model's schema_definition,
                      e.g. schema[feature_name].get("type", "continuous").
        bins:         Histogram bins for continuous PSI (ignored for categorical).

    Returns:
        PSI score (float ≥ 0).

    Example:
        >>> schema = {"os": {"type": "categorical"}, "amount": {"type": "float"}}
        >>> smart_psi(baseline_os, current_os, feature_type=schema["os"]["type"])
        >>> smart_psi(baseline_amt, current_amt, feature_type="continuous")
    """
    ft = (feature_type or "continuous").lower()
    if ft in ("categorical", "string", "str", "object"):
        return categorical_psi(expected, actual)
    return population_stability_index(expected, actual, bins=bins)


# ---------------------------------------------------------------------------
# Adversarial AUC
# ---------------------------------------------------------------------------


def adversarial_auc(
    baseline_features: List[List[float]],
    current_features: List[List[float]],
) -> float:
    """
    Estimate distribution shift via adversarial validation AUC.

    Labels baseline samples as 0 and current samples as 1, then trains a
    lightweight Random Forest binary classifier and returns its cross-validated
    AUC-ROC score.

    An AUC close to 0.50 means the two distributions are indistinguishable.
    An AUC >= 0.72 (configurable) indicates meaningful drift.

    Requires: scikit-learn

    Returns:
        AUC-ROC float in [0.5, 1.0], or 0.5 if sklearn is unavailable or
        insufficient data.
    """
    if not baseline_features or not current_features:
        return 0.5

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        logger.warning(
            "scikit-learn not installed; adversarial AUC check will be skipped. "
            "Install with: pip install scikit-learn"
        )
        return 0.5

    try:
        X = np.array(baseline_features + current_features, dtype=float)
        y = np.array(
            [0] * len(baseline_features) + [1] * len(current_features), dtype=int
        )

        if len(X) < 10:
            # Not enough samples for reliable cross-validation
            return 0.5

        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=min(3, len(X) // 2), scoring="roc_auc")
        return float(scores.mean())
    except Exception as exc:
        logger.warning("Adversarial AUC computation failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Prometheus emission helpers (no-ops when Prometheus is unavailable)
# ---------------------------------------------------------------------------


def record_psi(tenant_id: str, model_id: str, psi: float) -> None:
    """Emit PSI gauge to Prometheus."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_PSI_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(psi)


def record_adversarial_auc(tenant_id: str, model_id: str, auc: float) -> None:
    """Emit adversarial AUC gauge to Prometheus."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_AUC_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(auc)


def record_drift_event(tenant_id: str, model_id: str, metric: str) -> None:
    """Increment the drift-events counter."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_EVENT_COUNTER.labels(
            tenant_id=tenant_id, model_id=model_id, metric=metric
        ).inc()


def record_retrain_trigger(tenant_id: str, model_id: str, reason: str) -> None:
    """Increment the retraining-trigger counter."""
    if PROMETHEUS_AVAILABLE:
        _RETRAIN_TRIGGER_COUNTER.labels(
            tenant_id=tenant_id, model_id=model_id, reason=reason
        ).inc()


def record_queue_depth(tenant_id: str, model_id: str, depth: int) -> None:
    """Set the telemetry queue depth gauge."""
    if PROMETHEUS_AVAILABLE:
        _TELEMETRY_RECORDS_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(depth)


def drift_check_timer(tenant_id: str, model_id: str):
    """
    Context manager / decorator for timing a drift check cycle.

    Usage:
        with drift_check_timer("client-a", "fraudnet-v1"):
            ...
    """
    if PROMETHEUS_AVAILABLE:
        return _INFERENCE_LATENCY_HISTOGRAM.labels(
            tenant_id=tenant_id, model_id=model_id
        ).time()

    # Fallback no-op context manager
    import contextlib
    return contextlib.nullcontext()
