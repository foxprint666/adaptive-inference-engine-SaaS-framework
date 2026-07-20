"""
worker/worker_multitenant.py

Production-grade, multi-tenant drift detection worker.

Each call to ``run_drift_check_for_tenant`` reads the most recent telemetry
window from tenant-isolated Redis, computes PSI and (optionally) adversarial
AUC, compares them to the thresholds stored in the Tenant Model Registry, and
automatically enqueues a Celery retraining job when drift is detected.

The module can also run as a continuous polling daemon (``__main__``).

Architecture notes
------------------
* Redis keys are hash-tagged (``{tenant_id}:...``) to prevent CROSSSLOT errors
  in clustered Redis.
* All Prometheus metric emission is delegated to ``worker.metrics`` helpers so
  this module stays free of Prometheus imports.
* Retraining is enqueued via ``admin_api.retraining_orchestrator`` — the same
  code path used by the admin HTTP API — ensuring the job contract is stable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from inference.tenant_redis_client import TenantRedisClient
from inference.tenant_model_registry import TenantModelRegistry, ModelMetadata
from worker.metrics import (
    population_stability_index,
    adversarial_auc,
    record_psi,
    record_adversarial_auc,
    record_drift_event,
    record_retrain_trigger,
    record_queue_depth,
    drift_check_timer,
)
from worker.drift_history import record_drift_metrics as _persist_drift_row

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHECK_INTERVAL_SECONDS: float = float(os.getenv("WORKER_CHECK_INTERVAL_SECONDS", "10"))
TELEMETRY_WINDOW_SIZE: int = int(os.getenv("TELEMETRY_WINDOW_SIZE", "500"))

# Default drift thresholds (overridden per-model by the registry)
DEFAULT_PSI_THRESHOLD: float = float(os.getenv("DEFAULT_PSI_THRESHOLD", "0.25"))
DEFAULT_AUC_THRESHOLD: float = float(os.getenv("DEFAULT_AUC_THRESHOLD", "0.72"))

# ---------------------------------------------------------------------------
# Module-level registry (shared across all checks in this process)
# ---------------------------------------------------------------------------

_model_registry: Optional[TenantModelRegistry] = None


def _get_registry() -> TenantModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = TenantModelRegistry()
    return _model_registry


# ---------------------------------------------------------------------------
# Core drift-check function
# ---------------------------------------------------------------------------


def run_drift_check_for_tenant(
    tenant_id: str,
    model_id: str,
    *,
    redis_url: str = REDIS_URL,
    window_size: int = TELEMETRY_WINDOW_SIZE,
    enqueue_retraining: bool = True,
) -> Dict:
    """
    Execute a complete drift check cycle for one (tenant, model) pair.

    Steps
    -----
    1. Pull the most recent ``window_size`` telemetry records from Redis.
    2. Compute PSI against a uniform baseline (probability = 0.5).
    3. Optionally compute adversarial AUC against the first half of the window
       (used as a proxy baseline when no separate held-out set is stored).
    4. Compare scores to per-model thresholds from the Tenant Model Registry.
    5. Emit Prometheus metrics.
    6. If any threshold is breached, enqueue a Celery retraining job.

    Returns
    -------
    A dictionary with the full drift summary (suitable for logging or returning
    as an API response).
    """
    summary: Dict = {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "records": 0,
        "psi": 0.0,
        "adversarial_auc": 0.5,
        "drift_detected": False,
        "retraining_enqueued": False,
        "drift_reasons": [],
        "timestamp": time.time(),
    }

    try:
        with drift_check_timer(tenant_id, model_id):
            cycle_start = time.time()
            _execute_drift_check(
                summary,
                tenant_id=tenant_id,
                model_id=model_id,
                redis_url=redis_url,
                window_size=window_size,
                enqueue_retraining=enqueue_retraining,
            )
            summary["check_duration_ms"] = (time.time() - cycle_start) * 1000
    except Exception as exc:
        logger.error(
            "Drift check failed for %s/%s: %s", tenant_id, model_id, exc, exc_info=True
        )
        summary["error"] = str(exc)
    finally:
        # Always persist to Postgres (silently no-ops if DATABASE_URL unset)
        _persist_drift_row(
            tenant_id=tenant_id,
            model_id=model_id,
            psi=summary.get("psi", 0.0),
            adversarial_auc=summary.get("adversarial_auc", 0.5),
            records_checked=summary.get("records", 0),
            drift_detected=summary.get("drift_detected", False),
            retraining_triggered=summary.get("retraining_enqueued", False),
            drift_reasons=summary.get("drift_reasons", []),
            psi_per_feature=summary.get("psi_per_feature"),
            check_duration_ms=summary.get("check_duration_ms"),
        )

    return summary


def _execute_drift_check(
    summary: Dict,
    *,
    tenant_id: str,
    model_id: str,
    redis_url: str,
    window_size: int,
    enqueue_retraining: bool,
) -> None:
    """Inner implementation — separated so the timer context wraps the full block."""

    # ── 1. Pull telemetry ──────────────────────────────────────────────────
    redis_client = TenantRedisClient(redis_url, tenant_id)
    records = redis_client.get_telemetry_batch(model_id, limit=window_size)
    n_records = len(records)
    summary["records"] = n_records
    record_queue_depth(tenant_id, model_id, n_records)

    if n_records < 2:
        logger.debug(
            "Tenant %s / model %s: only %d records — skipping drift check",
            tenant_id,
            model_id,
            n_records,
        )
        return

    # ── 2. Extract probability scores ─────────────────────────────────────
    probabilities: List[float] = [
        float(r.get("probability", 0.5))
        for r in records
        if isinstance(r, dict)
    ]

    # ── 3. PSI against a uniform baseline (p=0.5 for binary classifiers) ──
    baseline_probs: List[float] = [0.5] * len(probabilities)
    psi_score = population_stability_index(baseline_probs, probabilities)
    summary["psi"] = psi_score
    record_psi(tenant_id, model_id, psi_score)
    redis_client.set_metrics(model_id, "psi_probability", psi_score)

    # ── 4. Adversarial AUC (first half vs second half of the window) ──────
    mid = len(records) // 2
    baseline_features = _extract_features(records[:mid])
    current_features = _extract_features(records[mid:])
    auc_score = adversarial_auc(baseline_features, current_features)
    summary["adversarial_auc"] = auc_score
    record_adversarial_auc(tenant_id, model_id, auc_score)
    redis_client.set_metrics(model_id, "adversarial_auc", auc_score)

    # ── 5. Fetch per-model thresholds from registry ────────────────────────
    registry = _get_registry()
    metadata: Optional[ModelMetadata] = registry.get_latest_model(tenant_id, model_id)
    thresholds = metadata.drift_thresholds if metadata else {}
    psi_threshold = float(thresholds.get("psi_threshold", DEFAULT_PSI_THRESHOLD))
    auc_threshold = float(thresholds.get("auc_threshold", DEFAULT_AUC_THRESHOLD))

    logger.info(
        "Drift check %s/%s — records=%d  PSI=%.4f (threshold=%.2f)  AUC=%.4f (threshold=%.2f)",
        tenant_id,
        model_id,
        n_records,
        psi_score,
        psi_threshold,
        auc_score,
        auc_threshold,
    )

    # ── 6. Threshold comparisons ───────────────────────────────────────────
    drift_reasons: List[str] = []

    if psi_score >= psi_threshold:
        drift_reasons.append(f"psi={psi_score:.4f} >= threshold={psi_threshold}")
        record_drift_event(tenant_id, model_id, "psi")

    if auc_score >= auc_threshold:
        drift_reasons.append(f"adversarial_auc={auc_score:.4f} >= threshold={auc_threshold}")
        record_drift_event(tenant_id, model_id, "adversarial_auc")

    summary["drift_reasons"] = drift_reasons
    summary["drift_detected"] = bool(drift_reasons)

    # ── 7. Enqueue retraining if drift detected ────────────────────────────
    if drift_reasons and enqueue_retraining:
        _trigger_retraining(summary, tenant_id=tenant_id, model_id=model_id, reasons=drift_reasons)


def _trigger_retraining(
    summary: Dict,
    *,
    tenant_id: str,
    model_id: str,
    reasons: List[str],
) -> None:
    """Enqueue a Celery retraining job via the admin API orchestrator."""
    reason_str = "; ".join(reasons)
    logger.warning(
        "DRIFT DETECTED for %s/%s — enqueuing retraining. Reasons: %s",
        tenant_id,
        model_id,
        reason_str,
    )

    try:
        from admin_api.retraining_orchestrator import enqueue_retraining_job

        job = enqueue_retraining_job(
            tenant_id=tenant_id,
            model_id=model_id,
            trigger_reason=f"drift_worker: {reason_str}",
            force_retrain=False,
        )
        summary["retraining_job_id"] = job.get("job_id")
        summary["retraining_enqueued"] = True
        record_retrain_trigger(tenant_id, model_id, "drift_worker")
        logger.info("Retraining job enqueued: %s", job.get("job_id"))
    except ImportError:
        # Celery not installed — log and continue (dev / test environments)
        logger.warning(
            "celery not installed; retraining job was NOT enqueued for %s/%s",
            tenant_id,
            model_id,
        )
        summary["retraining_enqueued"] = False
    except Exception as exc:
        logger.error(
            "Failed to enqueue retraining for %s/%s: %s", tenant_id, model_id, exc
        )
        summary["retraining_enqueued"] = False


def _extract_features(records: List[Dict]) -> List[List[float]]:
    """
    Extract the ``features`` list from telemetry records.

    Each record is expected to have a ``"features"`` key containing a list of
    floats. Records that lack this key or have non-numeric values are skipped.
    """
    result: List[List[float]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        features = rec.get("features")
        if not features:
            continue
        try:
            result.append([float(f) for f in features])
        except (TypeError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility shim (test_phase2 and basic imports use this name)
# ---------------------------------------------------------------------------


def summarize_tenant_model(tenant_id: str, model_id: str) -> Dict:
    """
    Compatibility wrapper around ``run_drift_check_for_tenant``.

    Keeps the Phase 2 test and any existing callers working unchanged.
    """
    return run_drift_check_for_tenant(tenant_id, model_id, enqueue_retraining=False)


# ---------------------------------------------------------------------------
# Continuous polling daemon
# ---------------------------------------------------------------------------


def run_continuous(
    tenants_and_models: List[tuple],
    *,
    check_interval: float = CHECK_INTERVAL_SECONDS,
    redis_url: str = REDIS_URL,
) -> None:
    """
    Continuously poll all (tenant, model) pairs on a fixed interval.

    Args:
        tenants_and_models: list of (tenant_id, model_id) tuples.
        check_interval:     seconds between full check cycles.
        redis_url:          Redis connection string.
    """
    logger.info(
        "Starting continuous drift worker — %d tenant/model pairs, interval=%.1fs",
        len(tenants_and_models),
        check_interval,
    )
    while True:
        cycle_start = time.time()
        for tenant_id, model_id in tenants_and_models:
            summary = run_drift_check_for_tenant(
                tenant_id, model_id, redis_url=redis_url
            )
            logger.info("Cycle result: %s", summary)

        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, check_interval - elapsed)
        logger.debug("Cycle complete in %.2fs — sleeping %.2fs", elapsed, sleep_for)
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Simple env-based configuration for the daemon
    tenant = os.getenv("TENANT_ID", "client-a")
    model = os.getenv("MODEL_ID", "fraudnet-v1")

    # One-shot mode (default) vs continuous daemon
    if os.getenv("WORKER_DAEMON", "0") == "1":
        run_continuous([(tenant, model)])
    else:
        result = run_drift_check_for_tenant(tenant, model)
        import json
        print(json.dumps(result, indent=2))
        sys.exit(0 if not result.get("error") else 1)
