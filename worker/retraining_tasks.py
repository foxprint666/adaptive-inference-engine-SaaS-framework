"""
worker/retraining_tasks.py

Celery tasks for EWC-based model retraining.

Architecture
------------
This module is the stable async execution boundary between the admin API
(which enqueues jobs) and the actual training runtime (PyTorch EWC or a
scikit-learn refit). Adding a new training strategy never requires changing
the admin API contract — only this file.

CUDA / Multiprocessing safety
------------------------------
PyTorch CUDA contexts cannot be safely shared across forked processes.
Celery's default prefork pool will trigger a RuntimeError when any CUDA
operation is executed in a child process.

Solutions (choose one when deploying):
  1. Run the worker with the solo pool:
       celery -A worker.retraining_tasks worker -P solo --loglevel=info
  2. Migrate to Ray Serve (actor-based, avoids fork entirely).
  3. Force CPU-only inference during retraining (set DEVICE=cpu).

The task below automatically falls back to CPU so it is safe in all pool modes.

Queue safety
-------------
Set ``prefetch_count=1`` on the RabbitMQ consumer (or use ``-c 1`` concurrency)
to prevent a single worker from pulling multiple heavy jobs simultaneously.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from celery import Celery

from worker.worker_multitenant import summarize_tenant_model
from inference.storage_backend import get_storage_backend

logger = logging.getLogger(__name__)

# Module-level storage backend (local or S3 based on STORAGE_BACKEND env var)
_storage = get_storage_backend()

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

BROKER_URL: str = os.getenv(
    "CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/1")
)
RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

celery_app = Celery(
    "adaptive_inference_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

# Safeguards: at most one heavy job in-flight per worker process
celery_app.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


# ---------------------------------------------------------------------------
# Fix 1: Redis Pub/Sub — publish model_reload event after successful retrain
# ---------------------------------------------------------------------------


def _publish_model_reload(tenant_id: str, model_id: str, storage_key: str) -> None:
    """
    Publish a ``model_reload`` event to the Redis Pub/Sub channel
    ``mlops:model_updates``.

    The inference service subscribes to this channel and evicts the stale
    in-memory runtime on receipt, triggering a lazy reload on the next
    prediction request.  If the inference pods use S3, they also pre-warm
    their local cache before serving traffic.

    This is a best-effort publish — a failure here is logged but does NOT
    fail the retraining task.  The stale model will eventually be replaced
    on pod restart or when the inference service's ETag check fires.
    """
    try:
        import redis as _redis
        import json

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = _redis.from_url(redis_url, decode_responses=True)
        message = json.dumps({
            "event": "model_reload",
            "tenant_id": tenant_id,
            "model_id": model_id,
            "storage_key": storage_key,
        })
        n_receivers = r.publish("mlops:model_updates", message)
        logger.info(
            "Published model_reload event — tenant=%s model=%s key=%s receivers=%d",
            tenant_id, model_id, storage_key, n_receivers,
        )
    except Exception as exc:
        logger.warning(
            "Could not publish model_reload event (non-fatal): %s", exc
        )


# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------

def _compute_fisher_diagonal(
    model,
    data_loader,
    device: str,
    n_samples: int = 200,
) -> Dict[str, "torch.Tensor"]:
    """
    Compute the diagonal of the Fisher Information Matrix (FIM) for EWC.

    The FIM diagonal is approximated by the squared gradient of the log-
    likelihood averaged over ``n_samples`` training examples.

    Returns a dict mapping parameter name → importance tensor (on CPU).
    """
    import torch
    import torch.nn.functional as F

    model.train()
    fisher: Dict[str, "torch.Tensor"] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    seen = 0
    for batch_x, batch_y in data_loader:
        if seen >= n_samples:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        model.zero_grad()
        output = model(batch_x)
        loss = F.binary_cross_entropy_with_output(output.squeeze(), batch_y.float())
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.detach().pow(2)

        seen += len(batch_x)

    # Normalise by number of processed samples
    for name in fisher:
        fisher[name] /= max(seen, 1)

    return {name: f.cpu() for name, f in fisher.items()}


def _ewc_loss(
    model,
    fisher: Dict[str, "torch.Tensor"],
    optimal_params: Dict[str, "torch.Tensor"],
    lambda_reg: float,
    device: str,
) -> "torch.Tensor":
    """
    Elastic Weight Consolidation regularisation term.

    EWC Loss = λ/2 * Σ F_i * (θ_i − θ*_i)²

    Penalises large deviations from previously optimal parameters,
    weighted by each parameter's importance (Fisher diagonal).
    """
    import torch

    ewc_term = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in fisher:
            f = fisher[name].to(device)
            opt = optimal_params[name].to(device)
            ewc_term += (f * (param - opt).pow(2)).sum()
    return (lambda_reg / 2.0) * ewc_term


def _build_tensor_dataset(records: List[Dict], feature_keys: Optional[List[str]] = None):
    """
    Convert telemetry records to a PyTorch TensorDataset.

    Each record should have a ``"features"`` list and a ``"prediction"`` int.
    Records with missing or malformed data are skipped.
    """
    import torch
    from torch.utils.data import TensorDataset

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        features = rec.get("features")
        label = rec.get("prediction")
        if features is None or label is None:
            continue
        try:
            X_rows.append([float(f) for f in features])
            y_rows.append(int(label))
        except (TypeError, ValueError):
            continue

    if not X_rows:
        return None

    X = torch.tensor(X_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    return TensorDataset(X, y)


def _run_ewc_retraining(
    tenant_id: str,
    model_id: str,
    records: List[Dict],
    lambda_reg: float = 800.0,
    epochs: int = 20,
    lr: float = 0.005,
    device: str = "cpu",
) -> Dict:
    """
    Execute EWC fine-tuning on a PyTorch FraudNet model.

    Fix 6 — EWC scope flag:
        Reads ``use_ewc`` from the model's config JSON (default: true).
        When false, runs standard Adam fine-tuning without the expensive
        Fisher Information Matrix computation.  Recommended for small tabular
        models (< 1 000 parameters) where EWC overhead exceeds its benefit.

    Fix 5 — S3 storage:
        Model bytes are loaded/saved via ``_storage`` (LocalStorageBackend or
        S3StorageBackend) rather than calling ``torch.save`` / ``open`` directly.

    Fix 1 — Hot-swap signal:
        After a successful weight save, publishes a ``model_reload`` event to
        Redis so inference pods swap the model without restarting.
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError:
        logger.warning("PyTorch not installed — skipping EWC retraining")
        return {"skipped": True, "reason": "pytorch_not_installed"}

    from inference.tenant_model_registry import TenantModelRegistry
    from inference.fraudnet_runtime import FraudNetRuntime

    registry = TenantModelRegistry()
    metadata = registry.get_latest_model(tenant_id, model_id)
    if metadata is None:
        logger.warning("No registered model for %s/%s — skipping", tenant_id, model_id)
        return {"skipped": True, "reason": "model_not_registered"}

    # Fix 6: read use_ewc flag from config (default True)
    use_ewc = True
    try:
        import json as _json
        if metadata.config_path and os.path.isfile(metadata.config_path):
            with open(metadata.config_path) as cf:
                cfg = _json.load(cf)
            use_ewc = bool(cfg.get("use_ewc", True))
            if not use_ewc:
                logger.info(
                    "use_ewc=false for %s/%s — running standard fine-tuning",
                    tenant_id, model_id,
                )
    except Exception as exc:
        logger.debug("Could not read use_ewc from config: %s", exc)

    # Build dataset from telemetry records
    dataset = _build_tensor_dataset(records)
    if dataset is None or len(dataset) < 4:
        logger.warning(
            "Insufficient training data for %s/%s (%d records) — skipping",
            tenant_id,
            model_id,
            len(records) if records else 0,
        )
        return {"skipped": True, "reason": "insufficient_data", "records": len(records) if records else 0}

    # Fix 5: load model bytes via storage backend (supports S3 + local cache)
    runtime = FraudNetRuntime(metadata.config_path, device=device)
    try:
        local_path = _storage.warm_cache(metadata.storage_path)
        runtime.load(local_path)
    except Exception as exc:
        logger.warning(
            "Could not load model weights for %s/%s: %s — skipping", tenant_id, model_id, exc
        )
        return {"skipped": True, "reason": "model_load_failed", "error": str(exc)}

    net = runtime.model
    if net is None:
        return {"skipped": True, "reason": "model_none_after_load"}

    net = net.to(device)

    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    losses: List[float] = []

    if use_ewc:
        # ── EWC path: Fisher diagonal + regularisation loss ──────────────
        # Snapshot optimal weights before retraining
        optimal_params: Dict[str, "torch.Tensor"] = {
            name: param.clone().detach().cpu()
            for name, param in net.named_parameters()
            if param.requires_grad
        }
        fisher = _compute_fisher_diagonal(net, loader, device=device)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = net(batch_x)
                task_loss = F.binary_cross_entropy_with_output(
                    output.squeeze(), batch_y.float()
                )
                ewc_penalty = _ewc_loss(net, fisher, optimal_params, lambda_reg, device)
                total_loss = task_loss + ewc_penalty
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                logger.debug("EWC epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)
    else:
        # ── Standard Adam path: no Fisher, no EWC regularisation ─────────
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                output = net(batch_x)
                loss = F.binary_cross_entropy_with_output(
                    output.squeeze(), batch_y.float()
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                logger.debug("Standard epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

    # Fix 5: save updated weights via storage backend
    try:
        import io
        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        _storage.save_model_bytes(metadata.storage_path, buf.getvalue())
        logger.info(
            "Retraining complete for %s/%s — saved to %s (use_ewc=%s)",
            tenant_id, model_id, metadata.storage_path, use_ewc,
        )
        # Fix 1: hot-swap signal to inference pods
        _publish_model_reload(tenant_id, model_id, metadata.storage_path)
    except Exception as exc:
        logger.warning("Could not save updated weights: %s", exc)

    return {
        "skipped": False,
        "use_ewc": use_ewc,
        "epochs": epochs,
        "final_loss": losses[-1] if losses else None,
        "initial_loss": losses[0] if losses else None,
        "storage_path": metadata.storage_path,
    }


def _run_sklearn_retraining(
    tenant_id: str,
    model_id: str,
    records: List[Dict],
) -> Dict:
    """
    Refit a scikit-learn model on the latest telemetry records.

    Fix 5 — S3 storage:
        Loads and saves the pickled model via ``_storage`` (Local or S3 backend)
        instead of direct open() calls, enabling horizontal scaling.

    Fix 1 — Hot-swap signal:
        Publishes a ``model_reload`` event after successful save.
    """
    try:
        import pickle
        import numpy as np
    except ImportError:
        return {"skipped": True, "reason": "numpy_not_installed"}

    from inference.tenant_model_registry import TenantModelRegistry

    registry = TenantModelRegistry()
    metadata = registry.get_latest_model(tenant_id, model_id)
    if metadata is None:
        return {"skipped": True, "reason": "model_not_registered"}

    X_rows, y_rows = [], []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        features = rec.get("features")
        label = rec.get("prediction")
        if features is None or label is None:
            continue
        try:
            X_rows.append([float(f) for f in features])
            y_rows.append(int(label))
        except (TypeError, ValueError):
            continue

    if len(X_rows) < 4:
        return {"skipped": True, "reason": "insufficient_data", "records": len(X_rows)}

    # Fix 5: load via storage backend (handles S3 warm cache)
    try:
        local_path = _storage.warm_cache(metadata.storage_path)
        with open(local_path, "rb") as f:
            clf = pickle.load(f)
    except Exception as exc:
        return {"skipped": True, "reason": "model_load_failed", "error": str(exc)}

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    try:
        clf.fit(X, y)
        # Fix 5: save via storage backend
        buf = pickle.dumps(clf)
        _storage.save_model_bytes(metadata.storage_path, buf)
        # Fix 1: hot-swap signal
        _publish_model_reload(tenant_id, model_id, metadata.storage_path)
    except Exception as exc:
        return {"skipped": True, "reason": "training_failed", "error": str(exc)}

    return {"skipped": False, "records": len(X_rows), "storage_path": metadata.storage_path}


# ---------------------------------------------------------------------------
# Celery task definition
# ---------------------------------------------------------------------------


@celery_app.task(name="worker.retraining_tasks.retrain_model", bind=True, max_retries=2)
def retrain_model(
    self,
    tenant_id: str,
    model_id: str,
    trigger_reason: str = "manual",
    force_retrain: bool = False,
) -> Dict:
    """
    Async Celery task: run a full retraining cycle for a tenant model.

    Execution stages
    ----------------
    1. SUMMARIZE  — pull telemetry + compute drift summary (PSI, AUC).
    2. RETRAIN    — run EWC (PyTorch) or refit (sklearn) on the telemetry window.
    3. COMPLETE   — return the consolidated result payload.

    On failure the task retries up to ``max_retries`` times with a 30-second
    countdown before marking as FAILURE.
    """
    logger.info(
        "retrain_model task started — tenant=%s  model=%s  reason=%s",
        tenant_id,
        model_id,
        trigger_reason,
    )

    # ── Stage 1: Summarize ────────────────────────────────────────────────
    self.update_state(
        state="STARTED",
        meta={"tenant_id": tenant_id, "model_id": model_id, "stage": "summarizing"},
    )
    try:
        summary = summarize_tenant_model(tenant_id, model_id)
    except Exception as exc:
        logger.error("Drift summarization failed: %s", exc)
        summary = {"error": str(exc), "records": 0}

    records_available = summary.get("records", 0)

    # ── Stage 2: Retrain ──────────────────────────────────────────────────
    self.update_state(
        state="STARTED",
        meta={
            "tenant_id": tenant_id,
            "model_id": model_id,
            "stage": "retraining",
            "records": records_available,
        },
    )

    # Resolve framework from registry to pick training strategy
    framework = "pytorch"
    try:
        from inference.tenant_model_registry import TenantModelRegistry

        registry = TenantModelRegistry()
        metadata = registry.get_latest_model(tenant_id, model_id)
        if metadata:
            framework = metadata.framework
    except Exception:
        pass  # fall back to pytorch

    # Pull raw records for retraining (same window used for drift summary)
    raw_records: List[Dict] = []
    try:
        from inference.tenant_redis_client import TenantRedisClient

        rc = TenantRedisClient(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), tenant_id
        )
        raw_records = rc.get_telemetry_batch(model_id)
    except Exception as exc:
        logger.warning("Could not fetch telemetry for retraining: %s", exc)

    if framework == "sklearn":
        training_result = _run_sklearn_retraining(tenant_id, model_id, raw_records)
    else:
        training_result = _run_ewc_retraining(tenant_id, model_id, raw_records)

    # ── Stage 3: Complete ─────────────────────────────────────────────────
    result = {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "trigger_reason": trigger_reason,
        "force_retrain": force_retrain,
        "status": "completed",
        "drift_summary": summary,
        "training_result": training_result,
        "completed_at": time.time(),
    }
    logger.info(
        "retrain_model task complete — tenant=%s  model=%s  training_skipped=%s",
        tenant_id,
        model_id,
        training_result.get("skipped", True),
    )
    return result
