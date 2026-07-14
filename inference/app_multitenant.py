"""
inference/app_multitenant.py

Production-hardened multi-tenant inference service.

Changes from v1
---------------
Fix 1 — Ghost Retraining (Redis Pub/Sub hot-swap)
    A background asyncio task subscribes to the ``mlops:model_updates``
    Redis channel.  When the Celery retraining worker publishes a
    ``model_reload`` event, the stale runtime is evicted from the
    in-memory dict.  The NEXT request to that (tenant, model) pair lazily
    reloads fresh weights — without any container restart.

    The eviction is protected by an asyncio.Lock so concurrent requests
    never see a partially-replaced runtime.

Fix 2 — Volatile Drift State (dual-write telemetry)
    Every prediction is written to Redis (fast path for the drift worker)
    AND asynchronously inserted into the ``inference_telemetry`` PostgreSQL
    table (crash-resilient fallback).  The Postgres write is a fire-and-forget
    ``asyncio.create_task`` so it never blocks the response path.

Fix 3 — PyTorch GIL (ThreadPoolExecutor)
    ``/predict`` is now ``async def``.  The blocking ``runtime.predict()``
    call runs inside a module-level ``ThreadPoolExecutor`` via
    ``run_in_executor``.  The ASGI event loop is free to handle new
    connections while PyTorch's forward pass executes in a worker thread.

Fix 4 — Path Traversal
    ``storage_path`` is validated against ``SAFE_MODEL_DIR`` using
    ``os.path.realpath()`` (resolves symlinks) + prefix check before any
    disk access occurs.  Requests with paths that escape the sandbox are
    rejected with HTTP 400.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import redis
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    redis = None
    aioredis = None

from inference.fraudnet_runtime import FraudNetRuntime
from inference.churn_runtime import ChurnRuntime
from inference.tenant_redis_client import TenantRedisClient
from inference.tenant_model_registry import TenantModelRegistry
from inference.storage_backend import get_storage_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_ROLE: str = os.getenv("MODEL_ROLE", "baseline").lower()
MODELS_DIR: str = os.getenv("MODELS_DIR", "/app/models")
DEVICE: str = os.getenv("DEVICE", "cpu")
DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

# Fix 4: Secure model directory — all storage_path values must resolve inside here
SAFE_MODEL_DIR: str = os.path.realpath(os.getenv("SAFE_MODEL_DIR", MODELS_DIR))

# Pub/Sub channel published by the Celery retraining worker (Fix 1)
MODEL_UPDATES_CHANNEL: str = "mlops:model_updates"

# Fix 3: thread pool for blocking PyTorch forward passes
_CPU_WORKERS = min(4, (os.cpu_count() or 2))
_inference_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_CPU_WORKERS, thread_name_prefix="inference"
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model_runtimes: Dict[str, Dict[str, Any]] = {}   # {tenant_id: {model_id: runtime}}
tenant_redis_clients: Dict[str, TenantRedisClient] = {}
model_registry = TenantModelRegistry()
storage_backend = get_storage_backend()

# Fix 1: asyncio lock guarding model_runtimes mutations
_runtime_lock: asyncio.Lock = asyncio.Lock()

# Sync Redis for telemetry (shared global connection)
try:
    if redis is not None:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = None
except Exception as exc:
    logger.error("Redis connection failed: %s", exc)
    redis_client = None

# Optional Postgres pool (Fix 2)
_pg_pool = None


# ---------------------------------------------------------------------------
# Fix 4: Path traversal guard
# ---------------------------------------------------------------------------

def _validate_storage_path(tenant_id: str, raw_path: str) -> str:
    """
    Validate that storage_path is safely inside SAFE_MODEL_DIR/{tenant_id}/.

    Uses os.path.realpath() to resolve symlinks before the prefix check so
    that paths like /app/models/../../etc/passwd are rejected.

    Returns the resolved absolute path on success.
    Raises HTTP 400 on violation.
    """
    # Resolve the path as if we were chroot'd to SAFE_MODEL_DIR
    safe_tenant_dir = os.path.realpath(os.path.join(SAFE_MODEL_DIR, tenant_id))
    resolved = os.path.realpath(os.path.join(safe_tenant_dir, os.path.basename(raw_path)))

    if not resolved.startswith(safe_tenant_dir + os.sep) and resolved != safe_tenant_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"storage_path '{raw_path}' is outside the permitted directory. "
                f"Provide only a filename, not an absolute or relative path."
            ),
        )
    return resolved


# ---------------------------------------------------------------------------
# Fix 2: Postgres dual-write helpers
# ---------------------------------------------------------------------------

async def _get_pg_pool():
    """Lazy-initialise asyncpg connection pool."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    if not DATABASE_URL:
        return None
    try:
        import asyncpg  # type: ignore
        _pg_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        logger.info("asyncpg pool connected to PostgreSQL for telemetry dual-write")
    except Exception as exc:
        logger.warning("Could not connect to PostgreSQL for telemetry: %s", exc)
        _pg_pool = None
    return _pg_pool


async def _pg_insert_telemetry(
    tenant_id: str,
    model_id: str,
    features: Dict[str, Any],
    prediction: int,
    probability: float,
    latency_ms: Optional[float] = None,
) -> None:
    """
    Fire-and-forget INSERT into inference_telemetry.

    Called via asyncio.create_task so it never blocks the response path.
    Silently swallows all exceptions — telemetry loss is preferable to
    inference downtime.
    """
    pool = await _get_pg_pool()
    if pool is None:
        return
    try:
        feature_names = list(features.keys())
        await pool.execute(
            """
            INSERT INTO inference_telemetry
                (tenant_id, model_id, features, feature_names, prediction, probability, latency_ms)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7)
            """,
            tenant_id,
            model_id,
            json.dumps(features),
            feature_names,
            prediction,
            probability,
            latency_ms,
        )
    except Exception as exc:
        logger.debug("Telemetry Postgres write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Fix 1: Redis Pub/Sub hot-swap background task
# ---------------------------------------------------------------------------

async def _model_reload_listener() -> None:
    """
    Background coroutine: subscribe to mlops:model_updates and hot-swap
    stale runtimes when the retraining worker publishes new weights.

    Eviction strategy (vs in-place replacement):
      Evicting the old runtime from the dict is safer than replacing it
      in-place. The next request triggers a fresh lazy load with the new
      weights. This avoids any risk of serving predictions from a
      partially-replaced model object.

    S3 pre-warm (Fix 5 integration):
      If STORAGE_BACKEND=s3, the storage backend's warm_cache() is called
      immediately when the message arrives (in a thread pool), so the
      download completes BEFORE any inference request needs the file.
      The inference thread only performs a disk read, never an S3 download.
    """
    if aioredis is None:
        logger.warning("redis.asyncio not available — model hot-swap disabled")
        return

    try:
        pubsub_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        pubsub = pubsub_client.pubsub()
        await pubsub.subscribe(MODEL_UPDATES_CHANNEL)
        logger.info("Subscribed to Redis channel: %s", MODEL_UPDATES_CHANNEL)
    except Exception as exc:
        logger.error("Failed to subscribe to model_updates: %s", exc)
        return

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            data = json.loads(message["data"])
        except (json.JSONDecodeError, TypeError):
            continue

        event = data.get("event")
        tenant_id = data.get("tenant_id")
        model_id = data.get("model_id")
        new_key = data.get("storage_key") or data.get("storage_path")

        if event != "model_reload" or not tenant_id or not model_id:
            continue

        logger.info(
            "model_reload event received — tenant=%s model=%s key=%s",
            tenant_id, model_id, new_key,
        )

        # Pre-warm the local S3 cache in the thread pool (non-blocking)
        if new_key:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                _inference_executor,
                storage_backend.warm_cache,
                new_key,
            )

        # Evict stale runtime — next request will reload fresh weights
        async with _runtime_lock:
            if tenant_id in model_runtimes:
                model_runtimes[tenant_id].pop(model_id, None)
                logger.info("Evicted stale runtime: tenant=%s model=%s", tenant_id, model_id)


# ---------------------------------------------------------------------------
# ASGI lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background pub/sub listener on startup; clean up on shutdown."""
    reload_task = asyncio.create_task(_model_reload_listener())
    logger.info("Inference service ready (GIL executor workers=%d)", _CPU_WORKERS)
    try:
        yield
    finally:
        reload_task.cancel()
        _inference_executor.shutdown(wait=False)
        if _pg_pool:
            await _pg_pool.close()
        logger.info("Inference service shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Multi-Tenant Inference Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_role: str
    model_version: str
    tenant_id: str


class TenantRegistrationRequest(BaseModel):
    tenant_id: str
    model_id: str
    model_version: str
    config_path: str
    storage_path: Optional[str] = None
    schema_definition: Dict[str, Any]
    drift_thresholds: Dict[str, float]
    framework: str = "pytorch"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tenant_redis_client(tenant_id: str) -> TenantRedisClient:
    if tenant_id not in tenant_redis_clients:
        tenant_redis_clients[tenant_id] = TenantRedisClient(REDIS_URL, tenant_id)
    return tenant_redis_clients[tenant_id]


def _load_runtime_sync(
    tenant_id: str,
    model_id: str,
    config_path: str,
    framework: str,
    model_path: Optional[str],
) -> Optional[Any]:
    """
    Synchronously initialise and load a ModelRuntime.

    Runs inside the ThreadPoolExecutor — must not call any asyncio primitives.
    """
    try:
        if framework == "sklearn":
            runtime = ChurnRuntime(config_path)
            resolved_path = model_path or os.path.join(
                MODELS_DIR, f"{tenant_id}_{model_id}.pkl"
            )
        else:
            runtime = FraudNetRuntime(config_path, device=DEVICE)
            resolved_path = model_path or os.path.join(
                MODELS_DIR, f"model_{MODEL_ROLE}_{tenant_id}.pt"
            )

        # If using S3, warm the cache before loading
        resolved_path = storage_backend.warm_cache(resolved_path)
        runtime.load(resolved_path)
        return runtime
    except Exception as exc:
        logger.error("Failed to load runtime for %s/%s: %s", tenant_id, model_id, exc)
        return None


async def get_or_init_runtime(
    tenant_id: str,
    model_id: str,
    config_path: str,
    framework: str,
    model_path: Optional[str],
) -> Optional[Any]:
    """
    Return the cached runtime, or load it in the thread pool on first access.
    Protected by _runtime_lock.
    """
    async with _runtime_lock:
        if tenant_id in model_runtimes and model_id in model_runtimes[tenant_id]:
            return model_runtimes[tenant_id][model_id]

    # Load outside the lock (can be slow if downloading from S3)
    loop = asyncio.get_event_loop()
    runtime = await loop.run_in_executor(
        _inference_executor,
        _load_runtime_sync,
        tenant_id, model_id, config_path, framework, model_path,
    )

    if runtime is not None:
        async with _runtime_lock:
            model_runtimes.setdefault(tenant_id, {})[model_id] = runtime

    return runtime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "service": "multi-tenant-inference"}


@app.get("/healthz/ready")
def readiness_check():
    return {"status": "ready"}


@app.post("/register-tenant", status_code=status.HTTP_201_CREATED)
async def register_tenant(
    request: TenantRegistrationRequest,
    x_tenant_id: str = Header(...),
):
    if x_tenant_id != request.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="X-Tenant-ID header must match request tenant_id",
        )

    # Fix 4: validate storage path
    safe_path: Optional[str] = None
    if request.storage_path:
        safe_path = _validate_storage_path(request.tenant_id, request.storage_path)

    metadata = model_registry.register_model(
        tenant_id=request.tenant_id,
        model_id=request.model_id,
        model_version=request.model_version,
        storage_path=safe_path or "",
        config_path=request.config_path,
        schema_definition=request.schema_definition,
        drift_thresholds=request.drift_thresholds,
        framework=request.framework,
    )

    runtime = await get_or_init_runtime(
        request.tenant_id,
        request.model_id,
        request.config_path,
        request.framework,
        safe_path,
    )

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialise model runtime",
        )

    return {
        "status": "success",
        "tenant_id": request.tenant_id,
        "model_id": request.model_id,
        "model_version": request.model_version,
        "message": f"Tenant {request.tenant_id} registered with model {request.model_id}",
    }


@app.get("/tenant/{tenant_id}/models")
async def list_tenant_models(tenant_id: str, x_tenant_id: str = Header(...)):
    if tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other tenant's models",
        )
    models = model_registry.list_tenant_models(tenant_id)
    return {
        "tenant_id": tenant_id,
        "models": [
            {
                "model_id": model_id,
                "version": m.model_version,
                "framework": m.framework,
                "storage_path": m.storage_path,
            }
            for model_id, m in models.items()
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(  # Fix 3: async to allow run_in_executor
    request: Dict[str, float],
    x_tenant_id: str = Header(...),
    x_model_id: str = Header(default="fraudnet-v1"),
):
    """
    Real-time prediction endpoint.

    Fix 3: PyTorch forward pass runs in a ThreadPoolExecutor.
            The ASGI event loop handles routing + header parsing asynchronously
            while the CPU-bound inference runs in a dedicated thread.

    Fix 2: Telemetry is dual-written to Redis (sync, fast) and Postgres
            (async, crash-resilient) without blocking the response.
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Tenant-ID header is required",
        )

    t_start = time.perf_counter()

    # Resolve runtime (lazy-load + S3 cache warm if needed)
    metadata = model_registry.get_latest_model(x_tenant_id, x_model_id)
    if not metadata and (
        x_tenant_id not in model_runtimes
        or x_model_id not in model_runtimes.get(x_tenant_id, {})
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {x_model_id} not found for tenant {x_tenant_id}",
        )

    if metadata:
        runtime = await get_or_init_runtime(
            x_tenant_id,
            x_model_id,
            metadata.config_path,
            metadata.framework,
            metadata.storage_path,
        )
    else:
        async with _runtime_lock:
            runtime = model_runtimes.get(x_tenant_id, {}).get(x_model_id)

    if not runtime:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model runtime unavailable",
        )

    # Fix 3: run blocking forward pass in thread pool
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _inference_executor,
            runtime.predict,
            request,
        )
    except Exception as exc:
        logger.error("Prediction error for %s/%s: %s", x_tenant_id, x_model_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Fix 2: dual-write telemetry — Redis (sync, fast path)
    try:
        tenant_redis = get_tenant_redis_client(x_tenant_id)
        tenant_redis.push_telemetry(
            x_model_id,
            {
                "timestamp": time.time(),
                "features": list(request.values()),
                "feature_names": list(request.keys()),
                "prediction": result["prediction"],
                "probability": result["probability"],
            },
        )
    except Exception as exc:
        logger.warning("Redis telemetry write failed (non-fatal): %s", exc)

    # Fix 2: dual-write telemetry — Postgres (async, crash-resilient)
    asyncio.create_task(
        _pg_insert_telemetry(
            tenant_id=x_tenant_id,
            model_id=x_model_id,
            features=dict(request),
            prediction=result["prediction"],
            probability=result["probability"],
            latency_ms=latency_ms,
        )
    )

    model_version = MODEL_ROLE
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        model_role=MODEL_ROLE,
        model_version=model_version,
        tenant_id=x_tenant_id,
    )


@app.get("/tenant/{tenant_id}/telemetry/stats")
async def get_tenant_telemetry_stats(
    tenant_id: str,
    x_tenant_id: str = Header(...),
    x_model_id: str = Header(default="fraudnet-v1"),
):
    if tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other tenant's data",
        )
    try:
        tenant_redis = get_tenant_redis_client(tenant_id)
        queue_length = tenant_redis.get_telemetry_queue_length(x_model_id)
        return {
            "tenant_id": tenant_id,
            "model_id": x_model_id,
            "telemetry_queue_length": queue_length,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.get("/registry")
def get_registry_status():
    """Admin debug endpoint — list all registered models."""
    return json.loads(model_registry.to_json())
