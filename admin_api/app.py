"""
admin_api/app.py

Production FastAPI control plane — fully functional, no scaffolding.

New endpoints added over Phase 3:
  GET  /tenants                     — list all tenants (admin)
  GET  /tenants/{tenant_id}         — single tenant detail + model list
  GET  /admin/models                — all models across all tenants (admin)
  GET  /models/{model_id}/drift     — PSI/AUC time-series for chart
  POST /auth/dashboard/login        — real email+password login → session JWT
  POST /auth/dashboard/register     — create dashboard user account
  GET  /auth/dashboard/me           — return current session user info
  GET  /events/stream               — SSE stream of mlops:model_updates Redis channel
  GET  /health                      — service health + tenant/model counts
  GET  /status                      — aggregate status

CORS: all origins allowed in dev; set CORS_ORIGINS env var in prod.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI, Header, HTTPException, Query, Request, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from admin_api.auth import create_access_token, verify_bearer_token
from admin_api.user_auth import (
    DashboardUser,
    create_session_token,
    decode_session_token,
    user_store,
    verify_password,
)
from admin_api.models import (
    HealthResponse,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    RetrainingRequest,
    TenantRegistrationRequest,
)
from admin_api.rate_limiter import RateLimiter
from admin_api.retraining_orchestrator import (
    enqueue_retraining_job,
    get_retraining_job_status,
)
from inference.tenant_model_registry import ModelMetadata, TenantModelRegistry
from worker.drift_history import query_drift_history

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adaptive Inference Admin API",
    version="0.2.0",
    description="Control plane for tenant management, model registry, drift monitoring, and retraining.",
)

_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

started_at = time.time()
rate_limiter = RateLimiter()
model_registry = TenantModelRegistry()
retraining_jobs: Dict[str, Dict] = {}

SAFE_MODEL_DIR: str = os.path.realpath(
    os.getenv("SAFE_MODEL_DIR", os.getenv("MODELS_DIR", "/app/models"))
)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Path traversal guard
# ---------------------------------------------------------------------------

def _validate_storage_path(tenant_id: str, raw_path: str) -> str:
    safe_tenant_dir = os.path.realpath(os.path.join(SAFE_MODEL_DIR, tenant_id))
    filename = os.path.basename(raw_path) if raw_path else ""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="storage_path must be a non-empty filename.",
        )
    resolved = os.path.realpath(os.path.join(safe_tenant_dir, filename))
    if not resolved.startswith(safe_tenant_dir + os.sep) and resolved != safe_tenant_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"storage_path '{raw_path}' escapes the permitted directory.",
        )
    return resolved


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def require_tenant(authorization: str, x_tenant_id: str) -> str:
    """Validate ML service-to-service Bearer JWT."""
    token_tenant_id = verify_bearer_token(authorization)
    if token_tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token tenant does not match X-Tenant-ID",
        )
    if not rate_limiter.is_allowed(x_tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Tenant rate limit exceeded",
        )
    return token_tenant_id


def require_session(request: Request) -> DashboardUser:
    """Validate dashboard session JWT from Authorization header or cookie."""
    token: Optional[str] = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    if not token:
        token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Login at /auth/dashboard/login",
        )
    try:
        from jose import JWTError
        payload = decode_session_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token invalid or expired",
        )
    store = user_store()
    user = store.get_by_id(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )
    return user


def require_admin(request: Request) -> DashboardUser:
    user = require_session(request)
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


# ---------------------------------------------------------------------------
# Pydantic request/response models (new for this version)
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: Optional[str]
    display_name: str


class CreateUserRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = ""
    role: str = "viewer"
    tenant_id: Optional[str] = None


class TenantDetail(BaseModel):
    tenant_id: str
    tenant_name: str
    contact_email: str
    tier: str
    model_count: int


class ModelDetail(BaseModel):
    tenant_id: str
    model_id: str
    model_version: str
    framework: str
    storage_path: str
    config_path: str
    drift_thresholds: Dict[str, Any]


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime_seconds=time.time() - started_at,
        active_tenants=model_registry.count_tenants(),
        active_models=model_registry.count_models(),
    )


@app.get("/status")
def status_report():
    return {
        "tenants": model_registry.count_tenants(),
        "models": model_registry.count_models(),
        "retraining_jobs": len(retraining_jobs),
        "uptime_seconds": round(time.time() - started_at, 1),
    }


# ---------------------------------------------------------------------------
# Dashboard auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/dashboard/login", response_model=LoginResponse)
def dashboard_login(req: LoginRequest):
    """Authenticate a dashboard user with email + password."""
    store = user_store()
    user = store.get_by_email(req.email)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    # Record last login if Postgres store
    if hasattr(store, "record_login"):
        try:
            store.record_login(user.id)
        except Exception:
            pass
    token = create_session_token(user)
    return LoginResponse(
        access_token=token,
        role=user.role,
        tenant_id=user.tenant_id,
        display_name=user.display_name,
    )


@app.post("/auth/dashboard/register", status_code=status.HTTP_201_CREATED)
def dashboard_register(req: CreateUserRequest, request: Request):
    """Create a new dashboard user. Only admins can create admin users."""
    # Viewer self-registration is allowed; admin creation requires an existing admin session
    if req.role == "admin":
        require_admin(request)
    store = user_store()
    try:
        user = store.create_user(
            email=req.email,
            password=req.password,
            display_name=req.display_name,
            role=req.role,
            tenant_id=req.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return {"id": user.id, "email": user.email, "role": user.role}


@app.get("/auth/dashboard/me")
def dashboard_me(request: Request):
    """Return the current session user's profile."""
    user = require_session(request)
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "tenant_id": user.tenant_id,
    }


# ---------------------------------------------------------------------------
# ML service token issuance (unchanged, used by inference/worker services)
# ---------------------------------------------------------------------------

@app.post("/auth/token")
def issue_token(tenant_id: str = Query(...)):
    """Issue a machine-to-machine JWT for a tenant. No password required (service-internal)."""
    return {"access_token": create_access_token(tenant_id), "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Tenant management
# ---------------------------------------------------------------------------

@app.post("/register-tenant", status_code=status.HTTP_201_CREATED)
def register_tenant(request: TenantRegistrationRequest):
    model_registry.register_tenant(
        tenant_id=request.tenant_id,
        tenant_name=request.tenant_name,
        contact_email=request.contact_email,
        tier=request.tier,
    )
    return {
        "status": "success",
        "tenant_id": request.tenant_id,
        "message": f"Tenant {request.tenant_id} registered",
    }


@app.get("/tenants")
def list_tenants(request: Request):
    """
    Return all registered tenants.
    Admin users see all; viewer users see only their own tenant.
    """
    user = require_session(request)
    all_tenants = model_registry.list_all_tenants()

    if user.role == "viewer" and user.tenant_id:
        # Scope to their own tenant
        tenant = all_tenants.get(user.tenant_id)
        tenants_list = [tenant] if tenant else []
    else:
        tenants_list = list(all_tenants.values())

    result = []
    for t in tenants_list:
        models = model_registry.list_tenant_models(t.tenant_id)
        result.append({
            "tenant_id": t.tenant_id,
            "tenant_name": t.tenant_name,
            "contact_email": t.contact_email,
            "tier": t.tier,
            "model_count": len(models),
        })
    return {"tenants": result, "total": len(result)}


@app.get("/tenants/{tenant_id}")
def get_tenant(tenant_id: str, request: Request):
    """Single tenant detail with full model list."""
    user = require_session(request)
    if user.role == "viewer" and user.tenant_id != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another tenant's details",
        )
    tenant = model_registry.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    models = model_registry.list_tenant_models(tenant_id)
    return {
        "tenant_id": tenant.tenant_id,
        "tenant_name": tenant.tenant_name,
        "contact_email": tenant.contact_email,
        "tier": tenant.tier,
        "models": [_metadata_to_dict(m) for m in models.values()],
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@app.post("/models/register", response_model=ModelRegistrationResponse)
def register_model(
    request: ModelRegistrationRequest,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    safe_path = _validate_storage_path(x_tenant_id, request.storage_path)
    config_path = request.config_path or _default_config_for_framework(request.framework)
    metadata = model_registry.register_model(
        tenant_id=x_tenant_id,
        model_id=request.model_id,
        model_version=request.model_version,
        storage_path=safe_path,
        config_path=config_path,
        schema_definition=request.schema_definition,
        drift_thresholds=request.drift_thresholds,
        framework=request.framework,
    )
    return ModelRegistrationResponse(
        status="success",
        model_id=metadata.model_id,
        model_version=metadata.model_version,
        tenant_id=metadata.tenant_id,
        registration_id=f"{metadata.tenant_id}:{metadata.model_id}:{metadata.model_version}",
        created_at=datetime.now(timezone.utc),
        message=f"Model {metadata.model_id}:{metadata.model_version} registered",
    )


@app.get("/models")
def list_models(
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    """List models for the calling tenant (machine-to-machine)."""
    require_tenant(authorization, x_tenant_id)
    return {
        "tenant_id": x_tenant_id,
        "models": [
            _metadata_to_dict(m)
            for m in model_registry.list_tenant_models(x_tenant_id).values()
        ],
    }


@app.get("/admin/models")
def list_all_models(request: Request):
    """Admin-level: all models across all tenants."""
    require_admin(request)
    all_models = model_registry.list_all_models()
    return {
        "models": [_metadata_to_dict(m) for m in all_models],
        "total": len(all_models),
    }


@app.get("/models/{model_id}/drift")
def get_drift_history(
    model_id: str,
    request: Request,
    x_tenant_id: str = Header(...),
    limit: int = Query(default=200, le=500),
):
    """
    Return time-series PSI/AUC history for a model.
    Used by the dashboard to render drift charts.
    """
    user = require_session(request)
    if user.role == "viewer" and user.tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another tenant's drift metrics",
        )
    rows = query_drift_history(x_tenant_id, model_id, limit=limit)
    return {
        "tenant_id": x_tenant_id,
        "model_id": model_id,
        "history": rows,
        "points": len(rows),
    }


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

@app.post("/models/{model_id}/retrain", status_code=status.HTTP_202_ACCEPTED)
def request_retraining(
    model_id: str,
    request: RetrainingRequest,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    if request.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path model_id must match request model_id",
        )
    job = enqueue_retraining_job(
        tenant_id=x_tenant_id,
        model_id=model_id,
        trigger_reason=request.trigger_reason,
        force_retrain=request.force_retrain,
    )
    retraining_jobs[job["job_id"]] = job
    return job


@app.get("/retraining/{job_id}")
def retraining_status(
    job_id: str,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    known_job = retraining_jobs.get(job_id)
    if known_job and known_job["tenant_id"] != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another tenant's retraining job",
        )
    return get_retraining_job_status(job_id)


# ---------------------------------------------------------------------------
# SSE: live model update events from Redis pub/sub
# ---------------------------------------------------------------------------

@app.get("/events/stream")
async def events_stream(request: Request):
    """
    Server-Sent Events endpoint.

    Subscribes to the ``mlops:model_updates`` Redis channel and relays each
    message to the browser as an SSE event. The dashboard uses this for the
    live "model updated" notification banner and hot-swap timeline.

    Falls back to a keep-alive heartbeat if Redis is unavailable.
    """
    async def _generator():
        redis_available = False
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(REDIS_URL, decode_responses=True)
            pubsub = client.pubsub()
            await pubsub.subscribe("mlops:model_updates")
            redis_available = True
            logger.info("SSE /events/stream: subscribed to mlops:model_updates")
        except Exception as exc:
            logger.warning("SSE: Redis unavailable (%s) — sending heartbeats only", exc)

        heartbeat_interval = 15  # seconds
        last_heartbeat = asyncio.get_event_loop().time()

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                if redis_available:
                    try:
                        message = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True),
                            timeout=1.0,
                        )
                        if message and message.get("type") == "message":
                            data = message.get("data", "")
                            yield f"data: {data}\n\n"
                            last_heartbeat = asyncio.get_event_loop().time()
                    except asyncio.TimeoutError:
                        pass
                    except Exception as exc:
                        logger.error("SSE pubsub error: %s", exc)
                        redis_available = False

                # Periodic heartbeat to keep connection alive
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= heartbeat_interval:
                    ts = datetime.now(timezone.utc).isoformat()
                    yield f": heartbeat {ts}\n\n"
                    last_heartbeat = now

                if not redis_available:
                    await asyncio.sleep(1)
        finally:
            if redis_available:
                try:
                    await pubsub.unsubscribe("mlops:model_updates")
                    await client.aclose()
                except Exception:
                    pass

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metadata_to_dict(metadata: ModelMetadata) -> Dict:
    return {
        "tenant_id": metadata.tenant_id,
        "model_id": metadata.model_id,
        "model_version": metadata.model_version,
        "storage_path": metadata.storage_path,
        "config_path": metadata.config_path,
        "framework": metadata.framework,
        "drift_thresholds": metadata.drift_thresholds,
        "schema_features": list(metadata.schema_definition.keys()) if metadata.schema_definition else [],
    }


def _default_config_for_framework(framework: str) -> str:
    if framework == "sklearn":
        return "inference/config_churn.json"
    return "inference/config_fraudnet.json"
