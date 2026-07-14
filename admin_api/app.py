"""
FastAPI control plane for tenant and model administration.

State is persisted through TenantModelRegistry. Set DATABASE_URL to use
PostgreSQL; omit it for the in-memory development backend.
"""

import os
import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import FastAPI, Header, HTTPException, Query, status

from admin_api.auth import create_access_token, verify_bearer_token
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


app = FastAPI(title="Adaptive Inference Admin API", version="0.1.0")
started_at = time.time()
rate_limiter = RateLimiter()
model_registry = TenantModelRegistry()
retraining_jobs: Dict[str, Dict] = {}

# Fix 4: all registered storage_path values must resolve inside this directory
SAFE_MODEL_DIR: str = os.path.realpath(
    os.getenv("SAFE_MODEL_DIR", os.getenv("MODELS_DIR", "/app/models"))
)


def _validate_storage_path(tenant_id: str, raw_path: str) -> str:
    """
    Reject any storage_path that escapes SAFE_MODEL_DIR/{tenant_id}/.

    Uses os.path.realpath() to resolve symlinks before the prefix check,
    preventing attacks via relative components (../../) or symlink chains.

    Accepts:
      - A bare filename:       "fraudnet_v2.pt"   → /app/models/{tenant_id}/fraudnet_v2.pt
      - A full allowed path:   "/app/models/{tenant_id}/model.pt"

    Rejects:
      - "../../etc/passwd"
      - "/app/models/other_tenant/model.pt"  (cross-tenant)
      - Any path that resolves outside SAFE_MODEL_DIR
    """
    safe_tenant_dir = os.path.realpath(os.path.join(SAFE_MODEL_DIR, tenant_id))
    # Treat raw_path as a filename — strip any directory components first
    # so tenants cannot navigate laterally within the safe root
    filename = os.path.basename(raw_path) if raw_path else ""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="storage_path must be a non-empty filename (e.g. 'model_v2.pt').",
        )
    resolved = os.path.realpath(os.path.join(safe_tenant_dir, filename))
    # Double-check the resolved path is strictly inside the tenant directory
    if not resolved.startswith(safe_tenant_dir + os.sep) and resolved != safe_tenant_dir:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"storage_path '{raw_path}' resolves outside the permitted directory. "
                f"Provide only a filename, not a path."
            ),
        )
    return resolved


def require_tenant(authorization: str, x_tenant_id: str) -> str:
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


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime_seconds=time.time() - started_at,
        active_tenants=model_registry.count_tenants(),
        active_models=model_registry.count_models(),
    )


@app.post("/auth/token")
def issue_token(tenant_id: str = Query(...)):
    return {"access_token": create_access_token(tenant_id), "token_type": "bearer"}


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


@app.post("/models/register", response_model=ModelRegistrationResponse)
def register_model(
    request: ModelRegistrationRequest,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    # Fix 4: validate and sanitize storage_path before it touches the filesystem
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
def list_models(x_tenant_id: str = Header(...), authorization: str = Header(...)):
    require_tenant(authorization, x_tenant_id)
    return {
        "tenant_id": x_tenant_id,
        "models": [
            _metadata_to_dict(metadata)
            for metadata in model_registry.list_tenant_models(x_tenant_id).values()
        ],
    }


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
    status_payload = get_retraining_job_status(job_id)
    known_job = retraining_jobs.get(job_id)
    if known_job and known_job["tenant_id"] != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another tenant's retraining job",
        )
    return status_payload


@app.get("/status")
def status_report():
    return {
        "tenants": model_registry.count_tenants(),
        "models": model_registry.count_models(),
        "retraining_jobs": len(retraining_jobs),
    }


def _metadata_to_dict(metadata: ModelMetadata) -> Dict:
    return {
        "tenant_id": metadata.tenant_id,
        "model_id": metadata.model_id,
        "model_version": metadata.model_version,
        "storage_path": metadata.storage_path,
        "config_path": metadata.config_path,
        "framework": metadata.framework,
        "drift_thresholds": metadata.drift_thresholds,
    }


def _default_config_for_framework(framework: str) -> str:
    if framework == "sklearn":
        return "inference/config_churn.json"
    return "inference/config_fraudnet.json"
