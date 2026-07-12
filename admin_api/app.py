"""
FastAPI control plane for tenant and model administration.

This is intentionally lightweight for local development: state is in memory,
but the API shape mirrors the database-backed control plane the SaaS platform
will use in production.
"""

import time
import uuid
from datetime import datetime
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
from inference.tenant_model_registry import ModelMetadata, TenantModelRegistry


app = FastAPI(title="Adaptive Inference Admin API", version="0.1.0")
started_at = time.time()
rate_limiter = RateLimiter()
tenants: Dict[str, TenantRegistrationRequest] = {}
model_registry = TenantModelRegistry()
retraining_jobs: Dict[str, Dict] = {}


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
    active_models = sum(1 for _ in model_registry.registry.values())
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime_seconds=time.time() - started_at,
        active_tenants=len(tenants),
        active_models=active_models,
    )


@app.post("/auth/token")
def issue_token(tenant_id: str = Query(...)):
    return {"access_token": create_access_token(tenant_id), "token_type": "bearer"}


@app.post("/register-tenant", status_code=status.HTTP_201_CREATED)
def register_tenant(request: TenantRegistrationRequest):
    tenants[request.tenant_id] = request
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
    config_path = request.config_path or _default_config_for_framework(request.framework)
    metadata = model_registry.register_model(
        tenant_id=x_tenant_id,
        model_id=request.model_id,
        model_version=request.model_version,
        storage_path=request.storage_path,
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
        registration_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
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
    job_id = str(uuid.uuid4())
    retraining_jobs[job_id] = {
        "job_id": job_id,
        "tenant_id": x_tenant_id,
        "model_id": model_id,
        "trigger_reason": request.trigger_reason,
        "force_retrain": request.force_retrain,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    return retraining_jobs[job_id]


@app.get("/status")
def status_report():
    return {
        "tenants": len(tenants),
        "models": len(model_registry.registry),
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
