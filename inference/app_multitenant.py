"""
inference/app_multitenant.py

Refactored FastAPI inference service with multi-tenant support.

Key changes from single-tenant app.py:
- Accepts X-Tenant-ID header on all requests
- Manages separate ModelRuntime instances per tenant
- Namespaces telemetry writes to Redis using TenantRedisClient
- Routes requests to tenant-specific models
- Validates tenant authorization
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import redis
except ImportError:  # pragma: no cover - only for minimal test environments
    redis = None

from inference.fraudnet_runtime import FraudNetRuntime
from inference.churn_runtime import ChurnRuntime
from inference.tenant_redis_client import TenantRedisClient
from inference.tenant_model_registry import TenantModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Tenant Inference Service")

# Global state
model_runtimes: Dict[str, Dict[str, Any]] = {}  # {tenant_id: {model_id: runtime}}
tenant_redis_clients: Dict[str, TenantRedisClient] = {}  # {tenant_id: client}
model_registry = TenantModelRegistry()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_ROLE = os.getenv("MODEL_ROLE", "baseline").lower()
MODELS_DIR = "/app/models"
DEVICE = os.getenv("DEVICE", "cpu")

# Global Redis connection for admin operations
try:
    if redis is None:
        raise ImportError("redis package is not installed")
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Connected to global Redis for admin operations")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None


def get_tenant_redis_client(tenant_id: str) -> TenantRedisClient:
    """
    Get or create a tenant-isolated Redis client.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        TenantRedisClient for the tenant
    """
    if tenant_id not in tenant_redis_clients:
        tenant_redis_clients[tenant_id] = TenantRedisClient(REDIS_URL, tenant_id)
    return tenant_redis_clients[tenant_id]


def initialize_tenant_model(
    tenant_id: str,
    model_id: str,
    config_path: str,
    framework: str = "pytorch",
    model_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize a ModelRuntime for a specific tenant.
    
    Args:
        tenant_id: Tenant identifier
        model_id: Model identifier
        config_path: Path to model config.json
        
    Returns:
        Initialized FraudNetRuntime or None if failed
    """
    try:
        if tenant_id not in model_runtimes:
            model_runtimes[tenant_id] = {}
        
        if model_id in model_runtimes[tenant_id]:
            logger.debug(f"Model {model_id} already initialized for {tenant_id}")
            return model_runtimes[tenant_id][model_id]
        
        if framework == "sklearn":
            runtime = ChurnRuntime(config_path)
            resolved_model_path = model_path or os.path.join(MODELS_DIR, f"{tenant_id}_{model_id}.pkl")
        elif framework == "pytorch":
            runtime = FraudNetRuntime(config_path, device=DEVICE)
            resolved_model_path = model_path or os.path.join(MODELS_DIR, f"model_{MODEL_ROLE}_{tenant_id}.pt")
        else:
            raise ValueError(f"Unsupported model framework: {framework}")

        runtime.load(resolved_model_path)
        
        model_runtimes[tenant_id][model_id] = runtime
        logger.info(f"Initialized model {model_id} for tenant {tenant_id}")
        return runtime
    except Exception as e:
        logger.error(f"Error initializing model for {tenant_id}/{model_id}: {e}")
        return None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int
    probability: float
    model_role: str
    model_version: str
    tenant_id: str


class TenantRegistrationRequest(BaseModel):
    """Request model for tenant registration."""
    tenant_id: str
    model_id: str
    model_version: str
    config_path: str  # Path to config.json
    storage_path: Optional[str] = None  # Path to model binary; optional for demos
    schema_definition: Dict[str, Any]
    drift_thresholds: Dict[str, float]
    framework: str = "pytorch"


@app.on_event("startup")
async def startup_event():
    """Initialize default models on startup."""
    logger.info("Service startup: multi-tenant inference ready")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "multi-tenant-inference"}


@app.get("/healthz/ready")
def readiness_check():
    """Readiness probe."""
    return {"status": "ready"}


@app.post("/register-tenant", status_code=status.HTTP_201_CREATED)
def register_tenant(
    request: TenantRegistrationRequest,
    x_tenant_id: str = Header(...),
):
    """
    Register a new tenant and their model.
    
    In production, this would be protected with proper authentication.
    """
    if x_tenant_id != request.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="X-Tenant-ID header must match request tenant_id"
        )
    
    try:
        metadata = model_registry.register_model(
            tenant_id=request.tenant_id,
            model_id=request.model_id,
            model_version=request.model_version,
            storage_path=request.storage_path or "",
            config_path=request.config_path,
            schema_definition=request.schema_definition,
            drift_thresholds=request.drift_thresholds,
            framework=request.framework,
        )
        
        # Initialize the model
        runtime = initialize_tenant_model(
            request.tenant_id,
            request.model_id,
            request.config_path,
            framework=request.framework,
            model_path=request.storage_path,
        )
        
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize model"
            )
        
        return {
            "status": "success",
            "tenant_id": request.tenant_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "message": f"Tenant {request.tenant_id} registered with model {request.model_id}"
        }
    except Exception as e:
        logger.error(f"Error registering tenant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/tenant/{tenant_id}/models")
def list_tenant_models(tenant_id: str, x_tenant_id: str = Header(...)):
    """
    List all models registered for a tenant.
    """
    if tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other tenant's models"
        )
    
    models = model_registry.list_tenant_models(tenant_id)
    return {
        "tenant_id": tenant_id,
        "models": [
            {
                "model_id": model_id,
                "version": metadata.model_version,
                "framework": metadata.framework,
                "storage_path": metadata.storage_path,
            }
            for model_id, metadata in models.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: Dict[str, float],
    x_tenant_id: str = Header(...),
    x_model_id: str = Header(default="fraudnet-v1"),
):
    """
    Generic prediction endpoint with multi-tenant support.
    
    Required headers:
    - X-Tenant-ID: Tenant identifier
    - X-Model-ID: Model identifier (default: fraudnet-v1)
    
    Request body: feature values as dict
    """
    if not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Tenant-ID header is required"
        )
    
    try:
        # Get or initialize tenant's model
        if x_tenant_id not in model_runtimes or x_model_id not in model_runtimes[x_tenant_id]:
            # Try to load from registry
            metadata = model_registry.get_latest_model(x_tenant_id, x_model_id)
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {x_model_id} not found for tenant {x_tenant_id}"
                )
            
            runtime = initialize_tenant_model(
                x_tenant_id,
                x_model_id,
                metadata.config_path,
                framework=metadata.framework,
                model_path=metadata.storage_path,
            )
            if not runtime:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to initialize model"
                )
        else:
            runtime = model_runtimes[x_tenant_id][x_model_id]
        
        # Run prediction
        result = runtime.predict(request)
        
        # Log telemetry to tenant-isolated Redis
        try:
            tenant_redis = get_tenant_redis_client(x_tenant_id)
            telemetry_data = {
                "timestamp": time.time(),
                "features": list(request.values()),
                "prediction": result["prediction"],
                "probability": result["probability"],
            }
            tenant_redis.push_telemetry(x_model_id, telemetry_data)
        except Exception as e:
            logger.warning(f"Error writing tenant telemetry: {e}")
        
        # Determine version label
        model_version = MODEL_ROLE
        if MODEL_ROLE == "candidate":
            candidate_path = os.path.join(MODELS_DIR, "model_candidate.pt")
            if not os.path.exists(candidate_path):
                model_version = "baseline"
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            model_role=MODEL_ROLE,
            model_version=model_version,
            tenant_id=x_tenant_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for tenant {x_tenant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/tenant/{tenant_id}/telemetry/stats")
def get_tenant_telemetry_stats(
    tenant_id: str,
    x_tenant_id: str = Header(...),
    x_model_id: str = Header(default="fraudnet-v1"),
):
    """
    Get telemetry statistics for a tenant's model.
    """
    if tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other tenant's data"
        )
    
    try:
        tenant_redis = get_tenant_redis_client(tenant_id)
        queue_length = tenant_redis.get_telemetry_queue_length(x_model_id)
        
        return {
            "tenant_id": tenant_id,
            "model_id": x_model_id,
            "telemetry_queue_length": queue_length,
        }
    except Exception as e:
        logger.error(f"Error fetching telemetry stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/registry")
def get_registry_status():
    """
    Admin endpoint: view entire model registry (for debugging).
    In production, this would be protected and possibly removed.
    """
    return json.loads(model_registry.to_json())
