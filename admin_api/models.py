"""
admin_api/models.py - Pydantic schemas for Phase 3 control plane
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class ModelRegistrationRequest(BaseModel):
    """Request to register a new model for a tenant."""
    model_id: str = Field(..., example="fraudnet-v1")
    model_version: str = Field(..., example="1.0.0")
    storage_path: str = Field(..., example="s3://bucket/models/fraudnet.pt")
    config_path: Optional[str] = Field(default=None, example="inference/config_fraudnet.json")
    schema_definition: Dict[str, Any]
    drift_thresholds: Dict[str, float] = Field(default={"psi_threshold": 0.25, "auc_threshold": 0.72})
    framework: str = Field(default="pytorch")


class RetrainingRequest(BaseModel):
    """Request to trigger retraining for a model."""
    model_id: str
    trigger_reason: str = Field(default="manual")
    force_retrain: bool = Field(default=False)


class TenantRegistrationRequest(BaseModel):
    """Request to register a new tenant."""
    tenant_id: str
    tenant_name: str
    contact_email: str
    tier: str = Field(default="standard")  # standard, premium, enterprise


class ModelRegistrationResponse(BaseModel):
    """Response after model registration."""
    status: str
    model_id: str
    model_version: str
    tenant_id: str
    registration_id: str
    created_at: datetime
    message: str


class HealthResponse(BaseModel):
    """Service health status."""
    status: str
    version: str
    uptime_seconds: float
    active_tenants: int
    active_models: int
