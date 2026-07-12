"""
inference/tenant_model_registry.py

Tenant Model Registry abstraction.
Maps tenant_id -> model_id:version -> storage path (S3/GCS).

For Phase 2, we use an in-memory registry. In production, this would be
a PostgreSQL database with proper persistence.
"""

import json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    """
    tenant_id: str
    model_id: str
    model_version: str
    storage_path: str  # e.g., /app/models/model_baseline.pt or s3://bucket/path
    config_path: str  # e.g., inference/config_fraudnet.json
    schema_definition: Dict  # Feature schema from config.json
    drift_thresholds: Dict  # {"psi_threshold": 0.25, "auc_threshold": 0.72}
    framework: str  # "pytorch", "sklearn", etc.


class TenantModelRegistry:
    """
    In-memory registry for tenant models.
    
    Provides:
    - Model registration and lookup
    - Multi-tenant isolation
    - Schema and drift threshold storage
    
    In production, replace with a database.
    """

    def __init__(self):
        """
        Initialize empty registry.
        Format: {(tenant_id, model_id, version): ModelMetadata}
        """
        self.registry: Dict[Tuple[str, str, str], ModelMetadata] = {}
        logger.info("TenantModelRegistry initialized")

    def register_model(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        storage_path: str,
        schema_definition: Dict,
        drift_thresholds: Dict,
        framework: str = "pytorch",
        config_path: Optional[str] = None,
    ) -> ModelMetadata:
        """
        Register a model for a tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            model_id: Model name (e.g., "fraudnet-v1")
            model_version: Semantic version (e.g., "1.0.0")
            storage_path: Path/URI to model file
            schema_definition: Feature schema dict
            drift_thresholds: Drift detection thresholds dict
            framework: ML framework ("pytorch", "sklearn", etc.)
            config_path: Path to model config. Defaults to storage_path for legacy callers.
            
        Returns:
            ModelMetadata object
        """
        key = (tenant_id, model_id, model_version)
        
        metadata = ModelMetadata(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            storage_path=storage_path,
            config_path=config_path or storage_path,
            schema_definition=schema_definition,
            drift_thresholds=drift_thresholds,
            framework=framework,
        )
        
        self.registry[key] = metadata
        logger.info(
            f"Registered model: {tenant_id}/{model_id}:{model_version} at {storage_path}"
        )
        return metadata

    def get_model(
        self, tenant_id: str, model_id: str, model_version: str
    ) -> Optional[ModelMetadata]:
        """
        Retrieve model metadata.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model name
            model_version: Model version
            
        Returns:
            ModelMetadata if found, None otherwise
        """
        key = (tenant_id, model_id, model_version)
        return self.registry.get(key)

    def get_latest_model(
        self, tenant_id: str, model_id: str
    ) -> Optional[ModelMetadata]:
        """
        Get the latest version of a model for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model name
            
        Returns:
            Latest ModelMetadata, or None if not found
        """
        matching = [
            metadata
            for (tid, mid, _), metadata in self.registry.items()
            if tid == tenant_id and mid == model_id
        ]
        return matching[-1] if matching else None

    def list_tenant_models(self, tenant_id: str) -> Dict[str, ModelMetadata]:
        """
        List all models registered for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Dict of {model_id: ModelMetadata}
        """
        result = {}
        for (tid, mid, _), metadata in self.registry.items():
            if tid == tenant_id:
                result[mid] = metadata
        return result

    def update_drift_thresholds(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        new_thresholds: Dict,
    ) -> Optional[ModelMetadata]:
        """
        Update drift thresholds for a specific model version.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model name
            model_version: Model version
            new_thresholds: New threshold values
            
        Returns:
            Updated ModelMetadata, or None if not found
        """
        key = (tenant_id, model_id, model_version)
        if key not in self.registry:
            logger.warning(f"Model {key} not found for threshold update")
            return None
        
        metadata = self.registry[key]
        metadata.drift_thresholds.update(new_thresholds)
        logger.info(f"Updated thresholds for {key}: {new_thresholds}")
        return metadata

    def delete_model(
        self, tenant_id: str, model_id: str, model_version: str
    ) -> bool:
        """
        Delete a model registration.
        
        Args:
            tenant_id: Tenant identifier
            model_id: Model name
            model_version: Model version
            
        Returns:
            True if deleted, False if not found
        """
        key = (tenant_id, model_id, model_version)
        if key in self.registry:
            del self.registry[key]
            logger.info(f"Deleted model registration: {key}")
            return True
        logger.warning(f"Model {key} not found for deletion")
        return False

    def to_json(self) -> str:
        """
        Export entire registry as JSON (for debugging).
        
        Returns:
            JSON string representation of registry
        """
        data = {
            str(k): asdict(v) for k, v in self.registry.items()
        }
        return json.dumps(data, indent=2)
