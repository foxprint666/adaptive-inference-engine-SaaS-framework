"""
inference/tenant_redis_client.py

Tenant-isolated Redis client using hash tags to prevent CROSSSLOT errors
in clustered Redis environments.

Key pattern: {tenant_id}:model_id:metric_name
All operations are namespaced to a single tenant, with hash tags forcing
all keys to the same cluster slot.
"""

import json
import logging
from typing import Dict, Any, List, Optional

try:
    import redis
except ImportError:  # pragma: no cover - exercised only in minimal environments
    redis = None

logger = logging.getLogger(__name__)


class TenantRedisClient:
    """
    Wraps a standard Redis client to enforce tenant-prefixed namespacing
    with hash tags, preventing CROSSSLOT clustering errors.
    
    All keys are formatted as: {tenant_id}:key_suffix
    The curly braces force Redis to hash only the tenant_id portion,
    ensuring all tenant data maps to a single hash slot in clustered Redis.
    """

    def __init__(self, redis_url: str, tenant_id: str):
        """
        Initialize tenant Redis client.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
            tenant_id: Unique identifier for the tenant (e.g., "client-a")
        """
        if redis is None:
            raise ImportError("redis is required to create TenantRedisClient")
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()
            self.tenant_id = tenant_id
            self.prefix = f"{{{tenant_id}}}"  # Hash tag prefix
            logger.info(f"TenantRedisClient initialized for tenant: {tenant_id}")
        except Exception as e:
            logger.error(f"Failed to initialize TenantRedisClient: {e}")
            raise

    def _format_key(self, key: str) -> str:
        """
        Format a key with tenant prefix and hash tag.
        
        Args:
            key: Key suffix (e.g., "model_id:telemetry_queue")
            
        Returns:
            Formatted key with hash tag (e.g., "{tenant_id}:model_id:telemetry_queue")
        """
        return f"{self.prefix}:{key}"

    def push_telemetry(self, model_id: str, payload: Dict[str, Any]) -> None:
        """
        Push telemetry data to tenant-isolated queue.
        
        Args:
            model_id: Model identifier (e.g., "fraudnet-v1")
            payload: Telemetry payload dictionary
        """
        try:
            key = self._format_key(f"{model_id}:telemetry_queue")
            self.client.rpush(key, json.dumps(payload))
            # Maintain bounded queue (max 10,000 records per tenant per model)
            self.client.ltrim(key, -10000, -1)
            logger.debug(f"Pushed telemetry to {key}")
        except Exception as e:
            logger.error(f"Error pushing telemetry: {e}")
            raise

    def get_telemetry_queue_length(self, model_id: str) -> int:
        """
        Get the length of telemetry queue for a tenant's model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Number of records in queue
        """
        try:
            key = self._format_key(f"{model_id}:telemetry_queue")
            return self.client.llen(key)
        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0

    def get_telemetry_batch(self, model_id: str, start: int = 0, end: int = -1) -> List[Dict[str, Any]]:
        """
        Retrieve telemetry records from queue.
        
        Args:
            model_id: Model identifier
            start: Start index (default: 0)
            end: End index (default: -1, all)
            
        Returns:
            List of parsed telemetry records
        """
        try:
            key = self._format_key(f"{model_id}:telemetry_queue")
            raw_records = self.client.lrange(key, start, end)
            return [json.loads(rec) for rec in raw_records]
        except Exception as e:
            logger.error(f"Error retrieving telemetry batch: {e}")
            return []

    def set_metrics(self, model_id: str, metric_name: str, value: float) -> None:
        """
        Set a metric value (PSI, AUC, etc.) for a tenant's model.
        
        Args:
            model_id: Model identifier
            metric_name: Metric name (e.g., "psi_amount", "auc_score")
            value: Metric value
        """
        try:
            key = self._format_key(f"{model_id}:metrics:{metric_name}")
            self.client.set(key, str(value))
            logger.debug(f"Set metric {metric_name} = {value} for {model_id}")
        except Exception as e:
            logger.error(f"Error setting metric: {e}")
            raise

    def get_metrics(self, model_id: str, metric_name: str) -> Optional[float]:
        """
        Retrieve a metric value for a tenant's model.
        
        Args:
            model_id: Model identifier
            metric_name: Metric name
            
        Returns:
            Metric value as float, or None if not found
        """
        try:
            key = self._format_key(f"{model_id}:metrics:{metric_name}")
            value = self.client.get(key)
            return float(value) if value else None
        except Exception as e:
            logger.error(f"Error getting metric: {e}")
            return None

    def clear_tenant_data(self, model_id: str = None) -> None:
        """
        Clear all data for a tenant (or specific model if provided).
        WARNING: Use with caution in production.
        
        Args:
            model_id: If provided, only clear data for this model. If None, clear all.
        """
        try:
            if model_id:
                pattern = self._format_key(f"{model_id}:*")
            else:
                pattern = self._format_key("*")
            
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys for tenant {self.tenant_id}")
        except Exception as e:
            logger.error(f"Error clearing tenant data: {e}")
            raise

    def get_all_tenant_keys(self) -> List[str]:
        """
        Retrieve all keys for this tenant (for debugging).
        
        Returns:
            List of all keys belonging to this tenant
        """
        try:
            pattern = self._format_key("*")
            return self.client.keys(pattern)
        except Exception as e:
            logger.error(f"Error retrieving tenant keys: {e}")
            return []
