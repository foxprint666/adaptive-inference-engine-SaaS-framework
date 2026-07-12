"""
Background worker skeleton for tenant-isolated drift checks.

The production version will schedule retraining jobs. This local slice reads
tenant telemetry through TenantRedisClient and calculates a simple PSI-like
signal so the data path can be verified end to end.
"""

import os
from typing import Dict, List

from inference.tenant_redis_client import TenantRedisClient
from worker.metrics import population_stability_index


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def summarize_tenant_model(tenant_id: str, model_id: str) -> Dict:
    redis_client = TenantRedisClient(REDIS_URL, tenant_id)
    records = redis_client.get_telemetry_batch(model_id)
    probabilities: List[float] = [
        float(record.get("probability", 0.0))
        for record in records
        if isinstance(record, dict)
    ]
    baseline = [0.5 for _ in probabilities]
    psi = population_stability_index(baseline, probabilities)
    redis_client.set_metrics(model_id, "psi_probability", psi)
    return {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "records": len(records),
        "psi_probability": psi,
    }


if __name__ == "__main__":
    tenant = os.getenv("TENANT_ID", "client-a")
    model = os.getenv("MODEL_ID", "fraudnet-v1")
    print(summarize_tenant_model(tenant, model))
