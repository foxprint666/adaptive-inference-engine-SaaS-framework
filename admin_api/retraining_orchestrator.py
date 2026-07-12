"""
Retraining orchestration boundary.

This module deliberately keeps the first implementation synchronous and small.
It gives the admin API a stable place to hand off retraining requests before a
future Ray Serve or Celery backend is introduced.
"""

import uuid
from datetime import datetime
from typing import Dict


def enqueue_retraining_job(
    tenant_id: str,
    model_id: str,
    trigger_reason: str = "manual",
    force_retrain: bool = False,
) -> Dict:
    return {
        "job_id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "model_id": model_id,
        "trigger_reason": trigger_reason,
        "force_retrain": force_retrain,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
