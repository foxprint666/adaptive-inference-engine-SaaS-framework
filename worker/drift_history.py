"""
worker/drift_history.py

Writes timestamped drift metric rows to the drift_metrics Postgres table.

The worker previously only called redis_client.set_metrics() which overwrites
a single key — you can't draw a chart from a single value. This module writes
ONE ROW per drift-check cycle so the dashboard has time-series data.

Usage
-----
Called from worker_multitenant._execute_drift_check() after PSI/AUC are computed:

    from worker.drift_history import record_drift_metrics
    record_drift_metrics(
        tenant_id="bank-a",
        model_id="fraudnet-v1",
        psi=0.34,
        adversarial_auc=0.88,
        records_checked=500,
        drift_detected=True,
        retraining_triggered=False,
        drift_reasons=["psi=0.34 >= threshold=0.25"],
        psi_per_feature={"amount": 35.17, "risk_score": 34.49},
        check_duration_ms=142.3,
    )

Falls back silently when DATABASE_URL is not set.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATABASE_URL = os.getenv("DATABASE_URL")
_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    if not _DATABASE_URL:
        return None
    try:
        from sqlalchemy import (
            BigInteger, Boolean, Column, DateTime, Double,
            Integer, MetaData, String, Table, create_engine, func,
        )
        from sqlalchemy.dialects.postgresql import ARRAY, JSONB
        engine = create_engine(_DATABASE_URL, pool_pre_ping=True, future=True)
        meta = MetaData()
        table = Table(
            "drift_metrics",
            meta,
            Column("id", BigInteger, primary_key=True),
            Column("tenant_id", String, nullable=False),
            Column("model_id", String, nullable=False),
            Column("ts", DateTime(timezone=True), server_default=func.now()),
            Column("psi", Double, nullable=False),
            Column("adversarial_auc", Double, nullable=False),
            Column("records_checked", Integer, nullable=False, default=0),
            Column("drift_detected", Boolean, nullable=False, default=False),
            Column("retraining_triggered", Boolean, nullable=False, default=False),
            Column("psi_per_feature", JSONB, nullable=True),
            Column("drift_reasons", ARRAY(String), nullable=False, default=[]),
            Column("check_duration_ms", Double, nullable=True),
        )
        # Ensure the table exists (idempotent)
        meta.create_all(engine)
        _engine = (engine, table)
        logger.info("drift_history: connected to Postgres at %s", _DATABASE_URL[:30] + "...")
    except Exception as exc:
        logger.warning("drift_history: could not connect to Postgres (%s) — history disabled", exc)
        _engine = None
    return _engine


def record_drift_metrics(
    tenant_id: str,
    model_id: str,
    psi: float,
    adversarial_auc: float,
    records_checked: int = 0,
    drift_detected: bool = False,
    retraining_triggered: bool = False,
    drift_reasons: Optional[List[str]] = None,
    psi_per_feature: Optional[Dict[str, float]] = None,
    check_duration_ms: Optional[float] = None,
) -> bool:
    """
    Insert one drift metric row. Returns True on success, False on failure.

    Designed to be called from the drift worker — all exceptions are caught
    so a DB failure never interrupts the drift check loop.
    """
    result = _get_engine()
    if result is None:
        return False
    engine, table = result
    try:
        with engine.begin() as conn:
            conn.execute(
                table.insert().values(
                    tenant_id=tenant_id,
                    model_id=model_id,
                    psi=float(psi),
                    adversarial_auc=float(adversarial_auc),
                    records_checked=int(records_checked),
                    drift_detected=bool(drift_detected),
                    retraining_triggered=bool(retraining_triggered),
                    psi_per_feature=psi_per_feature,
                    drift_reasons=drift_reasons or [],
                    check_duration_ms=check_duration_ms,
                )
            )
        return True
    except Exception as exc:
        logger.error("drift_history: failed to write row for %s/%s: %s", tenant_id, model_id, exc)
        return False


def query_drift_history(
    tenant_id: str,
    model_id: str,
    limit: int = 200,
) -> List[Dict]:
    """
    Return the last `limit` drift metric rows for a (tenant, model) pair.
    Used by the admin API to serve chart data to the dashboard.
    Returns an empty list if Postgres is unavailable.
    """
    result = _get_engine()
    if result is None:
        return []
    engine, table = result
    try:
        from sqlalchemy import select
        q = (
            select(table)
            .where(
                table.c.tenant_id == tenant_id,
                table.c.model_id == model_id,
            )
            .order_by(table.c.ts.desc())
            .limit(limit)
        )
        with engine.connect() as conn:
            rows = conn.execute(q).mappings().all()
        return [_row_to_dict(r) for r in reversed(rows)]  # chronological order for charts
    except Exception as exc:
        logger.error("drift_history: query failed for %s/%s: %s", tenant_id, model_id, exc)
        return []


def _row_to_dict(row) -> Dict:
    return {
        "ts": row["ts"].isoformat() if row["ts"] else None,
        "psi": float(row["psi"]),
        "adversarial_auc": float(row["adversarial_auc"]),
        "records_checked": int(row["records_checked"]),
        "drift_detected": bool(row["drift_detected"]),
        "retraining_triggered": bool(row["retraining_triggered"]),
        "drift_reasons": list(row["drift_reasons"] or []),
        "psi_per_feature": dict(row["psi_per_feature"]) if row["psi_per_feature"] else None,
        "check_duration_ms": row["check_duration_ms"],
    }
