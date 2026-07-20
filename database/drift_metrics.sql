-- database/drift_metrics.sql
--
-- Time-series drift metric history.
--
-- The worker previously called set_metrics() which overwrites a single Redis key.
-- This table stores ONE ROW per drift-check cycle per (tenant, model) pair so
-- the dashboard can draw line charts of PSI/AUC over time.
--
-- Schema design decisions
-- -----------------------
-- * Partitioned by tenant_id (TEXT) + model_id (TEXT) rather than FK so the
--   table can be populated even when the tenant_models row does not yet exist
--   (race-condition safe).
-- * psi_per_feature JSONB holds per-feature PSI scores when available
--   (e.g. {"amount": 0.35, "distance": 0.22}) — NULL for aggregate-only checks.
-- * retraining_triggered BOOLEAN so the dashboard can show "⚡ retrained" markers.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS drift_metrics (
    id                   BIGSERIAL        PRIMARY KEY,
    tenant_id            TEXT             NOT NULL,
    model_id             TEXT             NOT NULL,
    ts                   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    psi                  DOUBLE PRECISION NOT NULL,
    adversarial_auc      DOUBLE PRECISION NOT NULL,
    records_checked      INTEGER          NOT NULL DEFAULT 0,
    drift_detected       BOOLEAN          NOT NULL DEFAULT FALSE,
    retraining_triggered BOOLEAN          NOT NULL DEFAULT FALSE,
    psi_per_feature      JSONB,                              -- optional per-feature breakdown
    drift_reasons        TEXT[]           NOT NULL DEFAULT '{}',
    check_duration_ms    DOUBLE PRECISION                    -- wall-clock time for the check
);

-- Primary query pattern: "last N drift scores for tenant+model for chart"
CREATE INDEX IF NOT EXISTS idx_drift_metrics_tenant_model_ts
    ON drift_metrics (tenant_id, model_id, ts DESC);

-- Allow quick filtering of drift-only rows for alert counting
CREATE INDEX IF NOT EXISTS idx_drift_metrics_detected
    ON drift_metrics (tenant_id, model_id, drift_detected, ts DESC);

-- ---------------------------------------------------------------------------
-- Helper view: last 200 drift check results per (tenant, model)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_drift_history AS
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY tenant_id, model_id
            ORDER BY ts DESC
        ) AS rn
    FROM drift_metrics
) ranked
WHERE rn <= 200;
