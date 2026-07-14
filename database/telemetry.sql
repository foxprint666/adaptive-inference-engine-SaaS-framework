-- database/telemetry.sql
-- Persistent inference telemetry table for crash-resilient drift detection.
--
-- This table is the durable source of truth for the drift detection worker.
-- Redis remains the hot cache for sub-second access; this table is the
-- fallback when Redis is empty (e.g., after a worker pod restart).
--
-- Append-only by convention — rows are NEVER updated or deleted during
-- normal operation. A background retention job may DELETE rows older than
-- the configured retention window (default: 30 days).

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- Telemetry table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS inference_telemetry (
    id              BIGSERIAL        PRIMARY KEY,
    tenant_id       TEXT             NOT NULL,
    model_id        TEXT             NOT NULL,
    model_version   TEXT             NOT NULL DEFAULT 'unknown',
    ts              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    features        JSONB            NOT NULL,   -- raw feature values as {"amount": 150.0, ...}
    feature_names   TEXT[]           NOT NULL DEFAULT '{}',
    prediction      SMALLINT         NOT NULL,   -- 0 or 1
    probability     DOUBLE PRECISION NOT NULL,   -- model confidence [0.0, 1.0]
    latency_ms      DOUBLE PRECISION,            -- inference wall-clock latency
    request_id      TEXT                         -- optional correlation ID
);

-- ---------------------------------------------------------------------------
-- Indexes for drift worker queries
-- ---------------------------------------------------------------------------

-- Primary query pattern: "last N records for this tenant+model"
CREATE INDEX IF NOT EXISTS idx_telemetry_tenant_model_ts
    ON inference_telemetry (tenant_id, model_id, ts DESC);

-- Secondary query: probability distribution for PSI calculation
CREATE INDEX IF NOT EXISTS idx_telemetry_probability
    ON inference_telemetry (tenant_id, model_id, probability);

-- ---------------------------------------------------------------------------
-- Retention: delete rows older than 30 days (run via pg_cron or external job)
-- ---------------------------------------------------------------------------

-- Example pg_cron schedule (uncomment if pg_cron is available):
-- SELECT cron.schedule(
--   'telemetry-retention',
--   '0 3 * * *',
--   $$DELETE FROM inference_telemetry WHERE ts < NOW() - INTERVAL '30 days'$$
-- );

-- ---------------------------------------------------------------------------
-- Helper view: last 500 predictions per tenant+model (used by worker)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_recent_telemetry AS
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY tenant_id, model_id
            ORDER BY ts DESC
        ) AS rn
    FROM inference_telemetry
) ranked
WHERE rn <= 500;
