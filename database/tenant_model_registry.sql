CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    tenant_name TEXT NOT NULL,
    contact_email TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'standard',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tenant_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    framework TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    config_path TEXT NOT NULL,
    schema_definition JSONB NOT NULL,
    drift_thresholds JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, model_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_tenant_models_lookup
    ON tenant_models (tenant_id, model_id, created_at DESC);
