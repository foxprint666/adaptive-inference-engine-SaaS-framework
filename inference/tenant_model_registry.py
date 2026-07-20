"""
Tenant model registry with PostgreSQL persistence.

Set DATABASE_URL to a PostgreSQL URL to use the production backend, for example:
postgresql+psycopg2://mlops:mlops@postgres:5432/mlops

When DATABASE_URL is not set, the registry falls back to an in-memory backend so
unit tests and local experiments can run without infrastructure.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TenantMetadata:
    tenant_id: str
    tenant_name: str
    contact_email: str
    tier: str = "standard"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    tenant_id: str
    model_id: str
    model_version: str
    storage_path: str
    config_path: str
    schema_definition: Dict
    drift_thresholds: Dict
    framework: str


class TenantModelRegistry:
    """
    Registry facade used by inference, worker, and admin services.

    The public API is stable across backends. PostgreSQL is selected when
    DATABASE_URL is configured; otherwise state is process-local.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url if database_url is not None else os.getenv("DATABASE_URL")
        if self.database_url:
            self.backend = PostgresTenantModelRegistry(self.database_url)
        else:
            self.backend = InMemoryTenantModelRegistry()
        logger.info("TenantModelRegistry initialized with %s backend", self.backend.name)

    def register_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        contact_email: str,
        tier: str = "standard",
    ) -> TenantMetadata:
        return self.backend.register_tenant(tenant_id, tenant_name, contact_email, tier)

    def get_tenant(self, tenant_id: str) -> Optional[TenantMetadata]:
        return self.backend.get_tenant(tenant_id)

    def count_tenants(self) -> int:
        return self.backend.count_tenants()

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
        return self.backend.register_model(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            storage_path=storage_path,
            config_path=config_path or storage_path,
            schema_definition=schema_definition,
            drift_thresholds=drift_thresholds,
            framework=framework,
        )

    def get_model(
        self, tenant_id: str, model_id: str, model_version: str
    ) -> Optional[ModelMetadata]:
        return self.backend.get_model(tenant_id, model_id, model_version)

    def get_latest_model(self, tenant_id: str, model_id: str) -> Optional[ModelMetadata]:
        return self.backend.get_latest_model(tenant_id, model_id)

    def list_tenant_models(self, tenant_id: str) -> Dict[str, ModelMetadata]:
        return self.backend.list_tenant_models(tenant_id)

    def list_all_tenants(self) -> Dict[str, TenantMetadata]:
        """Admin-level: return all registered tenants."""
        return self.backend.list_all_tenants()

    def list_all_models(self) -> list:
        """Admin-level: return all models across all tenants."""
        return self.backend.list_all_models()

    def count_models(self) -> int:
        return self.backend.count_models()

    def update_drift_thresholds(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        new_thresholds: Dict,
    ) -> Optional[ModelMetadata]:
        return self.backend.update_drift_thresholds(
            tenant_id, model_id, model_version, new_thresholds
        )

    def delete_model(self, tenant_id: str, model_id: str, model_version: str) -> bool:
        return self.backend.delete_model(tenant_id, model_id, model_version)

    def to_json(self) -> str:
        return self.backend.to_json()


class InMemoryTenantModelRegistry:
    name = "memory"

    def __init__(self):
        self.tenants: Dict[str, TenantMetadata] = {}
        self.registry: Dict[Tuple[str, str, str], ModelMetadata] = {}

    def register_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        contact_email: str,
        tier: str = "standard",
    ) -> TenantMetadata:
        tenant = TenantMetadata(tenant_id, tenant_name, contact_email, tier)
        self.tenants[tenant_id] = tenant
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[TenantMetadata]:
        return self.tenants.get(tenant_id)

    def count_tenants(self) -> int:
        return len(self.tenants)

    def register_model(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        storage_path: str,
        config_path: str,
        schema_definition: Dict,
        drift_thresholds: Dict,
        framework: str = "pytorch",
    ) -> ModelMetadata:
        key = (tenant_id, model_id, model_version)
        metadata = ModelMetadata(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            storage_path=storage_path,
            config_path=config_path,
            schema_definition=schema_definition,
            drift_thresholds=drift_thresholds,
            framework=framework,
        )
        self.registry[key] = metadata
        return metadata

    def get_model(
        self, tenant_id: str, model_id: str, model_version: str
    ) -> Optional[ModelMetadata]:
        return self.registry.get((tenant_id, model_id, model_version))

    def get_latest_model(self, tenant_id: str, model_id: str) -> Optional[ModelMetadata]:
        matching = [
            metadata
            for (tid, mid, _), metadata in self.registry.items()
            if tid == tenant_id and mid == model_id
        ]
        return matching[-1] if matching else None

    def list_tenant_models(self, tenant_id: str) -> Dict[str, ModelMetadata]:
        result = {}
        for (tid, mid, _), metadata in self.registry.items():
            if tid == tenant_id:
                result[mid] = metadata
        return result

    def list_all_tenants(self) -> Dict[str, TenantMetadata]:
        return dict(self.tenants)

    def list_all_models(self) -> list:
        return list(self.registry.values())

    def count_models(self) -> int:
        return len(self.registry)

    def update_drift_thresholds(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        new_thresholds: Dict,
    ) -> Optional[ModelMetadata]:
        metadata = self.get_model(tenant_id, model_id, model_version)
        if metadata is None:
            return None
        metadata.drift_thresholds.update(new_thresholds)
        return metadata

    def delete_model(self, tenant_id: str, model_id: str, model_version: str) -> bool:
        key = (tenant_id, model_id, model_version)
        if key not in self.registry:
            return False
        del self.registry[key]
        return True

    def to_json(self) -> str:
        data = {
            "tenants": {k: asdict(v) for k, v in self.tenants.items()},
            "models": {str(k): asdict(v) for k, v in self.registry.items()},
        }
        return json.dumps(data, indent=2)


class PostgresTenantModelRegistry:
    name = "postgres"

    def __init__(self, database_url: str):
        try:
            from sqlalchemy import (
                JSON,
                Column,
                DateTime,
                MetaData,
                String,
                Table,
                UniqueConstraint,
                create_engine,
                func,
                select,
            )
            from sqlalchemy.dialects.postgresql import UUID
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy and psycopg2-binary are required for PostgreSQL registry"
            ) from exc

        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()
        self.tenants = Table(
            "tenants",
            self.metadata,
            Column("tenant_id", String, primary_key=True),
            Column("tenant_name", String, nullable=False),
            Column("contact_email", String, nullable=False),
            Column("tier", String, nullable=False, default="standard"),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.tenant_models = Table(
            "tenant_models",
            self.metadata,
            Column("id", UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()),
            Column("tenant_id", String, nullable=False),
            Column("model_id", String, nullable=False),
            Column("model_version", String, nullable=False),
            Column("framework", String, nullable=False),
            Column("storage_path", String, nullable=False),
            Column("config_path", String, nullable=False),
            Column("schema_definition", JSON, nullable=False),
            Column("drift_thresholds", JSON, nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            UniqueConstraint("tenant_id", "model_id", "model_version"),
        )
        self.select = select
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            self.metadata.create_all(connection)

    def register_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        contact_email: str,
        tier: str = "standard",
    ) -> TenantMetadata:
        from sqlalchemy.dialects.postgresql import insert

        statement = insert(self.tenants).values(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            contact_email=contact_email,
            tier=tier,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["tenant_id"],
            set_={
                "tenant_name": statement.excluded.tenant_name,
                "contact_email": statement.excluded.contact_email,
                "tier": statement.excluded.tier,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)
        return TenantMetadata(tenant_id, tenant_name, contact_email, tier)

    def get_tenant(self, tenant_id: str) -> Optional[TenantMetadata]:
        query = self.select(self.tenants).where(self.tenants.c.tenant_id == tenant_id)
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return _tenant_from_row(row) if row else None

    def count_tenants(self) -> int:
        from sqlalchemy import func, select

        with self.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(self.tenants)).scalar_one())

    def register_model(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        storage_path: str,
        config_path: str,
        schema_definition: Dict,
        drift_thresholds: Dict,
        framework: str = "pytorch",
    ) -> ModelMetadata:
        from sqlalchemy.dialects.postgresql import insert

        if self.get_tenant(tenant_id) is None:
            self.register_tenant(tenant_id, tenant_id, f"{tenant_id}@example.local")

        statement = insert(self.tenant_models).values(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            framework=framework,
            storage_path=storage_path,
            config_path=config_path,
            schema_definition=schema_definition,
            drift_thresholds=drift_thresholds,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["tenant_id", "model_id", "model_version"],
            set_={
                "framework": statement.excluded.framework,
                "storage_path": statement.excluded.storage_path,
                "config_path": statement.excluded.config_path,
                "schema_definition": statement.excluded.schema_definition,
                "drift_thresholds": statement.excluded.drift_thresholds,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement)
        return ModelMetadata(
            tenant_id=tenant_id,
            model_id=model_id,
            model_version=model_version,
            storage_path=storage_path,
            config_path=config_path,
            schema_definition=schema_definition,
            drift_thresholds=drift_thresholds,
            framework=framework,
        )

    def get_model(
        self, tenant_id: str, model_id: str, model_version: str
    ) -> Optional[ModelMetadata]:
        query = self.select(self.tenant_models).where(
            self.tenant_models.c.tenant_id == tenant_id,
            self.tenant_models.c.model_id == model_id,
            self.tenant_models.c.model_version == model_version,
        )
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return _model_from_row(row) if row else None

    def get_latest_model(self, tenant_id: str, model_id: str) -> Optional[ModelMetadata]:
        query = (
            self.select(self.tenant_models)
            .where(
                self.tenant_models.c.tenant_id == tenant_id,
                self.tenant_models.c.model_id == model_id,
            )
            .order_by(self.tenant_models.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return _model_from_row(row) if row else None

    def list_tenant_models(self, tenant_id: str) -> Dict[str, ModelMetadata]:
        query = self.select(self.tenant_models).where(
            self.tenant_models.c.tenant_id == tenant_id
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return {row["model_id"]: _model_from_row(row) for row in rows}

    def list_all_tenants(self) -> Dict:
        query = self.select(self.tenants).order_by(self.tenants.c.created_at.desc())
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return {row["tenant_id"]: _tenant_from_row(row) for row in rows}

    def list_all_models(self) -> list:
        query = self.select(self.tenant_models).order_by(
            self.tenant_models.c.tenant_id,
            self.tenant_models.c.created_at.desc(),
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return [_model_from_row(row) for row in rows]

    def count_models(self) -> int:
        from sqlalchemy import func, select

        with self.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(self.tenant_models)).scalar_one())

    def update_drift_thresholds(
        self,
        tenant_id: str,
        model_id: str,
        model_version: str,
        new_thresholds: Dict,
    ) -> Optional[ModelMetadata]:
        from sqlalchemy import update

        metadata = self.get_model(tenant_id, model_id, model_version)
        if metadata is None:
            return None
        updated_thresholds = {**metadata.drift_thresholds, **new_thresholds}
        statement = (
            update(self.tenant_models)
            .where(
                self.tenant_models.c.tenant_id == tenant_id,
                self.tenant_models.c.model_id == model_id,
                self.tenant_models.c.model_version == model_version,
            )
            .values(drift_thresholds=updated_thresholds)
        )
        with self.engine.begin() as connection:
            connection.execute(statement)
        metadata.drift_thresholds = updated_thresholds
        return metadata

    def delete_model(self, tenant_id: str, model_id: str, model_version: str) -> bool:
        from sqlalchemy import delete

        statement = delete(self.tenant_models).where(
            self.tenant_models.c.tenant_id == tenant_id,
            self.tenant_models.c.model_id == model_id,
            self.tenant_models.c.model_version == model_version,
        )
        with self.engine.begin() as connection:
            result = connection.execute(statement)
        return result.rowcount > 0

    def to_json(self) -> str:
        with self.engine.connect() as connection:
            tenant_rows = connection.execute(self.select(self.tenants)).mappings().all()
            model_rows = connection.execute(self.select(self.tenant_models)).mappings().all()
        data = {
            "tenants": [dict(row) for row in tenant_rows],
            "models": [
                {
                    key: str(value) if key in {"id", "created_at"} else value
                    for key, value in dict(row).items()
                }
                for row in model_rows
            ],
        }
        return json.dumps(data, indent=2)


def _tenant_from_row(row) -> TenantMetadata:
    return TenantMetadata(
        tenant_id=row["tenant_id"],
        tenant_name=row["tenant_name"],
        contact_email=row["contact_email"],
        tier=row["tier"],
    )


def _model_from_row(row) -> ModelMetadata:
    return ModelMetadata(
        tenant_id=row["tenant_id"],
        model_id=row["model_id"],
        model_version=row["model_version"],
        storage_path=row["storage_path"],
        config_path=row["config_path"],
        schema_definition=row["schema_definition"],
        drift_thresholds=row["drift_thresholds"],
        framework=row["framework"],
    )
