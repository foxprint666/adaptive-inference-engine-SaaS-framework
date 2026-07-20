"""
admin_api/user_auth.py

Real user authentication for the dashboard.

Separate from the ML tenant JWT auth (admin_api/auth.py) which is for
machine-to-machine calls. This module handles:
  - Dashboard user accounts (email + bcrypt password)
  - Role-based access: admin (all tenants) vs viewer (own tenant only)
  - Session tokens (short-lived JWTs with role + tenant_id embedded)
  - In-memory user store as fallback when DATABASE_URL is not set

Production note: set DATABASE_URL and run database/users.sql to migrate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "8"))

try:
    from jose import JWTError, jwt
except ImportError:
    raise ImportError("python-jose is required. pip install python-jose[cryptography]")

try:
    import bcrypt as _bcrypt_lib
    _HAS_BCRYPT = True
except ImportError:
    _HAS_BCRYPT = False
    logger.warning("bcrypt not installed — password hashing falls back to SHA256 (dev only)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DashboardUser:
    id: str
    email: str
    password_hash: str
    display_name: str
    role: str          # "admin" | "viewer"
    tenant_id: Optional[str]  # None for admins
    is_active: bool = True


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(plaintext: str) -> str:
    if _HAS_BCRYPT:
        return _bcrypt_lib.hashpw(plaintext.encode(), _bcrypt_lib.gensalt()).decode()
    import hashlib
    return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest()


def verify_password(plaintext: str, hashed: str) -> bool:
    if hashed.startswith("sha256:"):
        import hashlib
        return "sha256:" + hashlib.sha256(plaintext.encode()).hexdigest() == hashed
    if _HAS_BCRYPT:
        return _bcrypt_lib.checkpw(plaintext.encode(), hashed.encode())
    return False


# ---------------------------------------------------------------------------
# Session JWT
# ---------------------------------------------------------------------------

def create_session_token(user: DashboardUser) -> str:
    """Issue a signed JWT embedding user id, role, and tenant scope."""
    expire = datetime.now(timezone.utc) + timedelta(hours=SESSION_EXPIRY_HOURS)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_session_token(token: str) -> dict:
    """Decode and validate a session JWT. Raises JWTError on failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ---------------------------------------------------------------------------
# User store abstraction
# ---------------------------------------------------------------------------

class InMemoryUserStore:
    """In-process store for development and testing."""

    def __init__(self):
        self._users: Dict[str, DashboardUser] = {}
        self._by_email: Dict[str, str] = {}
        # Seed a default admin account
        self._seed_admin()

    def _seed_admin(self):
        pw = os.getenv("ADMIN_PASSWORD", "admin123")
        admin = DashboardUser(
            id=str(uuid4()),
            email="admin@example.com",
            password_hash=hash_password(pw),
            display_name="Admin",
            role="admin",
            tenant_id=None,
        )
        self._users[admin.id] = admin
        self._by_email[admin.email.lower()] = admin.id
        logger.info("InMemoryUserStore: seeded admin@example.com (pwd from ADMIN_PASSWORD env)")

    def get_by_email(self, email: str) -> Optional[DashboardUser]:
        uid = self._by_email.get(email.lower())
        return self._users.get(uid) if uid else None

    def get_by_id(self, user_id: str) -> Optional[DashboardUser]:
        return self._users.get(user_id)

    def create_user(
        self,
        email: str,
        password: str,
        display_name: str,
        role: str,
        tenant_id: Optional[str],
    ) -> DashboardUser:
        if email.lower() in self._by_email:
            raise ValueError(f"User {email} already exists")
        user = DashboardUser(
            id=str(uuid4()),
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
            role=role,
            tenant_id=tenant_id,
        )
        self._users[user.id] = user
        self._by_email[email.lower()] = user.id
        return user

    def list_users(self) -> List[DashboardUser]:
        return list(self._users.values())


class PostgresUserStore:
    """Production user store backed by dashboard_users table."""

    def __init__(self, database_url: str):
        try:
            from sqlalchemy import (
                Boolean, Column, DateTime, MetaData,
                String, Table, create_engine, func, select,
            )
            from sqlalchemy.dialects.postgresql import UUID as PGUUID, CITEXT
        except ImportError as exc:
            raise ImportError("sqlalchemy + psycopg2-binary required") from exc

        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.meta = MetaData()
        self.users_table = Table(
            "dashboard_users",
            self.meta,
            Column("id", PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()),
            Column("email", String, nullable=False, unique=True),
            Column("password_hash", String, nullable=False),
            Column("display_name", String, nullable=False, default=""),
            Column("role", String, nullable=False, default="viewer"),
            Column("tenant_id", String, nullable=True),
            Column("is_active", Boolean, nullable=False, default=True),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column("last_login_at", DateTime(timezone=True), nullable=True),
        )
        self.select = select
        self._ensure_schema()

    def _ensure_schema(self):
        with self.engine.begin() as conn:
            conn.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
            self.meta.create_all(conn)

    def _row_to_user(self, row) -> DashboardUser:
        return DashboardUser(
            id=str(row["id"]),
            email=row["email"],
            password_hash=row["password_hash"],
            display_name=row["display_name"] or "",
            role=row["role"],
            tenant_id=row["tenant_id"],
            is_active=bool(row["is_active"]),
        )

    def get_by_email(self, email: str) -> Optional[DashboardUser]:
        q = self.select(self.users_table).where(
            self.users_table.c.email == email.lower()
        )
        with self.engine.connect() as conn:
            row = conn.execute(q).mappings().first()
        return self._row_to_user(row) if row else None

    def get_by_id(self, user_id: str) -> Optional[DashboardUser]:
        import uuid
        q = self.select(self.users_table).where(
            self.users_table.c.id == uuid.UUID(user_id)
        )
        with self.engine.connect() as conn:
            row = conn.execute(q).mappings().first()
        return self._row_to_user(row) if row else None

    def create_user(
        self,
        email: str,
        password: str,
        display_name: str,
        role: str,
        tenant_id: Optional[str],
    ) -> DashboardUser:
        from sqlalchemy.dialects.postgresql import insert
        stmt = insert(self.users_table).values(
            email=email.lower(),
            password_hash=hash_password(password),
            display_name=display_name,
            role=role,
            tenant_id=tenant_id,
            is_active=True,
        ).returning(self.users_table)
        with self.engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_user(row)

    def list_users(self) -> List[DashboardUser]:
        q = self.select(self.users_table).order_by(self.users_table.c.created_at.desc())
        with self.engine.connect() as conn:
            rows = conn.execute(q).mappings().all()
        return [self._row_to_user(r) for r in rows]

    def record_login(self, user_id: str) -> None:
        from sqlalchemy import update
        with self.engine.begin() as conn:
            conn.execute(
                update(self.users_table)
                .where(self.users_table.c.id == user_id)
                .values(last_login_at=func.now())
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def get_user_store():
    """Return the appropriate user store based on DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return PostgresUserStore(database_url)
    return InMemoryUserStore()


_store_instance = None


def user_store():
    global _store_instance
    if _store_instance is None:
        _store_instance = get_user_store()
    return _store_instance
