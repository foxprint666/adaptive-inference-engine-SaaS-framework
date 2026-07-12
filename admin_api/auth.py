"""
admin_api/auth.py - Authentication and authorization for Phase 3
"""

import os
import logging
from typing import Optional
from fastapi import HTTPException, status
from datetime import datetime, timedelta

try:
    from jose import JWTError, jwt
except ImportError:
    raise ImportError("python-jose is required for auth. Install with: pip install python-jose[cryptography]")

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_MINUTES = 60


def create_access_token(tenant_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token for a tenant."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    
    to_encode = {"tenant_id": tenant_id, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> str:
    """Verify JWT token and return tenant_id."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        tenant_id: str = payload.get("tenant_id")
        if tenant_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return tenant_id
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def verify_bearer_token(authorization: str) -> str:
    """Extract and verify bearer token from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    return verify_token(token)
