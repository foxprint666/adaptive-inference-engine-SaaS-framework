"""
admin_api/rate_limiter.py - Token bucket rate limiting for Phase 3
"""

import time
import logging
from typing import Dict
from threading import Lock

logger = logging.getLogger(__name__)


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens to add per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = Lock()
    
    def allow(self, tokens: int = 1) -> bool:
        """Check if request is allowed and consume tokens if so."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Replenish tokens
            self.tokens = min(
                float(self.capacity),
                self.tokens + (elapsed * self.refill_rate)
            )
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimiter:
    """Per-tenant rate limiter manager."""
    
    def __init__(self, default_capacity: int = 60, default_refill: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            default_capacity: Default token bucket capacity
            default_refill: Default refill rate (tokens/second)
        """
        self.default_capacity = default_capacity
        self.default_refill = default_refill
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()
    
    def get_bucket(self, tenant_id: str) -> TokenBucket:
        """Get or create token bucket for tenant."""
        if tenant_id not in self.buckets:
            with self.lock:
                if tenant_id not in self.buckets:
                    self.buckets[tenant_id] = TokenBucket(
                        self.default_capacity,
                        self.default_refill
                    )
        return self.buckets[tenant_id]
    
    def is_allowed(self, tenant_id: str, tokens: int = 1) -> bool:
        """Check if request is rate-limited."""
        bucket = self.get_bucket(tenant_id)
        return bucket.allow(tokens)
    
    def set_limits(self, tenant_id: str, capacity: int, refill_rate: float):
        """Set custom limits for a tenant."""
        with self.lock:
            self.buckets[tenant_id] = TokenBucket(capacity, refill_rate)
