"""
inference/model_lru_cache.py

Production async-safe LRU model cache for multi-tenant inference.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Wraps a loaded ModelRuntime with metadata for eviction decisions."""
    runtime: Any                          # ModelRuntime instance
    tenant_id: str
    model_id: str
    loaded_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    framework: str = "pytorch"            # "pytorch" | "sklearn" | etc.

    def touch(self) -> None:
        self.last_used_at = time.monotonic()


class AsyncLRUModelCache:
    """
    Async-safe, bounded LRU cache for ML model runtimes.
    """

    def __init__(self, maxsize: int = 10, min_free_ram_mb: int = 512):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._min_free_ram_mb = min_free_ram_mb

        # LRU store: key = f"{tenant_id}:{model_id}"
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Soft eviction: weak refs to recently evicted entries still in use
        self._soft_cache: weakref.WeakValueDictionary[str, CacheEntry] = (
            weakref.WeakValueDictionary()
        )

        # In-flight loads: prevent thundering herd for the same key
        self._inflight: Dict[str, asyncio.Future] = {}

        # Single lock protecting all mutations to _cache, _soft_cache, _inflight
        self._lock: asyncio.Lock = asyncio.Lock()

        logger.info("AsyncLRUModelCache init: maxsize=%d, min_free_ram_mb=%d",
                    maxsize, min_free_ram_mb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_load(
        self,
        tenant_id: str,
        model_id: str,
        loader_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Return the cached runtime for (tenant_id, model_id), loading it if
        necessary.
        """
        key = self._make_key(tenant_id, model_id)

        # ── Fast path ────────────────────────────────────────────────────
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._cache.move_to_end(key)
                return entry.runtime

            # Soft-evicted but still alive?
            entry = self._soft_cache.get(key)
            if entry is not None:
                logger.debug("Cache warm-promote (soft→hard): %s", key)
                entry.touch()
                self._cache[key] = entry
                self._cache.move_to_end(key)
                await self._maybe_evict_lru()
                return entry.runtime

            # Already loading? Join the in-flight future.
            if key in self._inflight:
                fut = self._inflight[key]
            else:
                # We are the designated loader.
                fut = asyncio.get_event_loop().create_future()
                self._inflight[key] = fut
                fut = None  # signal: we own the load

        # ── Slow path: we own the load ───────────────────────────────────
        if fut is None:
            return await self._do_load(key, tenant_id, model_id, loader_fn)

        # ── Wait path: join existing load ───────────────────────────────
        logger.debug("Cache coalescing load for %s", key)
        return await fut

    async def evict(self, tenant_id: str, model_id: str) -> bool:
        """
        Manually evict a model (e.g., on Redis model_reload pub/sub event).
        Returns True if the key was present and evicted.
        """
        key = self._make_key(tenant_id, model_id)
        async with self._lock:
            entry = self._cache.pop(key, None)
            self._soft_cache.pop(key, None)   # type: ignore[attr-defined]
            self._inflight.pop(key, None)

        if entry is not None:
            logger.info("Manual evict: %s", key)
            await asyncio.get_event_loop().run_in_executor(
                None, self._cleanup_entry, entry
            )
            return True
        return False

    def peek(self, tenant_id: str, model_id: str) -> Optional[CacheEntry]:
        """Read entry without promoting LRU order. Non-async, safe for admin."""
        key = self._make_key(tenant_id, model_id)
        return self._cache.get(key) or self._soft_cache.get(key)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def keys(self):
        return list(self._cache.keys())

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded": len(self._cache),
            "maxsize": self._maxsize,
            "soft_evicted": len(self._soft_cache),
            "inflight": len(self._inflight),
            "keys": list(self._cache.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(tenant_id: str, model_id: str) -> str:
        return f"{tenant_id}:{model_id}"

    async def _do_load(
        self,
        key: str,
        tenant_id: str,
        model_id: str,
        loader_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Execute the load, write result to cache, resolve the Future for
        any coalesced waiters, and clean up inflight tracking.
        """
        # Circuit-breaker: check system RAM before loading a new model
        if self._min_free_ram_mb > 0:
            self._check_ram(key)

        try:
            logger.info("Cache MISS — loading model: %s", key)
            runtime = await loader_fn()
            entry = CacheEntry(
                runtime=runtime,
                tenant_id=tenant_id,
                model_id=model_id,
            )
        except Exception as exc:
            # Resolve waiters with the exception, then clean up
            async with self._lock:
                fut = self._inflight.pop(key, None)
            if fut and not fut.done():
                fut.set_exception(exc)
            logger.error("Model load failed for %s: %s", key, exc)
            raise

        # Write to hard cache, evict LRU if needed
        evicted_entry: Optional[CacheEntry] = None
        async with self._lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            evicted_entry = await self._maybe_evict_lru()

            # Resolve all coalesced waiters
            fut = self._inflight.pop(key, None)

        if fut and not fut.done():
            fut.set_result(runtime)

        # Cleanup evicted model outside the lock (can be slow)
        if evicted_entry is not None:
            await asyncio.get_event_loop().run_in_executor(
                None, self._cleanup_entry, evicted_entry
            )

        return runtime

    async def _maybe_evict_lru(self) -> Optional[CacheEntry]:
        """
        Evict the LRU entry if cache is over capacity.
        MUST be called with self._lock held.
        """
        if len(self._cache) <= self._maxsize:
            return None

        lru_key, lru_entry = self._cache.popitem(last=False)  # oldest = LRU

        # Soft eviction: keep weak ref so in-flight requests can finish
        self._soft_cache[lru_key] = lru_entry

        logger.info(
            "LRU evict: %s (loaded %.0fs ago, last used %.0fs ago)",
            lru_key,
            time.monotonic() - lru_entry.loaded_at,
            time.monotonic() - lru_entry.last_used_at,
        )
        return lru_entry

    @staticmethod
    def _cleanup_entry(entry: CacheEntry) -> None:
        """
        Synchronous cleanup of a model runtime. Runs in thread pool executor
        so it doesn't block the event loop.
        """
        key = f"{entry.tenant_id}:{entry.model_id}"
        try:
            runtime = entry.runtime
            # Step 1: CPU offload (releases CUDA memory before del if applicable)
            if hasattr(runtime, "model_instance") and runtime.model_instance is not None:
                try:
                    if hasattr(runtime.model_instance, "cpu"):
                        runtime.model_instance.cpu()
                        logger.debug("Moved %s model to CPU before eviction", key)
                except Exception as e:
                    logger.warning("cpu() failed during eviction of %s: %s", key, e)

            # Step 2: delete references
            if hasattr(runtime, "model_instance"):
                runtime.model_instance = None
            del runtime
            del entry

        except Exception as e:
            logger.warning("Eviction cleanup error for %s: %s", key, e)
        finally:
            # Step 3: Python GC sweep
            gc.collect()

            # Step 4: release CUDA allocator cache (expensive — only on real eviction)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared after eviction of %s", key)
            except ImportError:
                pass
            except Exception as e:
                logger.warning("cuda.empty_cache() failed: %s", e)

    def _check_ram(self, key: str) -> None:
        """
        Circuit-breaker: raise RuntimeError if available system RAM is below
        min_free_ram_mb. Prevents OOM kills when loading new models.
        """
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            if available_mb < self._min_free_ram_mb:
                raise RuntimeError(
                    f"Refusing to load model {key}: only {available_mb:.0f} MB RAM "
                    f"available, threshold is {self._min_free_ram_mb} MB. "
                    f"Current cache: {list(self._cache.keys())}"
                )
        except ImportError:
            pass  # psutil not installed — skip check


# Module-level singleton
_model_cache: Optional[AsyncLRUModelCache] = None


def get_model_cache(maxsize: int = 10, min_free_ram_mb: int = 512) -> AsyncLRUModelCache:
    """Lazy singleton accessor for the module-level model cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = AsyncLRUModelCache(maxsize=maxsize, min_free_ram_mb=min_free_ram_mb)
    return _model_cache
