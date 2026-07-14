"""
inference/storage_backend.py

Pluggable model storage backend with local disk caching for S3.

Design
------
Two implementations:
  LocalStorageBackend  — reads/writes local filesystem (default for dev/test)
  S3StorageBackend     — reads/writes AWS S3 with a /tmp/cache warm-up layer

The S3 backend solves the cold-start spike problem:
  1. When a model_reload event fires, the new weights are downloaded from S3
     to /tmp/mlops_cache/<tenant_id>/<model_id>.pt using an atomic rename
     (write to .tmp → os.replace → final path).  Only the final rename is
     instantaneous, so no inference thread ever reads a half-written file.
  2. On every load_model_bytes() call, the local cache is checked first.
     A cache HIT avoids the S3 round-trip entirely.
  3. If the cache directory is on a Kubernetes emptyDir volume (recommended),
     it survives pod restarts within the same scheduling cycle and is shared
     across containers on the same node.

Environment variables
---------------------
  STORAGE_BACKEND        local | s3        (default: local)
  S3_BUCKET_NAME         name of the S3 bucket
  S3_REGION_NAME         AWS region        (default: us-east-1)
  S3_ENDPOINT_URL        override for MinIO / localstack
  MODEL_CACHE_DIR        local cache root  (default: /tmp/mlops_cache)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Recommended: mount a Kubernetes emptyDir volume at /model-cache and set
# MODEL_CACHE_DIR=/model-cache.  Using /tmp risks Node DiskPressure evictions
# because it shares the node's root disk.  emptyDir with sizeLimit is isolated
# and tracked by the kubelet separately from the container's ephemeral-storage.
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/mlops_cache")
_cache_lock = threading.Lock()  # Prevents concurrent downloads of the same key


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """
    Abstract storage backend for model weights.

    All paths passed to load/save are logical keys (e.g.
    ``"{tenant_id}/{model_id}/v1.0.0.pt"``). The backend maps them to
    physical locations (local filesystem or S3 object keys).
    """

    @abstractmethod
    def load_model_bytes(self, key: str) -> bytes:
        """Download model bytes for the given logical key."""

    @abstractmethod
    def save_model_bytes(self, key: str, data: bytes) -> None:
        """Persist model bytes under the given logical key."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if the key already exists in storage."""

    def warm_cache(self, key: str) -> str:
        """
        Pre-download the model to the local cache directory.

        Returns the local file path once the download is 100% complete.
        This is a no-op for LocalStorageBackend (the file IS the cache).

        The download uses an atomic rename pattern:
          1. Write to  <cache_dir>/<key>.tmp
          2. os.replace(<tmp>, <final>)   ← atomic on POSIX
        Inference threads never read a half-written file because the
        final path only appears after the rename succeeds.
        """
        return key  # default: key IS the local path (LocalStorageBackend)


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalStorageBackend(StorageBackend):
    """
    Simple backend that reads/writes local files directly.

    ``key`` is treated as an absolute or relative file path.
    This is the default backend and is used in dev, unit tests, and
    Docker Compose environments without S3.
    """

    def load_model_bytes(self, key: str) -> bytes:
        with open(key, "rb") as f:
            return f.read()

    def save_model_bytes(self, key: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(key)), exist_ok=True)
        # Atomic write: write to a sibling .tmp, then rename
        tmp_path = key + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, key)

    def exists(self, key: str) -> bool:
        return os.path.isfile(key)

    def warm_cache(self, key: str) -> str:
        """For local backend, the key is already the final path."""
        return key


# ---------------------------------------------------------------------------
# S3 backend with local disk cache
# ---------------------------------------------------------------------------


class S3StorageBackend(StorageBackend):
    """
    S3 backend with a local-disk warm-up cache to eliminate cold-start spikes.

    Cache strategy
    --------------
    * Key → local path: ``<MODEL_CACHE_DIR>/<sha256(key)[:16]>_<basename(key)>``
      Using a hash prefix prevents path collisions across tenants while keeping
      filenames human-readable.
    * Atomic download: stream from S3 to ``<local_path>.tmp`` using
      ``TransferConfig(multipart_threshold=…)`` then ``os.replace`` to the final
      path.  A threading.Lock per key prevents concurrent duplicate downloads.
    * ETag-based invalidation: each cache entry stores the S3 ETag in a
      companion ``.etag`` sidecar file. On cache HIT, we compare the stored ETag
      against a lightweight ``head_object`` call.  If they match, no download
      occurs.  This is more reliable than Pub/Sub alone and handles cases where
      the worker and inference pods are briefly out-of-sync.

    Why ETag over TTL?
    ------------------
    A fixed TTL (e.g. 5 minutes) would either serve stale weights for too long or
    cause unnecessary S3 requests.  S3 ETags change only when the object changes,
    so the comparison is deterministic and free (HEAD request = 1 ms vs GET = N ms
    for a 100 MB model file).
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3StorageBackend. "
                "Install with: pip install boto3"
            ) from exc

        self.bucket_name: str = bucket_name or os.environ["S3_BUCKET_NAME"]
        self.region_name: str = region_name or os.getenv("S3_REGION_NAME", "us-east-1")
        self.endpoint_url: Optional[str] = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        self.cache_dir: str = cache_dir or MODEL_CACHE_DIR

        self._s3 = boto3.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=10,
            ),
        )
        # Per-key download locks to prevent thundering-herd duplicate downloads
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_mutex = threading.Lock()

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            "S3StorageBackend initialised — bucket=%s  cache=%s",
            self.bucket_name,
            self.cache_dir,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _local_cache_path(self, key: str) -> str:
        """Compute the local cache file path for a given S3 key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        basename = Path(key).name
        return os.path.join(self.cache_dir, f"{key_hash}_{basename}")

    def _etag_sidecar_path(self, local_path: str) -> str:
        return local_path + ".etag"

    def _read_cached_etag(self, local_path: str) -> Optional[str]:
        etag_file = self._etag_sidecar_path(local_path)
        try:
            return Path(etag_file).read_text().strip()
        except FileNotFoundError:
            return None

    def _write_cached_etag(self, local_path: str, etag: str) -> None:
        Path(self._etag_sidecar_path(local_path)).write_text(etag)

    def _get_s3_etag(self, key: str) -> Optional[str]:
        """Fast HEAD request to get current S3 ETag without downloading."""
        try:
            resp = self._s3.head_object(Bucket=self.bucket_name, Key=key)
            return resp.get("ETag", "").strip('"')
        except Exception as exc:
            logger.warning("head_object failed for key=%s: %s", key, exc)
            return None

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._key_locks_mutex:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def _download_to_cache(self, key: str, local_path: str, s3_etag: Optional[str]) -> None:
        """
        Stream S3 object → local_path using a fully atomic download sequence.

        Research-backed implementation:
        1. ``tempfile.mkstemp(dir=dest_dir)`` — creates the temp file in the
           SAME directory as the final path, guaranteeing they are on the same
           filesystem mount.  This makes ``os.replace()`` truly atomic (a rename
           syscall), not a cross-device copy+delete.
        2. ``os.fsync()`` before rename — flushes OS write buffers to physical
           storage, preventing a partially-written file from being swapped in
           after a power/host failure.
        3. Cleanup on error — the temp file is removed if the download fails so
           it doesn't accumulate as disk waste.
        4. Managed multipart transfer — boto3 TransferConfig parallelises large
           downloads without buffering the entire file in RAM.
           RAM usage ≈ max_concurrency × multipart_chunksize = 4 × 16 MB = 64 MB.
        """
        from boto3.s3.transfer import TransferConfig

        dest_dir = os.path.dirname(local_path) or "."
        os.makedirs(dest_dir, exist_ok=True)

        transfer_cfg = TransferConfig(
            multipart_threshold=25 * 1024 * 1024,   # 25 MB
            multipart_chunksize=16 * 1024 * 1024,   # 16 MB chunks
            max_concurrency=4,
            use_threads=True,
        )

        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".download.tmp")
        logger.info("Downloading s3://%s/%s → %s", self.bucket_name, key, tmp_path)
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                self._s3.download_fileobj(
                    Bucket=self.bucket_name,
                    Key=key,
                    Fileobj=fh,
                    Config=transfer_cfg,
                )
                fh.flush()
                os.fsync(fh.fileno())   # ← flush OS buffers to disk before rename

            os.replace(tmp_path, local_path)   # ← atomic rename (POSIX + Windows 10+)
        except Exception:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

        if s3_etag:
            self._write_cached_etag(local_path, s3_etag)
        logger.info("Cache warmed: %s", local_path)

    def _is_cache_valid(self, key: str, local_path: str) -> bool:
        """
        Return True if the local cache is fresh.

        1. File must exist on disk.
        2. Stored ETag must match the current S3 ETag (HEAD request).
        """
        if not os.path.isfile(local_path):
            return False

        cached_etag = self._read_cached_etag(local_path)
        if cached_etag is None:
            # No ETag sidecar → treat as stale
            return False

        current_etag = self._get_s3_etag(key)
        if current_etag is None:
            # Can't reach S3 — serve from cache rather than crashing
            logger.warning("S3 unreachable; using stale cache for key=%s", key)
            return True

        return cached_etag == current_etag

    # ── Public interface ────────────────────────────────────────────────────

    def warm_cache(self, key: str) -> str:
        """
        Ensure the model is cached locally. Returns the local file path.

        This is called:
          • By the pub/sub reload handler (background, non-blocking for inference)
          • On first inference request if no cache entry exists (blocking)

        Thread-safe: a per-key lock prevents duplicate concurrent downloads.
        """
        local_path = self._local_cache_path(key)
        lock = self._get_key_lock(key)

        with lock:
            if self._is_cache_valid(key, local_path):
                logger.debug("Cache HIT for key=%s → %s", key, local_path)
                return local_path

            s3_etag = self._get_s3_etag(key)
            self._download_to_cache(key, local_path, s3_etag)

        return local_path

    def load_model_bytes(self, key: str) -> bytes:
        """
        Load model bytes. Always goes through the local cache layer.

        Cache HIT  → read from /tmp/mlops_cache (microseconds)
        Cache MISS → download from S3, populate cache, then read (seconds)
        """
        local_path = self.warm_cache(key)
        with open(local_path, "rb") as f:
            return f.read()

    def save_model_bytes(self, key: str, data: bytes) -> None:
        """
        Upload model bytes to S3 and update the local cache atomically.

        Flow:
          1. Write bytes to a local temp file (atomic rename pattern)
          2. Upload to S3 via managed transfer
          3. Update the ETag sidecar with the new S3 ETag
        """
        local_path = self._local_cache_path(key)
        lock = self._get_key_lock(key)

        with lock:
            # 1. Write to local cache atomically
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            tmp_path = local_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, local_path)

            # 2. Upload to S3
            logger.info("Uploading %d bytes to s3://%s/%s", len(data), self.bucket_name, key)
            self._s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
            )

            # 3. Refresh ETag sidecar
            new_etag = self._get_s3_etag(key)
            if new_etag:
                self._write_cached_etag(local_path, new_etag)

    def exists(self, key: str) -> bool:
        """Check if the S3 object exists (or the local cache is still valid)."""
        local_path = self._local_cache_path(key)
        if self._is_cache_valid(key, local_path):
            return True
        return self._get_s3_etag(key) is not None

    def evict_cache(self, key: str) -> None:
        """Remove the local cache entry for a key (e.g. after model deletion)."""
        local_path = self._local_cache_path(key)
        for path in [local_path, self._etag_sidecar_path(local_path)]:
            try:
                os.remove(path)
                logger.info("Evicted cache: %s", path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_storage_backend() -> StorageBackend:
    """
    Return the configured storage backend instance.

    Set ``STORAGE_BACKEND=s3`` to use S3StorageBackend.
    Defaults to LocalStorageBackend for dev/test environments.
    """
    backend_type = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend_type == "s3":
        return S3StorageBackend()
    return LocalStorageBackend()


# Module-level singleton — import this everywhere
storage_backend: StorageBackend = get_storage_backend()
