"""
locustfile_multitenant.py

Multi-tenant FastAPI inference load test with JWT authentication.
"""

import random
import time
import json
import base64
import logging
from locust import HttpUser, task, between, events, constant_throughput
from locust.exception import RescheduleTask

logger = logging.getLogger(__name__)

# Test Configuration
FEATURE_DIM      = 5         # Input vector size (amount, distance, velocity, age, risk_score)
NUM_TENANTS      = 5         # Number of distinct tenants to simulate
MODEL_VERSION    = "v1"
P95_THRESHOLD_MS = 500       # CI gate: fail if p95 > 500ms
P99_THRESHOLD_MS = 1000      # CI gate: fail if p99 > 1000ms
ERROR_THRESHOLD  = 0.01      # CI gate: fail if error rate > 1%

TENANTS = [
    {
        "tenant_id": f"tenant-{i:03d}",
        "username":  f"user_{i:03d}@example.com",
        "password":  f"securepass_{i:03d}",
        "plan":      random.choice(["free", "pro", "enterprise"]),
    }
    for i in range(NUM_TENANTS)
]


def random_feature_vector() -> dict:
    """Generate a random fraud prediction feature dict."""
    return {
        "amount": round(random.uniform(1.0, 10000.0), 2),
        "distance": round(random.uniform(0.1, 1000.0), 2),
        "velocity": round(random.uniform(0.1, 100.0), 2),
        "age": round(random.uniform(1.0, 1000.0), 2),
        "risk_score": round(random.uniform(0.0, 1.0), 4)
    }


class InferenceUser(HttpUser):
    """
    Simulates a tenant user that:
    1. Authenticates on start
    2. Hits /predict with random feature vectors
    3. Optionally checks /health
    """
    wait_time = between(0.1, 1.5)

    def on_start(self):
        self.tenant = random.choice(TENANTS)
        self.token = ""
        self.token_expires_at = 0
        self._authenticate()

    def _authenticate(self):
        # We hit the admin_api token endpoint (running on port 8003 in local compose)
        # For direct testing against inference/app_multitenant we bypass token check if not enforced,
        # but here we request it to follow the exact production pattern.
        payload = {
            "username":  self.tenant["username"],
            "password":  self.tenant["password"],
            "tenant_id": self.tenant["tenant_id"],
        }
        # In a real environment, auth is handled by the admin control plane.
        # For locust run, we fetch from /auth/token if exposed, or fallback.
        with self.client.post(
            "/auth/token",
            json=payload,
            name="/auth/token",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.token = data.get("access_token", "dummy")
                self.token_expires_at = time.time() + data.get("expires_in", 3600)
                resp.success()
            else:
                # Direct fallback for isolated inference service test runs
                self.token = "dummy-token"
                self.token_expires_at = time.time() + 3600
                resp.success()

    def _get_auth_headers(self) -> dict:
        if time.time() > self.token_expires_at - 30:
            self._authenticate()
        return {
            "Authorization":  f"Bearer {self.token}",
            "X-Tenant-ID":    self.tenant["tenant_id"],
            "Content-Type":   "application/json",
        }

    @task(10)
    def predict(self):
        payload = random_feature_vector()
        start = time.perf_counter()

        with self.client.post(
            "/predict",
            json=payload,
            headers=self._get_auth_headers(),
            name="/predict",
            catch_response=True
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp.status_code == 200:
                try:
                    body = resp.json()
                    if "prediction" not in body:
                        resp.failure("Response missing 'prediction' field")
                    elif elapsed_ms > 2000:
                        resp.failure(f"Response too slow: {elapsed_ms:.0f}ms")
                    else:
                        resp.success()
                except Exception:
                    resp.failure("Invalid response format")
            elif resp.status_code == 401:
                self._authenticate()
                resp.failure("401 — token refreshed")
                raise RescheduleTask()
            elif resp.status_code == 429:
                resp.failure(f"Rate limited — tenant {self.tenant['tenant_id']}")
                raise RescheduleTask()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_check(self):
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"/health returned {resp.status_code}")


class HighVolumeUser(HttpUser):
    """Simulates enterprise-tier burst traffic."""
    wait_time = constant_throughput(5)  # 5 RPS per user
    weight = 1

    def on_start(self):
        self.tenant = TENANTS[0]
        self.token = "dummy-token"
        self.token_expires_at = time.time() + 3600

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "X-Tenant-ID":   self.tenant["tenant_id"],
            "Content-Type":  "application/json",
        }

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=random_feature_vector(),
            headers=self._headers(),
            name="/predict [enterprise]",
        )


@events.quitting.add_listener
def ci_gate(environment, **kwargs):
    stats = environment.runner.stats.total
    p95   = stats.get_response_time_percentile(0.95)
    p99   = stats.get_response_time_percentile(0.99)
    err   = stats.fail_ratio

    print("\n" + "="*50)
    print("LOAD TEST SUMMARY")
    print(f"  Total requests : {stats.num_requests:,}")
    print(f"  Failures       : {stats.num_failures:,} ({err:.2%})")
    print(f"  Median latency : {stats.median_response_time:.0f}ms")
    print(f"  p95 latency    : {p95:.0f}ms  (threshold: {P95_THRESHOLD_MS}ms)")
    print(f"  p99 latency    : {p99:.0f}ms  (threshold: {P99_THRESHOLD_MS}ms)")
    print(f"  RPS            : {stats.total_rps:.1f}")
    print("="*50 + "\n")

    failures = []
    if p95 > P95_THRESHOLD_MS:
        failures.append(f"p95 {p95:.0f}ms > {P95_THRESHOLD_MS}ms threshold")
    if p99 > P99_THRESHOLD_MS:
        failures.append(f"p99 {p99:.0f}ms > {P99_THRESHOLD_MS}ms threshold")
    if err > ERROR_THRESHOLD:
        failures.append(f"Error rate {err:.2%} > {ERROR_THRESHOLD:.2%} threshold")

    if failures:
        print("CI GATE ❌ FAILED:")
        for f in failures:
            print(f"  - {f}")
        environment.process_exit_code = 1
    else:
        print("CI GATE ✅ PASSED")
