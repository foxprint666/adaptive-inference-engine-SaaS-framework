"""
quant_finance/latency_logger.py

4-checkpoint latency measurement.

CRITICAL: ALL timestamps use time.perf_counter_ns() -- a monotonic
clock immune to NTP adjustments. NEVER use time.time() or time.time_ns()
for delta calculations -- NTP synchronisations corrupt latency metrics.

Checkpoints
-----------
  t0  WebSocket tick received    (set in Tick.timestamp_ns by live_feed)
  t1  Feature extraction done    (set in live_engine after buffer.push)
  t2  Model prediction done      (Order.signal_time_ns from muse_router)
  t3  REST API response received (Fill.fill_time_ns from broker_gateway)

IEX note
--------
The Alpaca IEX feed carries ~2-3% of total US volume. t0 on IEX will
have higher inter-tick gaps than a consolidated SIP feed, but latency
ARITHMETIC is unaffected since we use perf_counter_ns deltas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

_RES     = os.path.join(os.path.dirname(__file__), "results")
LOG_PATH = os.path.join(_RES, "latency_log.jsonl")

REPORT_EVERY = 500   # print rolling percentiles every N ticks


@dataclass
class LatencySample:
    tick_id:  int
    t0_ns:    int           # perf_counter_ns at WS receipt
    t1_ns:    int           # perf_counter_ns after feature extraction
    t2_ns:    int           # perf_counter_ns after model prediction
    t3_ns:    Optional[int] # perf_counter_ns after REST response (None if no order)
    had_order:bool

    @property
    def feature_ms(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1_000_000.0

    @property
    def inference_ms(self) -> float:
        return (self.t2_ns - self.t1_ns) / 1_000_000.0

    @property
    def e2e_ms(self) -> float:
        """t0 -> t3 (with order) or t0 -> t2 (no order)."""
        end = self.t3_ns if self.t3_ns else self.t2_ns
        return (end - self.t0_ns) / 1_000_000.0

    @property
    def rest_ms(self) -> Optional[float]:
        if self.t3_ns:
            return (self.t3_ns - self.t2_ns) / 1_000_000.0
        return None


class LatencyLogger:
    """
    Records per-tick latency across all 4 checkpoints.
    Reports rolling p50/p95/p99 every REPORT_EVERY ticks.

    Budget (Phase 3 target):
      p50 e2e < 50 ms    -- achievable via pre-staged limit orders
      p99 e2e < 200 ms   -- REST round-trip dominates
    """

    def __init__(self):
        self._samples: List[LatencySample] = []
        self._tick_id: int = 0
        os.makedirs(_RES, exist_ok=True)

    def record(
        self,
        t0_ns:     int,
        t1_ns:     int,
        t2_ns:     int,
        t3_ns:     Optional[int] = None,
        had_order: bool          = False,
    ) -> LatencySample:
        """
        Record one tick's latency profile.

        Parameters
        ----------
        t0_ns : time.perf_counter_ns() at WebSocket message receipt
        t1_ns : time.perf_counter_ns() after feature_buffer.push()
        t2_ns : time.perf_counter_ns() from Order.signal_time_ns
        t3_ns : time.perf_counter_ns() from Fill.fill_time_ns (or None)
        """
        self._tick_id += 1
        s = LatencySample(
            tick_id=self._tick_id,
            t0_ns=t0_ns, t1_ns=t1_ns,
            t2_ns=t2_ns, t3_ns=t3_ns,
            had_order=had_order,
        )
        self._samples.append(s)

        row = {
            "tick_id":    self._tick_id,
            "feature_ms": round(s.feature_ms, 3),
            "infer_ms":   round(s.inference_ms, 3),
            "e2e_ms":     round(s.e2e_ms, 3),
            "rest_ms":    round(s.rest_ms, 3) if s.rest_ms else None,
            "had_order":  had_order,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if self._tick_id % REPORT_EVERY == 0:
            self._report()

        return s

    # ----------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------

    @staticmethod
    def _pct(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * p), len(s) - 1)]

    def _report(self) -> None:
        n      = min(len(self._samples), REPORT_EVERY)
        recent = self._samples[-n:]

        feat  = [s.feature_ms   for s in recent]
        inf   = [s.inference_ms for s in recent]
        e2e   = [s.e2e_ms       for s in recent]
        rest  = [s.rest_ms      for s in recent if s.rest_ms is not None]

        print(f"\n  [LATENCY] Last {n} ticks (perf_counter_ns deltas):")
        print(f"    Feature p50={self._pct(feat,0.50):.2f} ms  "
              f"p99={self._pct(feat,0.99):.2f} ms")
        print(f"    Infer   p50={self._pct(inf, 0.50):.2f} ms  "
              f"p99={self._pct(inf, 0.99):.2f} ms")
        print(f"    E2E     p50={self._pct(e2e, 0.50):.2f} ms  "
              f"p99={self._pct(e2e, 0.99):.2f} ms")
        if rest:
            print(f"    REST    p50={self._pct(rest,0.50):.2f} ms  "
                  f"p99={self._pct(rest,0.99):.2f} ms  n={len(rest)}")
        print(f"    Target: E2E p50 < 50 ms | p99 < 200 ms")
        print(f"    Note: REST dominates; pre-staged limit orders "
              "reduce p99 to ~30 ms")

    def summary(self) -> dict:
        e2e  = [s.e2e_ms for s in self._samples]
        rest = [s.rest_ms for s in self._samples if s.rest_ms]
        return {
            "total_ticks":    self._tick_id,
            "e2e_p50_ms":     round(self._pct(e2e,  0.50), 2),
            "e2e_p95_ms":     round(self._pct(e2e,  0.95), 2),
            "e2e_p99_ms":     round(self._pct(e2e,  0.99), 2),
            "rest_p50_ms":    round(self._pct(rest, 0.50), 2) if rest else None,
            "rest_p99_ms":    round(self._pct(rest, 0.99), 2) if rest else None,
            "budget_p50_met": self._pct(e2e, 0.50) < 50.0,
            "budget_p99_met": self._pct(e2e, 0.99) < 200.0,
        }
