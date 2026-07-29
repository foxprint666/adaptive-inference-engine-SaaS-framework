"""
quant_finance/muse_router.py

Async prediction gateway.

All model inference is fast (< 0.5 ms, pure Python, no IO) and runs
directly on the asyncio event loop. Background EWC retraining (~10 s)
is offloaded to a ThreadPoolExecutor so incoming ticks are never blocked.

Marketable limit orders
-----------------------
Signals are returned as limit orders placed just beyond the current
top-of-book (ask + $0.01 for buys, bid - $0.01 for sells) to cap
slippage during high-volatility windows while maintaining near-instant
fill rates. This avoids the ~80-150 ms market-order REST round-trip
being on the critical latency path.
"""

from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

from quant_finance.ingress_sim import compute_psi, compute_adversarial_auc


@dataclass
class Order:
    symbol:        str
    side:          str      # "buy" or "sell"
    qty:           int      # integer lots, vol-targeted, min 1
    limit_price:   float    # ask+0.01 (buy) or bid-0.01 (sell)
    signal_prob:   float    # calibrated model probability
    vol_ann:       float    # annualised vol at signal time
    signal_time_ns:int      # time.perf_counter_ns() at prediction (t2)


class MuseRouter:
    """
    Async prediction and drift-monitoring gateway.

    Every PSI_WINDOW ticks:
      - Computes PSI (first-window reference) and Adversarial AUC
      - Triggers ActiveModelCalibrator swap if drift detected
      - Schedules EWC background retrain (non-blocking)
    """

    PSI_WINDOW    = 200
    PSI_THRESHOLD = 0.25
    AUC_THRESHOLD = 0.72

    def __init__(
        self,
        calibrator,          # ActiveModelCalibrator
        vol_executor,        # VolatilityTargetedExecutor
        symbol: str = "SPY",
        retrain_fn: Optional[Callable] = None,
    ):
        self._cal         = calibrator
        self._vex         = vol_executor
        self._symbol      = symbol
        self._retrain_fn  = retrain_fn

        self._probs:     List[float]       = []
        self._feats:     List[List[float]] = []
        self._ref_probs: Optional[List[float]] = None

        self._drift_count:    int  = 0
        self._retrain_active: bool = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ewc")

    async def route(
        self,
        tick,
        feat:    List[float],
        vol_ann: float,
    ) -> Optional[Order]:
        """
        Synchronous model call wrapped in async signature.
        Returns Order or None (flat signal).
        """
        # 1. Calibrated probability  (< 0.5 ms, no IO -- safe on event loop)
        prob = self._cal.predict_calibrated_proba(feat)
        self._probs.append(prob)
        self._feats.append(feat)

        # 2. Drift check every PSI_WINDOW ticks
        if len(self._probs) % self.PSI_WINDOW == 0:
            await self._drift_check()

        # 3. Vol-targeted execution
        qty_f, signal = self._vex.compute_execution_parameters(vol_ann, prob)
        qty = max(1, int(math.floor(qty_f)))   # integer lots, min 1 share

        if signal == 0.0:
            return None  # flat

        # 4. Marketable limit price
        #    ask + $0.01 for buys : fills immediately but caps slippage
        #    bid - $0.01 for sells: fills immediately but caps slippage
        #    This removes the market-order REST round-trip from the latency
        #    critical path (pre-staged resting orders).
        if signal > 0:
            side        = "buy"
            limit_price = round(tick.ask + 0.01, 2)
        else:
            side        = "sell"
            limit_price = round(tick.bid - 0.01, 2)

        return Order(
            symbol=self._symbol,
            side=side,
            qty=qty,
            limit_price=limit_price,
            signal_prob=prob,
            vol_ann=vol_ann,
            signal_time_ns=time.perf_counter_ns(),  # t2 checkpoint
        )

    async def _drift_check(self) -> None:
        n      = self.PSI_WINDOW
        window = self._probs[-n:]

        # PSI: first-window reference (reset after each swap)
        if self._ref_probs is None:
            self._ref_probs = list(window)
            mean = sum(window) / len(window)
            print(f"  [MUSE] PSI reference stored (mean_prob={mean:.4f})")
            return

        psi = compute_psi(self._ref_probs, window)
        feats = self._feats[-n:]
        mid   = n // 2
        auc   = compute_adversarial_auc(feats[:mid], feats[mid:])

        drift = psi >= self.PSI_THRESHOLD or auc >= self.AUC_THRESHOLD
        print(f"  [MUSE] Drift: PSI={psi:.4f}  AUC={auc:.4f}  "
              f"{'DRIFT' if drift else 'stable'}")

        if drift:
            self._drift_count += 1
            self._ref_probs = None  # reset for new active model

            if self._retrain_fn and not self._retrain_active:
                X  = list(self._feats[-500:])
                print(f"  [MUSE] Scheduling background retrain ({len(X)} samples)")
                loop = asyncio.get_event_loop()
                self._retrain_active = True
                loop.run_in_executor(self._pool, self._bg_retrain, X)

    def _bg_retrain(self, X: list) -> None:
        """Runs in thread executor -- does NOT block the event loop."""
        try:
            if self._retrain_fn:
                self._retrain_fn(X)
        finally:
            self._retrain_active = False

    @property
    def drift_count(self) -> int:
        return self._drift_count

    def describe(self) -> dict:
        return {
            "drift_count":     self._drift_count,
            "ticks_routed":    len(self._probs),
            "retrain_active":  self._retrain_active,
            "psi_threshold":   self.PSI_THRESHOLD,
            "auc_threshold":   self.AUC_THRESHOLD,
        }
