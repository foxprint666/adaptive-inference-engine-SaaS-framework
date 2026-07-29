"""
quant_finance/circuit_breaker.py

Hard circuit breaker -- operates OUTSIDE all ML logic.
No model calls. No calibration. Pure risk arithmetic.

Six Guards
----------
1. Daily Drawdown Limit   -- hard kill if equity drops > 2.0% in 24h
2. Max Order Quantity     -- rejects any order > 100 shares
3. Consecutive Error Limit-- halts on 3 consecutive REST/WS failures
4. Open Position Guard    -- prevents double-entry on reconnect race
5. Market Hours Gate      -- no orders outside 09:30-16:00 ET
6. Warm-up Guard          -- no orders before 200-tick buffer fills
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    WARMUP  = "warmup"   # buffer not ready
    ARMED   = "armed"    # normal operation
    TRIPPED = "tripped"  # breaker open -- no orders


@dataclass
class OrderProposal:
    symbol: str
    side: str                     # "buy" or "sell"
    qty: int
    limit_price: float
    signal_time_ns: int = 0
    model_prob: float = 0.5


class CircuitBreaker:
    """
    Independent risk monitor.
    All state is plain Python -- no numpy, no external libs.

    Usage
    -----
        breaker = CircuitBreaker(starting_equity=10_000)
        breaker.set_buffer_ready()           # after 200-tick warm-up
        breaker.update_equity(new_equity)    # call every N ticks
        approved, reason = breaker.approve(order_proposal)
    """

    # ET offset (conservative: use EST = UTC-5 year-round).
    # For accurate EDT switching use zoneinfo (Python 3.9+) but that
    # would add a soft dependency. The 30-min conservative bias is
    # acceptable for paper trading.
    _ET_OFFSET   = datetime.timedelta(hours=-5)
    _MARKET_OPEN  = datetime.time(9, 30)
    _MARKET_CLOSE = datetime.time(16, 0)

    def __init__(
        self,
        starting_equity: float   = 10_000.0,
        dd_limit_pct: float      = 0.02,    # 2.0% daily drawdown
        max_qty: int             = 100,     # max shares per order
        max_errors: int          = 3,       # consecutive error limit
        _bypass_hours_for_testing: bool = False,  # TEST ONLY: skip hours gate
    ):
        self._start_equity  = starting_equity
        self._daily_equity  = starting_equity   # resets each UTC day
        self._cur_equity    = starting_equity
        self._dd_limit      = dd_limit_pct
        self._max_qty       = max_qty
        self._max_errors    = max_errors
        self._bypass_hours  = _bypass_hours_for_testing

        self._state: CircuitState      = CircuitState.WARMUP
        self._trip_reason: str         = ""
        self._consec_errors: int       = 0
        self._has_position: bool       = False
        self._buffer_ready: bool       = False
        self._last_reset_day: Optional[datetime.date] = None

    # ----------------------------------------------------------------
    # Public mutators
    # ----------------------------------------------------------------

    def set_buffer_ready(self) -> None:
        """Call exactly once when the 200-tick feature buffer fills."""
        self._buffer_ready = True
        if self._state == CircuitState.WARMUP:
            self._state = CircuitState.ARMED
            print("  [BREAKER] Armed: warm-up complete.")

    def update_equity(self, equity: float) -> None:
        """Update mark-to-market equity. Triggers drawdown guard if breached."""
        self._daily_reset()
        self._cur_equity = equity
        dd = (self._daily_equity - equity) / max(self._daily_equity, 1e-9)
        if dd >= self._dd_limit:
            self._trip(f"Daily drawdown {dd*100:.2f}% >= limit {self._dd_limit*100:.1f}%")

    def record_error(self) -> None:
        """Call on every REST/WS error."""
        self._consec_errors += 1
        if self._consec_errors >= self._max_errors:
            self._trip(f"{self._consec_errors} consecutive errors")

    def clear_errors(self) -> None:
        """Call on every successful REST response."""
        self._consec_errors = 0

    def set_position_open(self, open_: bool) -> None:
        self._has_position = open_

    # ----------------------------------------------------------------
    # Core gate
    # ----------------------------------------------------------------

    def approve(self, order: OrderProposal) -> tuple[bool, str]:
        """
        Returns (approved, reason).
        reason is empty string when approved.
        Checks are ordered cheapest-to-most-expensive.
        """
        # Guard 6: warm-up
        if not self._buffer_ready:
            return False, "WARMUP: 200-tick buffer not yet full"

        # Guard 5: market hours (ET)
        if not self._bypass_hours and not self._in_market_hours():
            return False, "HOURS: Outside 09:30-16:00 ET"

        # Guard 1 + 3: tripped
        if self._state == CircuitState.TRIPPED:
            return False, f"TRIPPED: {self._trip_reason}"

        # Guard 2: qty
        if order.qty > self._max_qty:
            return False, f"QTY: {order.qty} > max {self._max_qty}"

        # Guard 2b: minimum
        if order.qty < 1:
            return False, "QTY_ZERO: qty < 1"

        # Guard 4: open position
        if self._has_position:
            return False, "POSITION: existing open position"

        return True, ""

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _in_market_hours(self) -> bool:
        et_tz = datetime.timezone(self._ET_OFFSET)
        now_et = datetime.datetime.now(datetime.timezone.utc).astimezone(et_tz).time()
        return self._MARKET_OPEN <= now_et < self._MARKET_CLOSE

    def _trip(self, reason: str) -> None:
        if self._state != CircuitState.TRIPPED:
            self._state = CircuitState.TRIPPED
            self._trip_reason = reason
            print(f"  [BREAKER] *** TRIPPED *** {reason}")

    def _daily_reset(self) -> None:
        today = datetime.date.today()
        if self._last_reset_day != today:
            self._last_reset_day = today
            self._daily_equity   = self._cur_equity
            # Auto-reset drawdown trip at market open (new trading day)
            if self._state == CircuitState.TRIPPED and "drawdown" in self._trip_reason:
                self._state       = CircuitState.ARMED
                self._trip_reason = ""
                print("  [BREAKER] Daily reset -- drawdown trip cleared.")

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_armed(self) -> bool:
        return self._state == CircuitState.ARMED

    def describe(self) -> dict:
        return {
            "state":                 self._state.value,
            "trip_reason":           self._trip_reason,
            "daily_drawdown_pct":    round(
                (self._daily_equity - self._cur_equity)
                / max(self._daily_equity, 1e-9) * 100, 3
            ),
            "dd_limit_pct":          self._dd_limit * 100,
            "consecutive_errors":    self._consec_errors,
            "has_open_position":     self._has_position,
            "buffer_ready":          self._buffer_ready,
            "in_market_hours":       self._in_market_hours(),
        }
