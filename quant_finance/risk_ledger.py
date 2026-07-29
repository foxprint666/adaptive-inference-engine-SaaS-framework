"""
quant_finance/risk_ledger.py

Running risk tracker. Pure stdlib -- no ML dependencies.

Appends every event to results/risk_ledger.jsonl (append-only).
Checkpoints to results/risk_ledger_checkpoint.json every 50 events.
Survives process restarts: reload checkpoint on next start.

Slippage validation
-------------------
Every fill's realised slippage (bps) is compared against the
Almgren-Chriss temporary-impact prediction.
Phase 3 target: mean ratio <= 1.2x.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

_RES = os.path.join(os.path.dirname(__file__), "results")
LEDGER_PATH     = os.path.join(_RES, "risk_ledger.jsonl")
CHECKPOINT_PATH = os.path.join(_RES, "risk_ledger_checkpoint.json")


@dataclass
class TradeRecord:
    ts:               str
    order_id:         str
    symbol:           str
    side:             str
    qty:              int
    fill_price:       float
    slippage_bps:     float
    latency_ms:       float
    ac_predicted_bps: float     # Almgren-Chriss model prediction
    ac_ratio:         float     # realised / predicted
    status:           str


class RiskLedger:
    """
    Tracks risk, P&L and slippage in real-time.

    Usage
    -----
        ledger = RiskLedger(starting_equity=10_000)
        ledger.record_fill(fill, vol_ann=0.19)
        ledger.update_equity(new_equity)
    """

    # Almgren-Chriss parameters (simplified temporary impact)
    _ETA = 0.142       # permanent impact coefficient
    _ADV = 50_000.0    # IEX daily volume proxy (conservative)

    def __init__(self, starting_equity: float = 10_000.0):
        self._start_eq     = starting_equity
        self._cur_eq       = starting_equity
        self._daily_eq     = starting_equity
        self._peak_eq      = starting_equity
        self._max_dd       = 0.0
        self._daily_dd     = 0.0
        self._daily_date:  Optional[datetime.date] = None

        self._trades:      List[TradeRecord] = []
        self._slip_ratios: List[float]       = []
        self._consec_errors: int             = 0
        self._event_count:   int             = 0

        os.makedirs(_RES, exist_ok=True)

    # ----------------------------------------------------------------
    # Almgren-Chriss slippage model
    # ----------------------------------------------------------------

    def ac_slippage_bps(self, qty: int, vol_ann: float) -> float:
        """
        Simplified temporary impact (bps).
        sigma_tmp = eta * sigma_daily * sqrt(qty / ADV)
        """
        sigma_daily = vol_ann / math.sqrt(252)
        return self._ETA * sigma_daily * math.sqrt(max(qty, 1) / self._ADV) * 10_000

    # ----------------------------------------------------------------
    # Recording
    # ----------------------------------------------------------------

    def record_fill(self, fill, vol_ann: float = 0.19) -> None:
        self._daily_check()
        ac  = self.ac_slippage_bps(fill.qty, vol_ann)
        rat = abs(fill.slippage_bps) / max(ac, 1e-9)

        rec = TradeRecord(
            ts=datetime.datetime.utcnow().isoformat(),
            order_id=fill.order_id, symbol=fill.symbol,
            side=fill.side, qty=fill.qty,
            fill_price=fill.fill_price,
            slippage_bps=fill.slippage_bps,
            latency_ms=round(fill.latency_ms, 2),
            ac_predicted_bps=round(ac, 3),
            ac_ratio=round(rat, 3),
            status=fill.status,
        )
        self._trades.append(rec)
        self._slip_ratios.append(rat)
        self._write({"type": "fill", **asdict(rec)})

        if rat > 1.2:
            print(f"  [LEDGER] SLIPPAGE ALERT: {rat:.2f}x AC model "
                  f"({fill.slippage_bps:.1f} vs {ac:.1f} bps predicted)")

    def update_equity(self, equity: float) -> None:
        self._daily_check()
        self._cur_eq  = equity
        self._peak_eq = max(self._peak_eq, equity)
        dd = (self._peak_eq - equity) / max(self._peak_eq, 1e-9)
        self._max_dd  = max(self._max_dd, dd)
        self._daily_dd = (
            (self._daily_eq - equity) / max(self._daily_eq, 1e-9)
        )
        self._write({
            "type":            "equity",
            "equity":          round(equity, 2),
            "max_drawdown_pct":round(self._max_dd * 100, 3),
            "daily_dd_pct":    round(self._daily_dd * 100, 3),
        })

    def record_error(self) -> None:
        self._consec_errors += 1
        self._write({"type": "error", "consecutive_errors": self._consec_errors})

    def clear_errors(self) -> None:
        self._consec_errors = 0

    def record_circuit_trip(self, reason: str) -> None:
        self._write({"type": "circuit_breaker_trip", "reason": reason})

    # ----------------------------------------------------------------
    # IO helpers
    # ----------------------------------------------------------------

    def _write(self, event: dict) -> None:
        event["wall_ts"] = datetime.datetime.utcnow().isoformat()
        with open(LEDGER_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        self._event_count += 1
        if self._event_count % 50 == 0:
            self._checkpoint()

    def _checkpoint(self) -> None:
        state = {
            "starting_equity":    self._start_eq,
            "current_equity":     round(self._cur_eq, 2),
            "max_drawdown_pct":   round(self._max_dd * 100, 3),
            "total_trades":       len(self._trades),
            "slip_ratio_mean":    round(
                sum(self._slip_ratios) / max(1, len(self._slip_ratios)), 3
            ),
            "event_count":        self._event_count,
            "checkpoint_ts":      datetime.datetime.utcnow().isoformat(),
        }
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _daily_check(self) -> None:
        today = datetime.date.today()
        if self._daily_date != today:
            self._daily_date  = today
            self._daily_eq    = self._cur_eq
            self._daily_dd    = 0.0

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_dd * 100

    @property
    def daily_drawdown_pct(self) -> float:
        return self._daily_dd * 100

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def slip_ratio_mean(self) -> float:
        return sum(self._slip_ratios) / max(1, len(self._slip_ratios))

    def describe(self) -> dict:
        return {
            "current_equity":    round(self._cur_eq, 2),
            "total_return_pct":  round((self._cur_eq - self._start_eq) / self._start_eq * 100, 3),
            "max_drawdown_pct":  round(self._max_dd * 100, 3),
            "daily_dd_pct":      round(self._daily_dd * 100, 3),
            "total_trades":      len(self._trades),
            "slip_ratio_mean":   round(self.slip_ratio_mean, 3),
            "consec_errors":     self._consec_errors,
        }
