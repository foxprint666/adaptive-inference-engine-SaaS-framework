"""
quant_finance/tick_replay.py

Replay engine: feeds synthetic market CSV through the live engine
code path without a real WebSocket connection.

Critical for validating all async code (circuit breaker, MUSE router,
bid/ask processing, latency logger) before touching the live Alpaca API.

Interface
---------
Identical to LiveFeed.stream() -- drop-in replacement:

    async for tick in TickReplay(csv_path).stream():
        process(tick)   # same logic as live mode

IEX volume note
---------------
The synthetic CSV was generated with ADV ~50,000 (SIP-scale).
Real IEX feed will have ~2-3% of this. Volume-derived features
(ema_imb, vol_of_vol) will be proportionally smaller in live mode.
The model's rolling normalisation handles this -- just be aware when
comparing paper-replay vs live PSI drift metrics.

Timestamps
----------
tick.timestamp_ns is set with time.perf_counter_ns() at replay time
-- identical to live_feed behaviour, so latency_logger arithmetic
works correctly in both modes.
"""

from __future__ import annotations

import asyncio
import csv
import math
import time
from typing import AsyncIterator

from quant_finance.live_feed import Tick


class TickReplay:
    """
    Async generator replaying synthetic_market_data.csv as Tick objects.

    Parameters
    ----------
    csv_path  : path to synthetic_market_data.csv
    speed     : 0.0 = max speed (default), 1.0 = ~1 tick/ms real-time
    symbol    : embedded symbol name in produced Ticks
    """

    def __init__(
        self,
        csv_path: str,
        speed:    float = 0.0,
        symbol:   str   = "SPY",
    ):
        self._csv  = csv_path
        self._spd  = speed
        self._sym  = symbol
        self._cnt  = 0

    async def stream(self) -> AsyncIterator[Tick]:
        with open(self._csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                price = float(row.get("price", 0.0))
                if price <= 0:
                    continue

                # Synthesise realistic bid/ask from rolling vol
                rv = float(row.get("rolling_volatility", 0.01))
                half_spread = max(0.01, price * rv * 0.001)  # min 1 cent
                bid = round(price - half_spread, 4)
                ask = round(price + half_spread, 4)

                tick = Tick(
                    symbol=self._sym,
                    price=price,
                    bid=bid,
                    ask=ask,
                    volume=float(row.get("volume", 1_000.0)),
                    spread=ask - bid,
                    timestamp_ns=time.perf_counter_ns(),  # monotonic t0
                    msg_type="t",
                    raw_ts=row.get("timestamp", ""),
                )

                if self._spd > 0:
                    await asyncio.sleep(self._spd * 0.001)
                else:
                    await asyncio.sleep(0)  # yield control to event loop

                self._cnt += 1
                yield tick

    @property
    def tick_count(self) -> int:
        return self._cnt
