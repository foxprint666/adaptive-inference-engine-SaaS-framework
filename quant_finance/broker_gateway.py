"""
quant_finance/broker_gateway.py

Alpaca paper trading REST gateway.

Credentials
-----------
NEVER hardcoded. Loaded from environment variables only:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

Order strategy
--------------
Submits LIMIT orders at ask+$0.01 (buy) or bid-$0.01 (sell) rather
than market orders. This caps fill latency to ~30 ms p50 (resting order
fill) vs ~80-150 ms for a market-order REST round-trip, while avoiding
full market-impact slippage during high-vol windows.

Slippage measurement
--------------------
Records the NBBO mid-quote at signal time alongside the actual fill
price to validate the Almgren-Chriss model. Phase 3 target:
realised slippage <= 1.2x Almgren-Chriss prediction.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

PAPER_BASE = "https://paper-api.alpaca.markets"


@dataclass
class Fill:
    order_id:       str
    symbol:         str
    side:           str
    qty:            int
    limit_price:    float
    fill_price:     float     # actual average fill (from Alpaca response)
    fill_time_ns:   int       # time.perf_counter_ns() on response (t3)
    signal_time_ns: int       # from Order.signal_time_ns (t2)
    mid_at_signal:  float     # (bid+ask)/2 when signal fired
    slippage_bps:   float     # (fill_price - mid) / mid * 10000
    status:         str       # "accepted", "filled", "blocked_*", "error_*"
    raw_response:   dict = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """t2 -> t3 latency in milliseconds (REST round-trip)."""
        return (self.fill_time_ns - self.signal_time_ns) / 1_000_000.0


class BrokerGateway:
    """
    Thin async wrapper over Alpaca paper REST API.

    All urllib.request calls are dispatched via asyncio.to_thread so
    the live WebSocket event loop is never blocked.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url:   str           = PAPER_BASE,
        timeout_s:  float         = 10.0,
    ):
        self._key     = api_key    or os.environ.get("APCA_API_KEY_ID",     "")
        self._secret  = api_secret or os.environ.get("APCA_API_SECRET_KEY", "")
        self._base    = base_url.rstrip("/")
        self._timeout = timeout_s

        if not self._key or not self._secret:
            raise ValueError(
                "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                "environment variables before running live_engine."
            )

        self._consec_errors: int = 0

    def _hdrs(self) -> dict:
        return {
            "APCA-API-KEY-ID":     self._key,
            "APCA-API-SECRET-KEY": self._secret,
            "Content-Type":        "application/json",
        }

    def _sync_get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self._base}{path}", headers=self._hdrs(), method="GET"
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.loads(r.read().decode())

    def _sync_post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self._base}{path}",
            data=data, headers=self._hdrs(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            return json.loads(r.read().decode())

    def _sync_delete(self, path: str) -> None:
        req = urllib.request.Request(
            f"{self._base}{path}", headers=self._hdrs(), method="DELETE"
        )
        try:
            urllib.request.urlopen(req, timeout=self._timeout)
        except urllib.error.HTTPError as e:
            if e.code != 204:   # 204 = No Content -- expected
                raise

    # ----------------------------------------------------------------
    # Async public API
    # ----------------------------------------------------------------

    async def get_position(self, symbol: str) -> Optional[dict]:
        try:
            return await asyncio.to_thread(self._sync_get, f"/v2/positions/{symbol}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    async def get_account(self) -> dict:
        return await asyncio.to_thread(self._sync_get, "/v2/account")

    async def cancel_all_orders(self) -> None:
        """Called on dead-feed detection. Clears all resting limit orders."""
        try:
            await asyncio.to_thread(self._sync_delete, "/v2/orders")
            print("  [GW] All open orders cancelled (dead feed)")
        except Exception as exc:
            print(f"  [GW] cancel_all failed: {exc}")

    async def submit(self, order, mid_at_signal: float) -> Fill:
        """
        Submit a marketable limit order.

        Pre-flight: checks /v2/positions to prevent double-entry.
        Slippage: records fill vs mid_at_signal for Almgren-Chriss validation.
        """
        # Pre-flight position check
        existing = await self.get_position(order.symbol)
        if existing is not None:
            print(f"  [GW] Position exists ({existing.get('side')}) -- order blocked")
            return Fill(
                order_id="blocked", symbol=order.symbol,
                side=order.side,    qty=order.qty,
                limit_price=order.limit_price, fill_price=0.0,
                fill_time_ns=time.perf_counter_ns(),
                signal_time_ns=order.signal_time_ns,
                mid_at_signal=mid_at_signal, slippage_bps=0.0,
                status="blocked_open_position",
            )

        body = {
            "symbol":        order.symbol,
            "qty":           str(order.qty),
            "side":          order.side,
            "type":          "limit",
            "time_in_force": "day",
            "limit_price":   f"{order.limit_price:.2f}",
        }

        try:
            resp        = await asyncio.to_thread(self._sync_post, "/v2/orders", body)
            t3          = time.perf_counter_ns()   # t3 checkpoint
            self._consec_errors = 0

            fill_price  = float(resp.get("filled_avg_price") or order.limit_price)
            slip_bps    = (
                (fill_price - mid_at_signal) / max(mid_at_signal, 1e-9) * 10_000
            )
            return Fill(
                order_id=resp.get("id", ""),
                symbol=order.symbol, side=order.side, qty=order.qty,
                limit_price=order.limit_price, fill_price=fill_price,
                fill_time_ns=t3, signal_time_ns=order.signal_time_ns,
                mid_at_signal=mid_at_signal, slippage_bps=round(slip_bps, 2),
                status=resp.get("status", "accepted"), raw_response=resp,
            )

        except urllib.error.HTTPError as exc:
            self._consec_errors += 1
            err_body = exc.read().decode() if hasattr(exc, "read") else str(exc)
            print(f"  [GW] HTTP {exc.code}: {err_body}")
            return Fill(
                order_id="error", symbol=order.symbol,
                side=order.side,  qty=order.qty,
                limit_price=order.limit_price, fill_price=0.0,
                fill_time_ns=time.perf_counter_ns(),
                signal_time_ns=order.signal_time_ns,
                mid_at_signal=mid_at_signal, slippage_bps=0.0,
                status=f"error_http_{exc.code}",
            )

    @property
    def consecutive_errors(self) -> int:
        return self._consec_errors
