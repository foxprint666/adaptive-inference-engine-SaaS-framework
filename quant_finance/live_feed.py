"""
quant_finance/live_feed.py

Async WebSocket consumer for Alpaca IEX market data.

IEX Volume Note
---------------
The free Alpaca IEX feed carries ~2-3% of total consolidated US stock
volume. Quote density on SPY is high, but volume and size fields will be
proportionally smaller than SIP feeds. Volume-derived features (ema_imb,
vol_of_vol) reflect IEX order flow only -- not the consolidated market.
This is handled gracefully by the model's normalisation, but be aware
when interpreting imbalance feature magnitudes in live trading.

Heartbeat
---------
If no tick is received for > 5 seconds, the engine is notified via
the on_dead_feed async callback. It should cancel all open orders and
pause signal generation. Reconnection uses exponential backoff:
1 s -> 2 s -> 4 s -> 8 s -> 16 s -> 30 s max.

Latency
-------
tick.timestamp_ns is set with time.perf_counter_ns() immediately on
WebSocket message receipt -- before any parsing. This is the t0
checkpoint for latency_logger. Do NOT use time.time_ns() -- NTP
synchronisations would corrupt latency deltas.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, List, Optional

# websockets is imported lazily inside LiveFeed.stream() so that
# Tick and helper classes are importable without the package installed.
# Install when running live:  pip install websockets>=11.0
_websockets = None


def _get_websockets():
    """Lazy import of websockets -- only needed for live feed mode."""
    global _websockets
    if _websockets is None:
        try:
            import websockets as _ws
            import websockets.exceptions  # noqa: F401
            _websockets = _ws
        except ImportError:
            raise ImportError(
                "Live feed requires websockets. "
                "Install when online: pip install websockets>=11.0"
            )
    return _websockets


WS_URL_IEX        = "wss://stream.data.alpaca.markets/v2/iex"
HEARTBEAT_TIMEOUT = 5.0      # seconds before dead-feed declared
MAX_BACKOFF       = 30.0     # seconds


@dataclass
class Tick:
    """
    Unified tick -- produced from both quote ('q') and trade ('t') messages.

    timestamp_ns : time.perf_counter_ns() at WebSocket message receipt.
                   Use this -- not raw_ts -- for all latency arithmetic.
    volume       : IEX-scale volume (~2-3% of consolidated US market).
    """
    symbol:       str
    price:        float     # trade price or (bid+ask)/2 for quotes
    bid:          float
    ask:          float
    volume:       float     # IEX size; ~2-3% of consolidated US volume
    spread:       float     # ask - bid
    timestamp_ns: int       # perf_counter_ns at receipt  <-- t0 for latency
    msg_type:     str       # 'q' = quote, 't' = trade
    raw_ts:       str       # Alpaca ISO timestamp


class LiveFeed:
    """
    Async generator yielding Ticks from Alpaca IEX WebSocket.

    Usage
    -----
        feed = LiveFeed("SPY", api_key, api_secret)
        async for tick in feed.stream():
            process(tick)
    """

    def __init__(
        self,
        symbol: str,
        api_key: str,
        api_secret: str,
        ws_url: str = WS_URL_IEX,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
        on_dead_feed: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.symbol             = symbol.upper()
        self._key               = api_key
        self._secret            = api_secret
        self._url               = ws_url
        self._hb_timeout        = heartbeat_timeout
        self._on_dead_feed      = on_dead_feed

        self._tick_count:   int   = 0
        self._reconnects:   int   = 0
        self._last_mono:    float = 0.0
        self._gap_samples:  List[float] = []

    # ----------------------------------------------------------------
    # WebSocket handshake
    # ----------------------------------------------------------------

    async def _auth(self, ws) -> None:
        await ws.send(json.dumps({
            "action": "auth",
            "key":    self._key,
            "secret": self._secret,
        }))
        # Alpaca WebSocket sends [{"T": "success", "msg": "connected"}] upon open
        # followed by [{"T": "success", "msg": "authenticated"}] after auth packet.
        for _ in range(5):
            raw = await ws.recv()
            resp = json.loads(raw)
            msgs = resp if isinstance(resp, list) else [resp]
            for m in msgs:
                if m.get("T") == "success" and m.get("msg") == "authenticated":
                    return
                if m.get("T") == "error":
                    raise ConnectionError(f"Alpaca auth error: {m.get('msg')}")
                if m.get("T") == "success" and m.get("msg") == "connected":
                    continue  # Ignore welcome frame, await auth response
        raise ConnectionError("Timed out waiting for Alpaca authentication response")

    async def _subscribe(self, ws) -> None:
        await ws.send(json.dumps({
            "action": "subscribe",
            "quotes": [self.symbol],
            "trades": [self.symbol],
        }))
        ack = json.loads(await ws.recv())
        print(f"  [FEED] Subscription ack: {ack}")

    # ----------------------------------------------------------------
    # Tick parsing
    # ----------------------------------------------------------------

    def _parse(self, raw: str) -> List[Tick]:
        # timestamp_ns captured BEFORE json.loads -- true t0
        ts_ns = time.perf_counter_ns()
        ticks: List[Tick] = []

        messages = json.loads(raw)
        if not isinstance(messages, list):
            messages = [messages]

        for m in messages:
            t = m.get("T")
            s = m.get("S", "")
            if s != self.symbol:
                continue

            if t == "q":    # Quote
                bp  = float(m.get("bp", 0.0))
                ap  = float(m.get("ap", 0.0))
                mid = (bp + ap) / 2.0 if bp and ap else max(bp, ap)
                vol = float(m.get("bs", 0) + m.get("as", 0))  # combined size
                ticks.append(Tick(
                    symbol=s, price=mid, bid=bp, ask=ap,
                    volume=vol, spread=ap - bp,
                    timestamp_ns=ts_ns, msg_type="q",
                    raw_ts=m.get("t", ""),
                ))

            elif t == "t": # Trade
                p = float(m.get("p", 0.0))
                s_ = float(m.get("s", 0.0))
                ticks.append(Tick(
                    symbol=s, price=p, bid=p, ask=p,
                    volume=s_, spread=0.0,
                    timestamp_ns=ts_ns, msg_type="t",
                    raw_ts=m.get("t", ""),
                ))

        return ticks

    # ----------------------------------------------------------------
    # Feed quality monitor
    # ----------------------------------------------------------------

    def _track_gap(self) -> None:
        now = time.monotonic()
        if self._last_mono > 0:
            gap_ms = (now - self._last_mono) * 1000.0
            self._gap_samples.append(gap_ms)
            if len(self._gap_samples) > 500:
                self._gap_samples.pop(0)
            if len(self._gap_samples) >= 100:
                s = sorted(self._gap_samples)
                p99 = s[int(len(s) * 0.99)]
                if p99 > 2000.0:
                    print(f"  [FEED QUALITY] WARN: tick-gap p99={p99:.0f}ms > 2000ms")
        self._last_mono = now

    # ----------------------------------------------------------------
    # Public stream
    # ----------------------------------------------------------------

    async def stream(self) -> AsyncIterator[Tick]:
        """Async generator. Reconnects automatically on failure."""
        ws_lib = _get_websockets()  # lazy import -- raises if not installed
        backoff = 1.0
        while True:
            try:
                async with ws_lib.connect(
                    self._url, ping_interval=20, ping_timeout=10
                ) as ws:
                    await self._auth(ws)
                    await self._subscribe(ws)
                    backoff = 1.0
                    self._reconnects = 0
                    print(f"  [FEED] Connected | {self.symbol} | {self._url}")

                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=self._hb_timeout
                            )
                        except asyncio.TimeoutError:
                            print(f"  [FEED] DEAD: no tick for {self._hb_timeout}s")
                            if self._on_dead_feed:
                                await self._on_dead_feed()
                            break  # reconnect

                        self._track_gap()
                        for tick in self._parse(raw):
                            self._tick_count += 1
                            yield tick

            except Exception as exc:
                # Catch websockets errors by name to avoid import-time binding
                exc_type = type(exc).__name__
                if exc_type in (
                    "ConnectionClosed", "WebSocketException",
                    "ConnectionError", "OSError",
                ):
                    self._reconnects += 1
                    print(
                        f"  [FEED] Disconnected ({exc}). "
                        f"Retry #{self._reconnects} in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, MAX_BACKOFF)
                else:
                    raise

    def describe(self) -> dict:
        s = sorted(self._gap_samples) if self._gap_samples else [0.0]
        return {
            "symbol":         self.symbol,
            "tick_count":     self._tick_count,
            "reconnect_count":self._reconnects,
            "gap_p50_ms":     round(s[len(s) // 2], 1),
            "gap_p99_ms":     round(s[int(len(s) * 0.99)], 1),
            "iex_volume_note":"IEX ~2-3% of consolidated US volume",
        }
