"""
quant_finance/feature_buffer.py

200-tick sliding window feature extractor.

IEX Volume Note
---------------
IEX accounts for ~2-3% of total consolidated US stock volume.
Volume and imbalance features (ema_imb, vol_of_vol) will be
proportionally smaller than SIP-equivalent features.
The model's ridge regression is scale-agnostic (features are
normalised by rolling std inside build_features), but monitor
the raw feature distributions during the first week of paper
trading to confirm they match the training distribution.

Warm-up Guard
-------------
Returns None for the first 200 ticks. The live_engine checks for
None and skips execution. The circuit breaker is notified via
set_buffer_ready() once the buffer is full.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from quant_finance.quant_model import build_features

BUFFER_SIZE = 200
LOOK_BACK   = 20    # same as training


class FeatureBuffer:
    """
    Maintains a rolling 200-tick window of (price, volume, imbalance).
    Calls build_features() on every push.

    Output features (6-dim):
        [mom_short, mom_med, vol, vol_z, ema_imb, vol_of_vol]
    """

    def __init__(self, window: int = BUFFER_SIZE):
        self._window        = window
        self._prices:     deque = deque(maxlen=window + 1)
        self._volumes:    deque = deque(maxlen=window + 1)
        self._imbalances: deque = deque(maxlen=window + 1)
        self._tick_count: int   = 0
        self._vol_history:deque = deque(maxlen=50)
        self._ready:      bool  = False

    def push(self, tick) -> Optional[List[float]]:
        """
        Push tick and return 6-dim feature vector, or None during warm-up.

        Imbalance proxy: normalised (bid - ask) / ask.
        IEX note: bid/ask sizes are IEX-only; ratio is directionally meaningful
        even if absolute magnitudes are smaller than consolidated.
        """
        if tick.ask > 0 and tick.bid > 0:
            imb = (tick.bid - tick.ask) / max(tick.ask, 1e-9)
        else:
            imb = 0.0

        self._prices.append(tick.price)
        self._volumes.append(max(tick.volume, 1.0))
        self._imbalances.append(imb)
        self._tick_count += 1

        if len(self._prices) < self._window:
            return None   # warm-up

        if not self._ready:
            self._ready = True

        feat = build_features(
            list(self._prices),
            list(self._volumes),
            list(self._imbalances),
            look_back=LOOK_BACK,
        )
        if feat is not None:
            vol_ann = feat[2] * math.sqrt(252)
            self._vol_history.append(vol_ann)

        return feat

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def fill_pct(self) -> float:
        return min(100.0, len(self._prices) / self._window * 100.0)

    @property
    def latest_vol_ann(self) -> float:
        """Most recent annualised vol estimate (default 19% if buffer empty)."""
        if not self._vol_history:
            return 0.19
        return self._vol_history[-1]

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def describe(self) -> dict:
        return {
            "tick_count":     self._tick_count,
            "fill_pct":       round(self.fill_pct, 1),
            "is_ready":       self._ready,
            "latest_vol_ann": round(self.latest_vol_ann, 4),
            "iex_note":       "Volume = IEX order flow (~2-3% of US consolidated)",
        }
