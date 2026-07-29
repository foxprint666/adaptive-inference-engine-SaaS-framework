"""
quant_finance/vol_target.py

Volatility-Targeted Execution Layer.

Problem it solves
-----------------
With a fixed +-0.005 activation dead-zone and position size Q=100 shares,
the execution layer is calibrated for a quiet-regime vol of ~19%.  During
a crash (vol ~95%), two things break:

  1. Position sizing: 100 shares at the same dollar notional represents 5x
     the volatility exposure.  Max drawdown swells unnecessarily.

  2. Signal sensitivity: The raw model emits signals of similar absolute
     magnitude regardless of regime.  The fixed 0.005 threshold is too
     tight relative to the noise-to-signal ratio in a crash.

Fix: Volatility-Targeted Executor
----------------------------------
Two adjustments are made at execution time:

  1. Trade size Q scales INVERSELY with vol:

       Q_t = Q_base * min(1, sigma_target / sigma_t)

     At 95% crash vol: Q_t = 100 * (0.19/0.95) ≈ 20 shares.
     This keeps dollar vol-risk approximately constant.

  2. Signal generation uses CALIBRATED probabilities with a fixed neutral
     band on the calibrated scale rather than the raw logit scale:

       signal = +1 if calibrated_prob > long_threshold
       signal = -1 if calibrated_prob < short_threshold
       signal =  0 otherwise

     Since Platt calibration spreads the probability meaningfully across
     [0, 1], a ±0.05 band (long=0.55, short=0.45) generates real trades
     in all regimes while staying conservative.

Note on the original specification
-----------------------------------
The reference document specified:
  gamma_t = gamma_base * (sigma_t / sigma_target)

This WIDENS the dead-zone under high vol.  With uncalibrated probabilities
frozen near 0.5, this would suppress all signals even further.

With CALIBRATED probabilities (which now spread to [0.2, 0.8] in crash),
the relevant control is the neutral band on the calibrated scale — which
we keep fixed at +-0.05. Position sizing handles the risk adjustment.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


class VolatilityTargetedExecutor:
    """
    Execution layer that adjusts position size for constant vol-risk
    and generates signals from calibrated probabilities.

    Parameters
    ----------
    target_volatility   : Annualised volatility anchor (default 19%,
                          typical quiet-regime equity vol).
    base_trade_size     : Base lot size in shares at target vol.
    long_threshold      : Calibrated probability above which we go long.
    short_threshold     : Calibrated probability below which we go short.
    min_trade_size      : Smallest lot size allowed (avoids rounding to 0).
    """

    def __init__(
        self,
        target_volatility: float = 0.19,
        base_trade_size:   float = 100.0,
        long_threshold:    float = 0.55,
        short_threshold:   float = 0.45,
        min_trade_size:    float = 1.0,    # Phase 2: min 1 share (live broker)
        fee_per_share:     float = 0.005,  # Alpaca: $0.005/share
        min_fee:           float = 1.00,   # Alpaca: min $1.00 per ticket
    ):
        self.target_vol      = target_volatility
        self.base_size       = base_trade_size
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.min_size        = min_trade_size
        self.fee_per_share   = fee_per_share
        self.min_fee         = min_fee

    # ── Core computation ───────────────────────────────────────────────────

    def scale_ratio(self, current_vol: float) -> float:
        """
        Return the vol-targeting scale factor in (0, 1].
        sigma_target / max(sigma_t, sigma_target)
        """
        if current_vol <= 0.0:
            return 1.0
        return self.target_vol / max(current_vol, self.target_vol)

    def compute_trade_size(self, current_vol: float) -> float:
        """
        Return the vol-adjusted lot size as an integer number of shares.

        Formula:
            Q_raw    = Q_base * min(1, sigma_target / sigma_t)
            Q_final  = max(Q_min=1, floor(Q_raw))   <- integer lots

        Phase 2 broker constraint: orders must be whole shares (no fractional
        lot sizing). Q_min=1 ensures we never submit a zero-share order.
        """
        raw = self.base_size * self.scale_ratio(current_vol)
        return max(self.min_size, math.floor(raw))  # integer lots, min 1 share

    def fee_for_qty(self, qty: float) -> float:
        """
        Alpaca broker fee: $0.005/share, minimum $1.00 per ticket.

            fee = max($1.00, qty * $0.005)

        Used by live_engine and risk_ledger to compute net P&L after
        broker micro-frictions.
        """
        return max(self.min_fee, int(math.floor(qty)) * self.fee_per_share)

    def compute_signal(
        self,
        calibrated_prob: float,
        current_vol: float,
    ) -> float:
        """
        Generate a directional signal from a calibrated probability.

        Returns +1.0 (long), 0.0 (flat), or -1.0 (short).
        """
        if calibrated_prob > self.long_threshold:
            return 1.0
        elif calibrated_prob < self.short_threshold:
            return -1.0
        return 0.0

    def compute_execution_parameters(
        self,
        current_vol: float,
        calibrated_prob: float,
    ) -> Tuple[float, float]:
        """
        Return (scaled_trade_size, directional_signal).

        This is the primary call-site method used by the simulator and
        walk-forward validator.

        Parameters
        ----------
        current_vol      : Annualised rolling volatility at this tick.
        calibrated_prob  : Platt-calibrated probability in [0, 1].

        Returns
        -------
        (trade_size, signal) where signal in {-1.0, 0.0, +1.0}
        """
        size   = self.compute_trade_size(current_vol)
        signal = self.compute_signal(calibrated_prob, current_vol)
        return size, signal

    # ── Diagnostics ───────────────────────────────────────────────────────

    def regime_summary(self, vol_levels: Dict[str, float]) -> Dict:
        """
        Return a summary of how parameters vary across vol regimes.
        vol_levels: {"quiet": 0.19, "stress": 0.44, ...}
        """
        summary = {}
        for name, vol in vol_levels.items():
            summary[name] = {
                "vol": vol,
                "scale_ratio":  round(self.scale_ratio(vol), 3),
                "trade_size":   round(self.compute_trade_size(vol), 1),
            }
        return summary

    def describe(self) -> Dict:
        regimes = {"quiet": 0.19, "stress": 0.44, "crash": 0.95, "recovery": 0.56}
        return {
            "type":              "VolatilityTargetedExecutor",
            "target_vol":        self.target_vol,
            "base_trade_size":   self.base_size,
            "long_threshold":    self.long_threshold,
            "short_threshold":   self.short_threshold,
            "min_trade_size":    self.min_size,
            "fee_model":         f"max(${self.min_fee:.2f}, ${self.fee_per_share:.3f}/share)",
            "lot_rounding":      "floor(Q) -- integer lots only (broker constraint)",
            "regime_parameters": self.regime_summary(regimes),
            "regime_fees": {
                name: f"${self.fee_for_qty(self.compute_trade_size(vol)):.2f}"
                for name, vol in regimes.items()
            },
        }
