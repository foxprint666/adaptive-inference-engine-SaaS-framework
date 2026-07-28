"""
quant_finance/hysteresis.py

Volatility-Scaled Dynamic Cooldown Controller.

Problem with a static cooldown
-------------------------------
The original SwapHysteresisController uses a fixed cooldown of 50 ticks.
This creates a regime-lag risk:

  - If the system swaps BACK to the baseline model just before a crash
    begins, it is locked in the wrong model for 50 ticks regardless of
    how severe the drawdown becomes.

  - Conversely, during quiet periods a short cooldown causes unnecessary
    swap friction even when drift oscillates near the threshold.

Fix: Volatility-Scaled Cooldown
--------------------------------
The dynamic cooldown inversely scales with current realised volatility:

    cooldown_t = base_cooldown * (vol_ref / vol_t)

Where vol_ref is the quiet-regime anchor volatility (annualised).

  Regime       | Ann. Vol | Cooldown (base=50)
  -------------|----------|-------------------
  Quiet        | 19%      | 50 ticks
  Stress       | 44%      | 21 ticks
  Crash        | 95%      | 10 ticks  <- fast defensive reaction
  Recovery     | 56%      | 17 ticks

This means the system can react to a sudden crash within 10 ticks of
observing an initial swap, rather than waiting a full 50 ticks.

The recovery hysteresis threshold is also scaled: to revert to baseline,
PSI must fall below (threshold * recovery_fraction), where recovery_fraction
is dynamically tightened during high-vol periods.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional


class VolatilityScaledCooldown:
    """
    Dynamic model-swap cooldown that inversely scales with rolling volatility.

    Parameters
    ----------
    base_cooldown        : Cooldown ticks at reference (quiet) volatility.
    vol_ref              : Annualised reference volatility (default 19%).
    min_cooldown         : Hard floor on cooldown (never faster than this).
    max_cooldown         : Hard ceiling on cooldown.
    recovery_fraction    : PSI must fall to threshold * this fraction to revert.
    vol_decay_lambda     : EMA decay for rolling vol estimate (0 = use raw input).
    """

    def __init__(
        self,
        base_cooldown: int = 50,
        vol_ref: float = 0.19,         # Quiet-regime annualised vol anchor
        min_cooldown: int = 5,
        max_cooldown: int = 200,
        recovery_fraction: float = 0.85,
        vol_decay_lambda: float = 0.0,  # 0 = use raw current_vol; >0 = EMA
    ):
        self.base_cooldown     = base_cooldown
        self.vol_ref           = vol_ref
        self.min_cooldown      = min_cooldown
        self.max_cooldown      = max_cooldown
        self.recovery_fraction = recovery_fraction
        self.vol_decay_lambda  = vol_decay_lambda

        # State
        self._last_swap_tick: int  = -max_cooldown       # allow swap at tick 0
        self._active_version: str  = "baseline"
        self._ema_vol: float       = vol_ref             # running EMA of vol
        self.swap_log: List[Dict]  = []

    # ── Cooldown calculation ───────────────────────────────────────────────

    def _update_ema_vol(self, current_vol: float) -> float:
        if self.vol_decay_lambda <= 0.0 or current_vol <= 0.0:
            return max(current_vol, 1e-6)
        lam = self.vol_decay_lambda
        self._ema_vol = lam * self._ema_vol + (1.0 - lam) * current_vol
        return self._ema_vol

    def calculate_cooldown(self, current_vol: float) -> int:
        """
        Return the current dynamic cooldown in ticks.

        cooldown = base * (vol_ref / current_vol)
        Clamped to [min_cooldown, max_cooldown].
        """
        effective_vol = self._update_ema_vol(current_vol)
        if effective_vol <= 0.0:
            return self.base_cooldown

        raw = int(self.base_cooldown * (self.vol_ref / effective_vol))
        return max(self.min_cooldown, min(self.max_cooldown, raw))

    # ── Swap permission ────────────────────────────────────────────────────

    def is_swap_permitted(self, current_tick: int, current_vol: float) -> bool:
        """Return True if enough ticks have elapsed since the last swap."""
        cooldown = self.calculate_cooldown(current_vol)
        elapsed  = current_tick - self._last_swap_tick
        return elapsed >= cooldown

    def should_revert_to_baseline(
        self, current_psi: float, threshold: float, current_vol: float
    ) -> bool:
        """
        Return True if PSI is sufficiently low to justify reverting.

        During high-vol regimes, the recovery criterion is tightened
        (recovery_fraction shrinks), so we demand a larger PSI drop before
        reverting — preventing premature return to baseline during a crash.
        """
        # During crash (vol >> vol_ref), tighten to 70%; quiet = 85%
        vol_ratio = min(1.0, self.vol_ref / max(current_vol, 1e-6))
        effective_fraction = self.recovery_fraction * vol_ratio
        effective_fraction = max(0.50, effective_fraction)  # never below 50%
        return current_psi < (threshold * effective_fraction)

    # ── Swap execution ─────────────────────────────────────────────────────

    def execute_swap(
        self,
        new_version: str,
        current_tick: int,
        psi: float,
        current_vol: float,
        auc: float = 0.0,
    ) -> None:
        """Record a swap and reset the cooldown timer."""
        cooldown_used = self.calculate_cooldown(current_vol)
        self.swap_log.append({
            "tick":         current_tick,
            "from":         self._active_version,
            "to":           new_version,
            "psi_at_swap":  round(psi, 4),
            "auc_at_swap":  round(auc, 4),
            "vol_at_swap":  round(current_vol, 4),
            "cooldown_used": cooldown_used,
        })
        self._active_version = new_version
        self._last_swap_tick = current_tick

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def active_version(self) -> str:
        return self._active_version

    # ── Diagnostics ───────────────────────────────────────────────────────

    def describe(self) -> Dict:
        return {
            "type":               "VolatilityScaledCooldown",
            "base_cooldown":      self.base_cooldown,
            "vol_ref":            self.vol_ref,
            "min_cooldown":       self.min_cooldown,
            "max_cooldown":       self.max_cooldown,
            "recovery_fraction":  self.recovery_fraction,
            "n_swaps":            len(self.swap_log),
        }
