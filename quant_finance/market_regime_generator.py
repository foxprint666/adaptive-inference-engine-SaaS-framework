"""
quant_finance/market_regime_generator.py

Synthetic multi-regime market data generator.

Produces a full multi-year tick dataset that covers:
  - Regime 0 – Quiet (low volatility, trending)
  - Regime 1 – Stress (rising vol, mean-reverting)
  - Regime 2 – Crash (VIX spike, fat tails, gap-down)
  - Regime 3 – Recovery (vol decay, bid-ask widening slowly normalises)

No external data dependency: all values are generated from statistical
processes calibrated to real equity-market stylised facts:
  - Autocorrelated volatility (GARCH(1,1) approximation)
  - Heavy-tailed returns during crashes (Student-t, df=3)
  - Bid-ask spread scaling with volatility
  - Order-imbalance drawn from a zero-mean AR(1) process

Output CSV columns
------------------
  timestamp, price, bid, ask, volume, rolling_volatility, imbalance, regime

This CSV is consumed by:
  - ingress_sim.py  (historical drift simulator)
  - run_validation.py (walk-forward cross-validation)
"""

from __future__ import annotations

import math
import random
import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional


# ── Regime definitions ──────────────────────────────────────────────────────

@dataclass
class RegimeSpec:
    name: str
    n_ticks: int           # Number of synthetic ticks in this regime
    mu: float              # Daily drift (annualised / 252)
    sigma_base: float      # Base daily volatility
    garch_alpha: float     # GARCH shock weight
    garch_beta: float      # GARCH persistence weight
    spread_bps: float      # Base bid-ask spread in basis points
    tail_df: Optional[float] = None  # Student-t df for crash tails (None = Gaussian)
    gap_prob: float = 0.0  # Probability of a gap (overnight jump) on any tick
    gap_magnitude: float = 0.0  # Gap size as fraction of price


REGIMES: List[RegimeSpec] = [
    RegimeSpec(
        name="quiet",
        n_ticks=1000,
        mu=0.08 / 252,        # 8 % pa drift
        sigma_base=0.012,     # 1.2 % daily vol (annualised ~19 %)
        garch_alpha=0.05,
        garch_beta=0.90,
        spread_bps=5,
    ),
    RegimeSpec(
        name="stress",
        n_ticks=500,
        mu=-0.05 / 252,       # Slight negative drift
        sigma_base=0.028,     # 2.8 % daily vol (annualised ~44 %)
        garch_alpha=0.12,
        garch_beta=0.85,
        spread_bps=15,
    ),
    RegimeSpec(
        name="crash",
        n_ticks=300,
        mu=-0.35 / 252,       # -35 % pa drift (rapid drawdown)
        sigma_base=0.060,     # 6 % daily vol (annualised ~95 %)
        garch_alpha=0.20,
        garch_beta=0.75,
        spread_bps=60,
        tail_df=3.0,          # Heavy tails
        gap_prob=0.04,        # 4 % chance of overnight gap per tick
        gap_magnitude=0.025,  # 2.5 % gap size
    ),
    RegimeSpec(
        name="recovery",
        n_ticks=700,
        mu=0.18 / 252,        # Strong recovery drift
        sigma_base=0.035,
        garch_alpha=0.10,
        garch_beta=0.86,
        spread_bps=25,
        gap_prob=0.01,
        gap_magnitude=0.015,
    ),
]


# ── Student-t variate sampler ────────────────────────────────────────────────

def _student_t(df: float, rng: random.Random) -> float:
    """Box-Muller + chi-squared approximation for Student-t(df)."""
    u1, u2 = rng.random(), rng.random()
    z = math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)
    # chi-squared via sum of squares of N(0,1)
    chi2 = sum(
        (-2.0 * math.log(rng.random() + 1e-12)) * (rng.random() < 0.5 and 1 or -1) ** 0  # noqa
        for _ in range(int(df))
    )
    chi2 = max(chi2, 0.01)
    return z / math.sqrt(chi2 / df)


def _normal(rng: random.Random) -> float:
    return rng.gauss(0, 1)


# ── Rolling volatility estimator (exponential) ───────────────────────────────

def _ewma_vol(returns: List[float], lam: float = 0.94) -> float:
    if not returns:
        return 0.0
    var = returns[0] ** 2
    for r in returns[1:]:
        var = lam * var + (1 - lam) * r ** 2
    return math.sqrt(var)


# ── Order imbalance AR(1) ─────────────────────────────────────────────────────

def _next_imbalance(prev: float, rng: random.Random, phi: float = 0.6) -> float:
    noise = rng.gauss(0, 1.0)
    return phi * prev + math.sqrt(1 - phi ** 2) * noise


# ── Main generator ────────────────────────────────────────────────────────────

def generate_market_data(
    output_path: str,
    start_price: float = 100.0,
    seed: int = 42,
) -> str:
    """
    Generate synthetic multi-regime market tick data and write to a CSV.

    Returns the path of the written file.
    """
    rng = random.Random(seed)
    rows: List[dict] = []

    price = start_price
    cond_var = REGIMES[0].sigma_base ** 2  # GARCH conditional variance
    imbalance = 0.0
    t = 0  # global tick counter
    recent_returns: List[float] = []
    window = 30  # rolling vol window

    for regime in REGIMES:
        for _ in range(regime.n_ticks):
            # ── GARCH(1,1) conditional variance ──────────────────────────────
            if len(recent_returns) > 0:
                last_r = recent_returns[-1]
            else:
                last_r = 0.0
            cond_var = (
                (1 - regime.garch_alpha - regime.garch_beta) * regime.sigma_base ** 2
                + regime.garch_alpha * last_r ** 2
                + regime.garch_beta * cond_var
            )
            sigma = math.sqrt(max(cond_var, 1e-10))

            # ── Draw return shock ─────────────────────────────────────────────
            if regime.tail_df is not None:
                z = _student_t(regime.tail_df, rng)
            else:
                z = _normal(rng)

            ret = regime.mu + sigma * z

            # ── Overnight gap injection ───────────────────────────────────────
            if regime.gap_prob > 0 and rng.random() < regime.gap_prob:
                gap_direction = 1.0 if rng.random() > 0.5 else -1.0
                ret += gap_direction * regime.gap_magnitude

            price = price * math.exp(ret)
            price = max(price, 0.01)  # No negative prices

            recent_returns.append(ret)
            if len(recent_returns) > window:
                recent_returns.pop(0)

            # ── Rolling annualised volatility ─────────────────────────────────
            rolling_vol = _ewma_vol(recent_returns[-window:]) * math.sqrt(252)

            # ── Bid-ask spread (vol-adjusted) ──────────────────────────────────
            spread_factor = 1.0 + 3.0 * (rolling_vol / 0.20)  # normalize to 20 % baseline
            half_spread = price * (regime.spread_bps / 10_000) * spread_factor / 2.0
            bid = price - half_spread
            ask = price + half_spread

            # ── Volume (log-normal, inversely correlated with spread) ──────────
            volume = max(1, int(rng.lognormvariate(7.5, 0.8) / spread_factor))

            # ── Order imbalance AR(1) ─────────────────────────────────────────
            imbalance = _next_imbalance(imbalance, rng)

            rows.append({
                "tick_index": t,
                "price": round(price, 4),
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "volume": volume,
                "rolling_volatility": round(rolling_vol, 6),
                "imbalance": round(imbalance, 4),
                "regime": regime.name,
                "ret": round(ret, 6),
            })
            t += 1

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[GENERATOR] Wrote {len(rows)} ticks to: {output_path}")
    print(f"[GENERATOR] Regime breakdown:")
    for regime in REGIMES:
        print(f"  {regime.name:<12} {regime.n_ticks:>5} ticks")

    return output_path


if __name__ == "__main__":
    generate_market_data("quant_finance/data/synthetic_market_data.csv")
