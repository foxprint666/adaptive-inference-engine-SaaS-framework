"""
quant_finance/ingress_sim.py

Historical Drift Simulator — Quantitative Finance Edition.

Replays synthetic market tick data (or any CSV with the right schema),
feeding each tick as a feature vector into the local adaptive models,
tracking model hot-swaps, and recording full execution accounting:

  raw price → slippage → transaction cost → execution price → P&L

Key additions over the reference implementation
-----------------------------------------------
1. Volatility-scaled slippage (Almgren-Chriss style):
     slippage = η * σ * √(Q / ADV)
   where σ = realised vol, Q = order size, ADV = avg daily volume.

2. Full per-tick P&L accounting with mark-to-market:
     unrealised_pnl = position * (current_price - entry_price)
     realised_pnl   = accumulated closed-trade profits

3. Anti-flicker hysteresis controller (SwapHysteresisController):
   Prevents model thrashing when drift oscillates near the threshold.

4. Walk-forward episode tracking: every ``window_size`` ticks, a new
   validation window begins and results are tabulated.

5. Result persistence: all tick results are written to a JSON-L file
   so the dashboard can render them without re-running the simulation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Hysteresis Controller ─────────────────────────────────────────────────────

@dataclass
class SwapHysteresisController:
    """
    Prevents rapid model thrashing by enforcing a cooldown period
    and a stricter recovery threshold before reverting to baseline.

    Parameters
    ----------
    cooldown_ticks        : Minimum ticks between hot-swaps.
    recovery_psi_fraction : PSI must fall below (threshold * fraction)
                            to permit reverting to baseline.
    """
    cooldown_ticks: int = 100
    recovery_psi_fraction: float = 0.85

    _ticks_since_swap: int = field(default=0, init=False)
    _active_version: str = field(default="baseline", init=False)
    swap_log: List[Dict] = field(default_factory=list, init=False)

    def tick(self) -> None:
        self._ticks_since_swap += 1

    def is_swap_allowed(self) -> bool:
        return self._ticks_since_swap >= self.cooldown_ticks

    def should_revert_to_baseline(self, current_psi: float, threshold: float) -> bool:
        return current_psi < (threshold * self.recovery_psi_fraction)

    def execute_swap(self, new_version: str, tick_idx: int, psi: float) -> None:
        old = self._active_version
        self._active_version = new_version
        self._ticks_since_swap = 0
        self.swap_log.append({
            "tick": tick_idx,
            "from": old,
            "to": new_version,
            "psi_at_swap": round(psi, 4),
        })

    @property
    def active_version(self) -> str:
        return self._active_version


# ── Slippage Models ───────────────────────────────────────────────────────────

def almgren_chriss_slippage(
    price: float,
    vol_annualised: float,
    order_size: float = 100.0,
    adv: float = 50_000.0,
    eta: float = 0.1,
) -> float:
    """
    Almgren-Chriss market impact model.

    Linear temporary impact:
      slippage = η * σ_daily * sqrt(Q / ADV) * price

    η      : market impact coefficient (≈0.1 for liquid equities)
    σ_daily: daily vol = annualised vol / sqrt(252)
    Q      : order size (shares)
    ADV    : average daily volume (shares)
    """
    sigma_daily = vol_annualised / math.sqrt(252)
    q_frac = order_size / max(adv, 1.0)
    impact_bps = eta * sigma_daily * math.sqrt(q_frac)
    return price * impact_bps  # absolute slippage in price units


def base_spread_cost(ask: float, bid: float) -> float:
    """Half-spread crossing cost."""
    return (ask - bid) / 2.0


# ── PSI Calculation (standalone, no Redis) ────────────────────────────────────

def compute_psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Compute PSI between two distributions.
    PSI < 0.10 → stable
    PSI 0.10–0.25 → moderate drift
    PSI ≥ 0.25 → significant drift → retrain
    """
    if not expected or not actual:
        return 0.0
    combined = expected + actual
    lo, hi = min(combined), max(combined)
    if lo == hi:
        return 0.0
    eps = 1e-8
    bw = (hi - lo) / bins
    edges = [lo + i * bw for i in range(bins + 1)]
    edges[-1] = hi + eps

    def _bin(vals: List[float]) -> List[int]:
        counts = [0] * bins
        for v in vals:
            for b in range(bins):
                if edges[b] <= v < edges[b + 1]:
                    counts[b] += 1
                    break
        return counts

    ec = _bin(expected)
    ac = _bin(actual)
    n_e, n_a = len(expected), len(actual)
    psi = 0.0
    for e, a in zip(ec, ac):
        ep = (e / n_e) if e > 0 else eps
        ap = (a / n_a) if a > 0 else eps
        psi += (ap - ep) * math.log(ap / ep)
    return max(0.0, psi)


# ── Adversarial AUC (manual, no sklearn) ──────────────────────────────────────

def compute_adversarial_auc(baseline: List[List[float]], current: List[List[float]]) -> float:
    """
    Adversarial validation AUC.
    Combine two sets of feature vectors, label baseline=0, current=1,
    then train a naive Bayes discriminator and compute AUC.

    AUC ≈ 0.5 → distributions are indistinguishable (no drift)
    AUC ≈ 0.72+ → significant distributional shift
    """
    if not baseline or not current:
        return 0.5
    n0, n1 = len(baseline), len(current)
    d = len(baseline[0])

    # Naive Bayes: compute per-feature means for each class
    def _mean(vecs: List[List[float]], j: int) -> float:
        return sum(v[j] for v in vecs) / len(vecs)

    mu0 = [_mean(baseline, j) for j in range(d)]
    mu1 = [_mean(current, j) for j in range(d)]

    # Score = log P(class=1) - log P(class=0), using Gaussian NB
    def _var(vecs: List[List[float]], j: int, mu: float) -> float:
        return sum((v[j] - mu) ** 2 for v in vecs) / max(1, len(vecs)) + 1e-9

    var0 = [_var(baseline, j, mu0[j]) for j in range(d)]
    var1 = [_var(current, j, mu1[j]) for j in range(d)]

    def _score(x: List[float]) -> float:
        ll1 = sum(-0.5 * ((x[j] - mu1[j]) ** 2 / var1[j]) - 0.5 * math.log(2 * math.pi * var1[j]) for j in range(d))
        ll0 = sum(-0.5 * ((x[j] - mu0[j]) ** 2 / var0[j]) - 0.5 * math.log(2 * math.pi * var0[j]) for j in range(d))
        return ll1 - ll0

    scores_0 = [(_score(x), 0) for x in baseline]
    scores_1 = [(_score(x), 1) for x in current]
    all_scores = sorted(scores_0 + scores_1, key=lambda t: t[0], reverse=True)

    auc = 0.0
    n_pos = 0
    for i, (_, label) in enumerate(all_scores):
        if label == 1:
            n_pos += 1
        else:
            auc += n_pos
    auc /= max(1, n0 * n1)
    return min(1.0, max(0.0, auc))


# ── Simulator ─────────────────────────────────────────────────────────────────

class HistoricalDriftSimulator:
    """
    Replays historical/synthetic tick data through adaptive quant models,
    tracking drift, hot-swaps, slippage, and full P&L accounting.
    """

    def __init__(
        self,
        csv_path: str,
        baseline_model,
        candidate_model,
        psi_threshold: float = 0.25,
        auc_threshold: float = 0.72,
        window_size: int = 100,
        ticket_fee: float = 1.00,
        bps_fee: float = 0.0005,
        output_path: Optional[str] = None,
        adv: float = 50_000.0,
        # --- Three new optional optimisation components ---
        calibrator=None,          # PlattCalibratedModel for probability calibration
        vol_executor=None,        # VolatilityTargetedExecutor for adaptive sizing
        dynamic_cooldown=None,    # VolatilityScaledCooldown replaces static hysteresis
    ):
        self.csv_path = csv_path
        self.baseline_model = baseline_model
        self.candidate_model = candidate_model
        self.psi_threshold = psi_threshold
        self.auc_threshold = auc_threshold
        self.window_size = window_size
        self.ticket_fee = ticket_fee
        self.bps_fee = bps_fee
        self.output_path = output_path or csv_path.replace(".csv", "_sim_results.jsonl")
        self.adv = adv

        # Optimisation components (None = use legacy behaviour)
        self.calibrator       = calibrator
        self.vol_executor     = vol_executor
        self.dynamic_cooldown = dynamic_cooldown

        # Legacy static hysteresis (used when dynamic_cooldown is None)
        self.hysteresis = SwapHysteresisController(cooldown_ticks=window_size // 2)

        # Per-tick result storage
        self.tick_results: List[Dict] = []
        self.drift_checks: List[Dict] = []

        # v3 Fix: PSI reference distribution (first observed window, not uniform 0.5)
        # Using uniform [0.5, ...] as reference is mathematically wrong because:
        # PSI = sum((A_i - E_i) * ln(A_i/E_i)) where E is the EMPIRICAL baseline,
        # not an uninformative uniform prior. Industry standard = first-window distribution.
        self._reference_probs: Optional[List[float]] = None

    def _load_csv(self) -> List[Dict]:
        rows = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _active_model(self):
        return self.candidate_model if self._get_active_version() == "candidate" else self.baseline_model

    def _get_active_version(self) -> str:
        if self.dynamic_cooldown is not None:
            return self.dynamic_cooldown.active_version
        return self.hysteresis.active_version

    def _is_swap_allowed(self, current_tick: int, current_vol: float) -> bool:
        if self.dynamic_cooldown is not None:
            return self.dynamic_cooldown.is_swap_permitted(current_tick, current_vol)
        return self.hysteresis.is_swap_allowed()

    def _should_revert(self, current_psi: float, current_vol: float) -> bool:
        if self.dynamic_cooldown is not None:
            return self.dynamic_cooldown.should_revert_to_baseline(
                current_psi, self.psi_threshold, current_vol
            )
        return self.hysteresis.should_revert_to_baseline(current_psi, self.psi_threshold)

    def _do_swap(self, new_version: str, tick: int, psi: float,
                 current_vol: float = 0.0, auc: float = 0.0) -> None:
        if self.dynamic_cooldown is not None:
            self.dynamic_cooldown.execute_swap(new_version, tick, psi, current_vol, auc)
        else:
            self.hysteresis.execute_swap(new_version, tick, psi)
        # v3 Fix: Notify ActiveModelCalibrator of model change so probabilities
        # are always computed from the currently-active model, not always baseline.
        if self.calibrator is not None and hasattr(self.calibrator, "notify_swap"):
            self.calibrator.notify_swap(new_version)
        # Final Fix: Reset PSI reference after each swap so PSI measures
        # within-model distributional drift, not between-model calibrator differences.
        # Without this, PSI spikes every swap cycle because the candidate and
        # baseline calibrators have different mean probability outputs (~0.40 vs ~0.32).
        self._reference_probs = None

    def _cooldown_label(self) -> str:
        if self.dynamic_cooldown is not None:
            return "dynamic (vol-scaled)"
        return f"{self.hysteresis.cooldown_ticks} ticks (static)"

    def run(self) -> Dict:
        """Execute the full simulation and return a summary dict."""
        print(f"\n{'='*60}")
        print("  QUANTITATIVE DRIFT SIMULATION - START")
        print(f"{'='*60}")
        print(f"  CSV            : {self.csv_path}")
        print(f"  PSI threshold  : {self.psi_threshold}")
        print(f"  AUC threshold  : {self.auc_threshold}")
        print(f"  Window size    : {self.window_size} ticks")
        print(f"  Cooldown mode  : {self._cooldown_label()}")
        print(f"  Calibration    : {'Platt-calibrated' if self.calibrator else 'raw sigmoid'}")
        print(f"  Vol executor   : {'active' if self.vol_executor else 'fixed Q=100'}")
        print()

        rows = self._load_csv()
        n = len(rows)
        print(f"  Loaded {n} ticks from CSV\n")

        # Rolling feature history for drift detection
        prices: List[float] = []
        volumes: List[float] = []
        imbalances: List[float] = []
        prob_history: List[float] = []   # baseline model probabilities
        feature_history: List[List[float]] = []

        # Portfolio state
        portfolio_equity = 10_000.0  # Starting capital
        position = 0.0              # Shares held (can be fractional)
        entry_price = 0.0
        cash = portfolio_equity
        total_tx_costs = 0.0
        total_slippage = 0.0
        realised_pnl = 0.0
        max_equity = portfolio_equity
        max_drawdown = 0.0

        # Feature window for look_back
        LOOK_BACK = 20

        t_start = time.perf_counter()

        for idx, row in enumerate(rows):
            price = float(row.get("price", 100.0))
            bid = float(row.get("bid", price * 0.9995))
            ask = float(row.get("ask", price * 1.0005))
            vol = float(row.get("rolling_volatility", 0.0))
            imb = float(row.get("imbalance", 0.0))
            volume = float(row.get("volume", 1000.0))
            regime = row.get("regime", "unknown")

            prices.append(price)
            volumes.append(volume)
            imbalances.append(imb)

            # ── Build feature vector ───────────────────────────────────────
            from quant_finance.quant_model import build_features
            feat = build_features(prices, volumes, imbalances, look_back=LOOK_BACK)
            if feat is None:
                # Not enough history yet
                self.tick_results.append({
                    "tick": idx, "regime": regime, "price": price,
                    "status": "warming_up", "equity": round(cash, 2),
                })
                continue

            feature_history.append(feat)

            # ── Active model + probability ─────────────────────────────────
            active_model = self._active_model()

            # Calibrated probability takes precedence when available
            if self.calibrator is not None:
                try:
                    prob = self.calibrator.predict_calibrated_proba(feat)
                except RuntimeError:
                    prob = active_model.probability(feat)
            else:
                prob = active_model.probability(feat)

            prob_history.append(prob)

            # ── Signal + execution sizing ──────────────────────────────────
            # Vol-targeted executor produces adaptive Q + calibrated signal
            vol_ann = float(row.get("vol_ann", feat[2] * math.sqrt(252)))
            if self.vol_executor is not None:
                order_qty, signal = self.vol_executor.compute_execution_parameters(
                    vol_ann, prob
                )
            else:
                order_qty = 100.0
                signal = active_model.predict(feat)

            slippage = almgren_chriss_slippage(price, vol, order_size=order_qty, adv=self.adv)
            spread_cost = base_spread_cost(ask, bid)
            tx_cost = self.ticket_fee + (price * self.bps_fee)

            if signal == 1.0 and position <= 0:
                # Buy: enter long
                exec_price = ask + slippage
                cost = order_qty * exec_price + tx_cost
                if cash >= cost:
                    if position < 0:  # close short first
                        realised_pnl += (-position) * (entry_price - exec_price)
                    position = order_qty
                    entry_price = exec_price
                    cash -= cost
                    total_tx_costs += tx_cost
                    total_slippage += slippage * order_qty

            elif signal == -1.0 and position >= 0:
                # Sell: enter short
                exec_price = bid - slippage
                proceed = order_qty * exec_price - tx_cost
                if position > 0:  # close long first
                    realised_pnl += position * (exec_price - entry_price)
                position = -order_qty
                entry_price = exec_price
                cash += proceed
                total_tx_costs += tx_cost
                total_slippage += slippage * order_qty

            # Mark-to-market equity
            unrealised = position * (price - entry_price) if position != 0 else 0.0
            portfolio_equity = cash + position * price + unrealised
            max_equity = max(max_equity, portfolio_equity)
            drawdown = (max_equity - portfolio_equity) / max_equity if max_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            # Advance legacy hysteresis tick counter
            if self.dynamic_cooldown is None:
                self.hysteresis.tick()

            # ── Drift check every window_size ticks ────────────────────────
            check_result = None
            if len(prob_history) >= self.window_size and idx % self.window_size == 0:
                window_probs = prob_history[-self.window_size:]
                # v3 Fix: Use FIRST OBSERVED probability window as reference distribution.
                # This is the model's baseline behavioral distribution before any regime shift.
                # PSI then correctly measures distributional DRIFT rather than
                # distance from an uninformative uniform prior.
                if self._reference_probs is None:
                    self._reference_probs = list(window_probs)
                    print(f"  [REF]    tick={idx:>4} | PSI reference stored "
                          f"(mean_prob={sum(window_probs)/len(window_probs):.4f}, "
                          f"std={( sum((p-sum(window_probs)/len(window_probs))**2 for p in window_probs)/len(window_probs))**0.5:.4f})")
                psi = compute_psi(self._reference_probs, window_probs)

                # Adversarial AUC on features
                mid = len(feature_history) // 2
                auc = compute_adversarial_auc(feature_history[:mid], feature_history[mid:])

                drift_detected = psi >= self.psi_threshold or auc >= self.auc_threshold
                check_result = {
                    "tick":           idx,
                    "psi":            round(psi, 4),
                    "auc":            round(auc, 4),
                    "drift_detected": drift_detected,
                    "active_version": self._get_active_version(),
                    "regime":         regime,
                    "equity":         round(portfolio_equity, 2),
                    "current_vol":    round(vol_ann, 4),
                    "cooldown":       (
                        self.dynamic_cooldown.calculate_cooldown(vol_ann)
                        if self.dynamic_cooldown else self.hysteresis.cooldown_ticks
                    ),
                }

                if drift_detected and self._is_swap_allowed(idx, vol_ann):
                    self._do_swap("candidate", idx, psi, vol_ann, auc)
                    check_result["swap_executed"] = True
                    print(f"  [SWAP OK] tick={idx:>4} | PSI={psi:.4f} | AUC={auc:.3f} | "
                          f"Regime={regime} -> switched to CANDIDATE")
                elif (
                    self._get_active_version() == "candidate"
                    and self._should_revert(psi, vol_ann)
                ):
                    self._do_swap("baseline", idx, psi, vol_ann, auc)
                    check_result["swap_executed"] = True
                    print(f"  [REVERT] tick={idx:>4} | PSI={psi:.4f} | AUC={auc:.3f} | "
                          f"Regime={regime} -> reverted to BASELINE")
                else:
                    check_result["swap_executed"] = False
                    if drift_detected:
                        print(f"  [LOCK]   tick={idx:>4} | PSI={psi:.4f} | AUC={auc:.3f} | "
                              f"Drift detected but cooldown active")
                    else:
                        print(f"  [OK]     tick={idx:>4} | PSI={psi:.4f} | AUC={auc:.3f} | "
                              f"Regime={regime} | No drift")

                self.drift_checks.append(check_result)

            self.tick_results.append({
                "tick": idx,
                "regime": regime,
                "price": round(price, 4),
                "signal": signal,
                "probability": round(prob, 4),
                "slippage": round(slippage, 4),
                "tx_cost": round(tx_cost, 4),
                "position": round(position, 2),
                "equity": round(portfolio_equity, 2),
                "drawdown_pct": round(drawdown * 100, 2),
                "active_version": self._get_active_version(),
            })

        elapsed = time.perf_counter() - t_start

        # ── Final summary ─────────────────────────────────────────────────
        final_equity = portfolio_equity
        total_return_pct = (final_equity - 10_000.0) / 10_000.0 * 100.0
        n_swaps = len(
            self.dynamic_cooldown.swap_log if self.dynamic_cooldown
            else self.hysteresis.swap_log
        )
        swap_log = (
            self.dynamic_cooldown.swap_log if self.dynamic_cooldown
            else self.hysteresis.swap_log
        )

        summary = {
            "total_ticks": n,
            "processed_ticks": len([r for r in self.tick_results if r.get("status") != "warming_up"]),
            "simulation_elapsed_s": round(elapsed, 3),
            "starting_equity": 10_000.0,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "realised_pnl": round(realised_pnl, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "total_tx_costs": round(total_tx_costs, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "n_drift_checks": len(self.drift_checks),
            "n_hot_swaps": n_swaps,
            "swap_log": swap_log,
            "drift_checks": self.drift_checks,
        }

        # Persist results
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for r in self.tick_results:
                f.write(json.dumps(r) + "\n")

        print(f"\n  Summary written to: {self.output_path}")
        return summary
