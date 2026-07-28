"""
quant_finance/walk_forward_validator.py

Walk-Forward Cross-Validation (WFCV) for the adaptive quant model.

Walk-forward validation is the gold standard for time-series model evaluation.
Unlike k-fold CV (which shuffles data), WFCV respects temporal ordering:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Fold 1: Train=[0..399]   Test=[400..499]  (quiet regime)      │
  │  Fold 2: Train=[0..699]   Test=[700..849]  (stress regime)     │
  │  Fold 3: Train=[0..999]   Test=[1000..1099] (crash regime)     │
  │  Fold 4: Train=[0..1299]  Test=[1300..1449] (recovery regime)  │
  └─────────────────────────────────────────────────────────────────┘

For each fold we compute:
  - Directional accuracy (% correct sign predictions)
  - Sharpe Ratio (annualised)
  - Maximum drawdown
  - PSI drift score of probabilities from that fold vs. baseline
  - Whether retraining would have been triggered

This produces a "multi-regime validation matrix" as recommended in the
architecture document's Mitigation 3 (Overfitting section).

Key insight: A model that ONLY works on the crash fold but fails on
quiet/recovery is overfitted to that crash. WFCV exposes this.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from quant_finance.quant_model import build_features, build_model
from quant_finance.ingress_sim import compute_psi, compute_adversarial_auc


# ── Fold definition ───────────────────────────────────────────────────────────

@dataclass
class WalkForwardFold:
    fold_id: int
    train_end: int
    test_start: int
    test_end: int
    label: str  # e.g. "quiet→stress", "stress→crash"


# ── Metrics ───────────────────────────────────────────────────────────────────

def _sharpe(returns: List[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / max(1, n - 1)) + 1e-9
    return (mean_r - risk_free / 252) / std_r * math.sqrt(252)


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / max(peak, 1e-9)
        max_dd = max(max_dd, dd)
    return max_dd


def _directional_accuracy(signals: List[float], actual_returns: List[float]) -> float:
    correct = sum(
        1 for s, r in zip(signals, actual_returns)
        if (s > 0 and r > 0) or (s < 0 and r < 0)
    )
    n = len([s for s in signals if s != 0.0])
    return correct / max(1, n)


# ── Walk-Forward Validator ─────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Runs WFCV on the full synthetic dataset.

    Each fold trains a fresh model on the expanding training window,
    then evaluates on the held-out test window, computing
    Sharpe, drawdown, accuracy, and drift metrics.
    """

    LOOK_BACK = 20

    def __init__(
        self,
        rows: List[Dict],
        model_type: str = "momentum",
        psi_threshold: float = 0.25,
        auc_threshold: float = 0.72,
        calibrator=None,    # v3: PlattCalibratedModel or ActiveModelCalibrator
        vol_executor=None,  # v3: VolatilityTargetedExecutor for adaptive sizing
    ):
        self.rows = rows
        self.model_type = model_type
        self.psi_threshold = psi_threshold
        self.auc_threshold = auc_threshold
        self.calibrator   = calibrator    # v3 addition
        self.vol_executor = vol_executor  # v3 addition

    def _build_folds(self) -> List[WalkForwardFold]:
        """
        Construct regime-aware folds from regime labels in the data.
        Boundaries are the transition points between regimes.
        """
        n = len(self.rows)
        transitions = [0]
        for i in range(1, n):
            if self.rows[i]["regime"] != self.rows[i - 1]["regime"]:
                transitions.append(i)
        transitions.append(n)

        folds = []
        for i in range(1, len(transitions) - 1):
            fold = WalkForwardFold(
                fold_id=i,
                train_end=transitions[i],
                test_start=transitions[i],
                test_end=transitions[i + 1],
                label=f"{self.rows[transitions[i-1]]['regime']}->{self.rows[transitions[i]]['regime']}",
            )
            folds.append(fold)

        return folds

    def _extract_features_and_labels(
        self, start: int, end: int
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """Extract feature vectors, forward returns, and regime labels for a slice."""
        slice_rows = self.rows[max(0, start - self.LOOK_BACK): end]
        prices = [float(r["price"]) for r in slice_rows]
        volumes = [float(r.get("volume", 1000)) for r in slice_rows]
        imbalances = [float(r.get("imbalance", 0)) for r in slice_rows]
        regimes = [r.get("regime", "unknown") for r in slice_rows]

        X, y, regime_out = [], [], []
        offset = max(0, start - self.LOOK_BACK)

        for i in range(self.LOOK_BACK, len(slice_rows) - 1):
            feat = build_features(
                prices[: i + 1], volumes[: i + 1], imbalances[: i + 1], look_back=self.LOOK_BACK
            )
            if feat is None:
                continue
            fwd_ret = math.log(prices[i + 1] / prices[i]) if prices[i] > 0 else 0.0
            X.append(feat)
            y.append(fwd_ret)
            regime_out.append(regimes[i])

        return X, y, regime_out

    def run(self) -> Dict:
        """Run all folds and return the validation matrix."""
        folds = self._build_folds()
        results = []

        print(f"\n{'='*60}")
        print("  WALK-FORWARD CROSS-VALIDATION - START")
        print(f"  Model type : {self.model_type}")
        print(f"  Total folds: {len(folds)}")
        print(f"{'='*60}\n")

        # Cumulative baseline probabilities for PSI reference
        baseline_probs_global: List[float] = []

        for fold in folds:
            print(f"  Fold {fold.fold_id}: {fold.label}")
            print(f"    Train: ticks [0..{fold.train_end}]  Test: [{fold.test_start}..{fold.test_end}]")

            # ── Train ───────────────────────────────────────────────────────
            X_train, y_train, _ = self._extract_features_and_labels(0, fold.train_end)
            if len(X_train) < 10:
                print(f"    ⚠ Insufficient training data ({len(X_train)} samples) — skipping\n")
                continue

            model = build_model(self.model_type)
            model.fit(X_train, y_train)

            # ── Test ────────────────────────────────────────────────────────
            X_test, y_test, regimes_test = self._extract_features_and_labels(
                fold.test_start, fold.test_end
            )
            if len(X_test) < 5:
                print(f"    ⚠ Insufficient test data ({len(X_test)} samples) — skipping\n")
                continue

            # v3: Use calibrated probabilities when calibrator is available
            if self.calibrator is not None:
                probs = [
                    self.calibrator.predict_calibrated_proba(x)
                    for x in X_test
                ]
            else:
                probs = [model.probability(x) for x in X_test]

            # v3: Use vol-targeted executor for signals + adaptive sizing
            # Feature index 2 is rolling vol; annualise it for the executor
            signals = []
            order_qtys = []
            for xi in X_test:
                vol_ann = xi[2] * (252 ** 0.5)
                prob_i  = (
                    self.calibrator.predict_calibrated_proba(xi)
                    if self.calibrator is not None
                    else model.probability(xi)
                )
                if self.vol_executor is not None:
                    qty, sig = self.vol_executor.compute_execution_parameters(vol_ann, prob_i)
                else:
                    qty = 100.0
                    sig = model.predict(xi)
                signals.append(sig)
                order_qtys.append(qty)

            # ── P&L simulation (vol-adjusted) ─────────────────────────────
            equity = 10_000.0
            equity_curve = [equity]
            trade_returns = []
            position = 0.0
            entry = 0.0
            TICKET_FEE = 1.0

            for i, (sig, ret) in enumerate(zip(signals, y_test)):
                price   = float(self.rows[min(fold.test_start + i, len(self.rows) - 1)]["price"])
                qty     = order_qtys[i]
                if sig == 1.0 and position <= 0:
                    if position < 0:
                        realised = abs(position) * (entry - price)
                        equity += realised - TICKET_FEE
                    position = qty
                    entry    = price
                    equity  -= TICKET_FEE
                elif sig == -1.0 and position >= 0:
                    if position > 0:
                        realised = position * (price - entry)
                        equity  += realised - TICKET_FEE
                    position = -qty
                    entry    = price
                    equity  -= TICKET_FEE

                mark         = position * (price - entry) if position != 0 else 0.0
                portfolio_val = equity + position * price + mark
                equity_curve.append(portfolio_val)
                trade_returns.append(ret * (1 if position > 0 else (-1 if position < 0 else 0)))

            # ── Drift detection: v3 PSI fix ──────────────────────────────
            # Reference = training-window probability distribution for THIS fold.
            # Industry standard (Basel III / CCAR): compare production scores
            # to the empirical DEVELOPMENT SAMPLE distribution, not to a uniform prior.
            # This produces PSI in the 0-0.25 range for stable regimes and
            # spikes cleanly for genuine drift.
            train_probs_ref = [
                (
                    self.calibrator.predict_calibrated_proba(x)
                    if self.calibrator is not None
                    else model.probability(x)
                )
                for x in X_train[-len(probs):]
            ]
            if len(train_probs_ref) < len(probs):
                # Pad with the available training probs if train set is smaller
                train_probs_ref = train_probs_ref + train_probs_ref[:len(probs)-len(train_probs_ref)]
            psi = compute_psi(train_probs_ref, probs)
            baseline_probs_global.extend(probs)  # keep for legacy accumulation

            mid = len(X_test) // 2
            auc = compute_adversarial_auc(X_test[:mid], X_test[mid:])

            drift_triggered = psi >= self.psi_threshold or auc >= self.auc_threshold
            n_long  = signals.count(1.0)
            n_short = signals.count(-1.0)
            n_flat  = signals.count(0.0)

            # ── Metrics ────────────────────────────────────────────────────
            dir_acc = _directional_accuracy(signals, y_test)
            sharpe  = _sharpe(trade_returns)
            max_dd  = _max_drawdown(equity_curve)
            fold_return = (equity_curve[-1] - 10_000.0) / 10_000.0 * 100.0

            fold_result = {
                "fold_id":              fold.fold_id,
                "label":               fold.label,
                "n_train":             len(X_train),
                "n_test":              len(X_test),
                "directional_accuracy":round(dir_acc, 4),
                "sharpe_ratio":        round(sharpe, 3),
                "max_drawdown_pct":    round(max_dd * 100, 2),
                "fold_return_pct":     round(fold_return, 2),
                "psi":                 round(psi, 4),
                "adversarial_auc":     round(auc, 4),
                "drift_triggered":     drift_triggered,
                "n_long":              n_long,
                "n_short":             n_short,
                "n_flat":              n_flat,
                "final_equity":        round(equity_curve[-1], 2),
                "calibrated":          self.calibrator is not None,
                "vol_executor_active": self.vol_executor is not None,
            }
            results.append(fold_result)


            # Print fold summary
            drift_flag = "!! DRIFT DETECTED" if drift_triggered else "OK stable"
            print(f"    Train samples : {len(X_train):>5}")
            print(f"    Test samples  : {len(X_test):>5}")
            print(f"    Dir. accuracy : {dir_acc*100:.1f}%")
            print(f"    Sharpe ratio  : {sharpe:>+.3f}")
            print(f"    Max drawdown  : {max_dd*100:.1f}%")
            print(f"    Fold return   : {fold_return:>+.2f}%")
            print(f"    PSI           : {psi:.4f}  AUC: {auc:.4f}  [{drift_flag}]")
            print()

        validation_matrix = {
            "model_type": self.model_type,
            "psi_threshold": self.psi_threshold,
            "auc_threshold": self.auc_threshold,
            "folds": results,
            "summary": {
                "mean_sharpe": round(sum(r["sharpe_ratio"] for r in results) / max(1, len(results)), 3),
                "mean_accuracy": round(sum(r["directional_accuracy"] for r in results) / max(1, len(results)), 4),
                "mean_max_dd_pct": round(sum(r["max_drawdown_pct"] for r in results) / max(1, len(results)), 2),
                "folds_with_drift": sum(1 for r in results if r["drift_triggered"]),
                "total_folds": len(results),
            },
        }
        return validation_matrix
