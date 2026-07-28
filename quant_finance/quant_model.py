"""
quant_finance/quant_model.py

Standalone, dependency-light quantitative trading signal model.

Architecture
------------
Three model variants are implemented, selectable by ``MODEL_TYPE`` env-var:

  "momentum"    — A linear momentum/mean-reversion model calibrated by
                  ridge regression on lagged return features.

  "volatility"  — A regime-aware vol-targeting allocation model. Reduces
                  position size as realised volatility rises.

  "ensemble"    — A simple equal-weight blend of the above two signals.

Each model exposes:
    fit(X, y)           — calibrate on training data
    predict(x_vec)      — return a scalar signal in [-1, 1]
                          (-1 = short, 0 = flat, +1 = long)
    probability(x_vec)  — map the signal to a [0,1] probability
                          (used by the drift detection worker)
    describe()          — return a human-readable summary dict

No PyTorch/sklearn required. All math is stdlib + numpy.

Design Note (EWC regularisation analogy)
-----------------------------------------
The ``ridge_lambda`` parameter in the momentum model plays the same
conceptual role as the Elastic Weight Consolidation (EWC) lambda in the
neural-network case described in the architecture document:

  "Minimise new-task loss + λ * Σ F_i (θ_i - θ*_i)²"

Here the "Fisher" weight for each coefficient is approximated as its
inverse variance across the training window. This penalises large shifts
in important features during incremental re-calibration, preserving
performance on the quiet-period baseline while adapting to stress.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple


# ── Feature engineering ────────────────────────────────────────────────────

def build_features(
    prices: List[float],
    volumes: List[float],
    imbalances: List[float],
    look_back: int = 20,
) -> Optional[List[float]]:
    """
    Build a fixed-length feature vector from raw tick arrays.

    Features (in order)
    -------------------
    0  : short-term momentum  (5-tick return)
    1  : medium-term momentum (20-tick return)
    2  : rolling volatility   (std of 20-tick log-returns)
    3  : volume z-score       (volume vs 20-tick mean)
    4  : order imbalance      (EMA of imbalance)
    5  : vol-of-vol           (std of rolling 5-tick vols)

    Returns None if there are fewer than look_back prices.
    """
    n = len(prices)
    if n < look_back + 1:
        return None

    log_returns = [
        math.log(prices[i] / prices[i - 1])
        for i in range(n - look_back, n)
        if prices[i - 1] > 0
    ]

    if len(log_returns) < 5:
        return None

    # Short-term momentum (5-tick)
    mom_short = sum(log_returns[-5:])

    # Medium-term momentum (20-tick)
    mom_med = sum(log_returns)

    # Rolling volatility (std of 20 log-returns)
    mean_ret = sum(log_returns) / len(log_returns)
    vol = math.sqrt(sum((r - mean_ret) ** 2 for r in log_returns) / max(1, len(log_returns) - 1))

    # Volume z-score
    recent_vols = volumes[max(0, n - look_back):]
    vol_mean = sum(recent_vols) / max(1, len(recent_vols))
    vol_std = math.sqrt(
        sum((v - vol_mean) ** 2 for v in recent_vols) / max(1, len(recent_vols) - 1)
    ) + 1e-9
    vol_z = (volumes[-1] - vol_mean) / vol_std

    # EMA of imbalance (lambda=0.9)
    ema_imb = 0.0
    lam = 0.9
    for imb in imbalances[max(0, n - look_back):]:
        ema_imb = lam * ema_imb + (1 - lam) * imb

    # Vol-of-vol (std of 5-tick sub-window vols)
    sub_vols = []
    step = 4
    for start in range(0, len(log_returns) - step, step):
        sub = log_returns[start: start + step]
        if len(sub) < 2:
            continue
        sm = sum(sub) / len(sub)
        sv = math.sqrt(sum((r - sm) ** 2 for r in sub) / (len(sub) - 1))
        sub_vols.append(sv)
    vol_of_vol = (
        math.sqrt(sum((sv - sum(sub_vols) / len(sub_vols)) ** 2 for sv in sub_vols) / max(1, len(sub_vols) - 1))
        if len(sub_vols) > 1
        else 0.0
    )

    return [mom_short, mom_med, vol, vol_z, ema_imb, vol_of_vol]


# ── Ridge Regression (pure stdlib) ───────────────────────────────────────────

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    return [_dot(row, v) for row in A]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    rows = len(A)
    cols = len(A[0]) if rows else 0
    return [[A[r][c] for r in range(rows)] for c in range(cols)]


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    m = len(B[0])
    p = len(B)
    return [[sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m)] for i in range(n)]


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _cholesky_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve Ax = b using Cholesky decomposition.
    A must be symmetric positive definite.
    """
    n = len(b)
    # Cholesky: A = L L^T
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-15))
            else:
                L[i][j] = (A[i][j] - s) / (L[j][j] + 1e-15)

    # Forward substitution: L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / (L[i][i] + 1e-15)

    # Backward substitution: L^T x = y
    LT = _transpose(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(LT[i][k] * x[k] for k in range(i + 1, n))) / (LT[i][i] + 1e-15)

    return x


# ── Model Implementations ─────────────────────────────────────────────────────

class MomentumModel:
    """
    Ridge-regularised linear model mapping feature vectors to direction signals.

    EWC analogy: fisher_weights penalise coefficient drift from baseline,
    providing the same catastrophic-forgetting protection as the neural-net EWC.

    Signal interpretation
    ---------------------
    +1 → Go long  (expected positive return)
     0 → Flat     (insufficient signal)
    -1 → Go short (expected negative return)
    """

    def __init__(
        self,
        ridge_lambda: float = 1e-3,
        fisher_weights: Optional[List[float]] = None,
        baseline_weights: Optional[List[float]] = None,
        ewc_lambda: float = 0.5,
    ):
        self.ridge_lambda = ridge_lambda
        self.ewc_lambda = ewc_lambda
        self.weights: Optional[List[float]] = None
        self.bias: float = 0.0
        self.n_features: int = 6

        # EWC anchor
        self.fisher_weights = fisher_weights or []
        self.baseline_weights = baseline_weights or []

        self._feature_means: List[float] = [0.0] * self.n_features
        self._feature_stds: List[float] = [1.0] * self.n_features

    # ── Normalisation ─────────────────────────────────────────────────────

    def _fit_scaler(self, X: List[List[float]]) -> None:
        n, d = len(X), len(X[0])
        self._feature_means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        self._feature_stds = [
            math.sqrt(
                sum((X[i][j] - self._feature_means[j]) ** 2 for i in range(n)) / max(1, n - 1)
            ) + 1e-9
            for j in range(d)
        ]

    def _scale(self, x: List[float]) -> List[float]:
        return [(x[j] - self._feature_means[j]) / self._feature_stds[j] for j in range(len(x))]

    # ── Ridge + EWC fit ───────────────────────────────────────────────────

    def fit(self, X: List[List[float]], y: List[float]) -> "MomentumModel":
        """
        Fit via closed-form ridge regression:
            w = (X^T X + λI + λ_ewc * diag(F))^{-1} X^T y

        EWC term: λ_ewc * Σ F_i (w_i − w*_i)² is absorbed into the
        regularisation diagonal and the target shift.
        """
        self._fit_scaler(X)
        Xs = [self._scale(x) for x in X]
        n, d = len(Xs), len(Xs[0])

        # X^T X
        XtX = _mat_mul(_transpose(Xs), Xs)

        # Ridge diagonal: λI
        reg = _identity(d)
        for i in range(d):
            reg[i][i] = self.ridge_lambda

        # EWC diagonal augmentation
        fisher_diag = [0.0] * d
        if self.fisher_weights and self.baseline_weights and len(self.fisher_weights) == d:
            for i in range(d):
                fisher_diag[i] = self.ewc_lambda * self.fisher_weights[i]

        # Augmented system: (X^T X + λI + F_diag) w = X^T y + F_diag * w*
        A = [[XtX[i][j] + reg[i][j] + (fisher_diag[i] if i == j else 0.0)
              for j in range(d)] for i in range(d)]
        Xty = [sum(Xs[i][j] * y[i] for i in range(n)) for j in range(d)]

        # EWC target shift
        if self.fisher_weights and self.baseline_weights:
            for j in range(d):
                if j < len(self.baseline_weights):
                    Xty[j] += fisher_diag[j] * self.baseline_weights[j]

        self.weights = _cholesky_solve(A, Xty)
        self.bias = (sum(y) / n) - _dot(self.weights, [sum(Xs[i][j] for i in range(n)) / n for j in range(d)])

        # Compute Fisher weights (inverse variance of features) for future EWC
        self.fisher_weights = [
            1.0 / max(self._feature_stds[j] ** 2, 1e-9)
            for j in range(d)
        ]
        self.baseline_weights = list(self.weights)
        return self

    def raw_signal(self, x: List[float]) -> float:
        if self.weights is None:
            return 0.0
        xs = self._scale(x)
        return _dot(self.weights, xs) + self.bias

    def predict(self, x: List[float]) -> float:
        sig = self.raw_signal(x)
        if sig > 0.005:
            return 1.0
        elif sig < -0.005:
            return -1.0
        return 0.0

    def probability(self, x: List[float]) -> float:
        """Map raw signal to [0,1] via sigmoid."""
        sig = self.raw_signal(x)
        return 1.0 / (1.0 + math.exp(-sig * 50.0))  # scale factor 50 sharpens sigmoid

    def describe(self) -> Dict:
        return {
            "type": "MomentumModel",
            "ridge_lambda": self.ridge_lambda,
            "ewc_lambda": self.ewc_lambda,
            "weights": [round(w, 5) for w in (self.weights or [])],
            "bias": round(self.bias, 5),
            "features": ["mom_short", "mom_med", "vol", "vol_z", "ema_imb", "vol_of_vol"],
        }


class VolTargetingModel:
    """
    Volatility-targeting allocation model.

    Scales position size inversely to realised volatility so that
    portfolio vol remains near a target level. During crashes the signal
    shrinks toward zero, preventing large drawdowns.

    Signal (position fraction): σ_target / σ_realised, capped at ±1
    """

    def __init__(self, vol_target: float = 0.15, lookback: int = 20):
        self.vol_target = vol_target
        self.lookback = lookback
        self._calibrated_mu: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> "VolTargetingModel":
        """Calibrate expected drift from training returns."""
        self._calibrated_mu = sum(y) / max(1, len(y))
        return self

    def probability(self, x: List[float]) -> float:
        """
        x[2] is rolling_volatility (annualised).
        Map position size to probability: 0.5 + 0.5 * position_fraction
        """
        realised_vol = x[2]  # feature index 2 = vol
        if realised_vol < 1e-6:
            pos = 1.0 * (1 if self._calibrated_mu >= 0 else -1)
        else:
            # Raw position fraction
            pos = min(self.vol_target / realised_vol, 1.0)
            if self._calibrated_mu < 0:
                pos = -pos

        # Map from [-1, 1] to [0, 1]
        return 0.5 + 0.5 * pos

    def predict(self, x: List[float]) -> float:
        p = self.probability(x)
        return 1.0 if p > 0.55 else (-1.0 if p < 0.45 else 0.0)

    def describe(self) -> Dict:
        return {
            "type": "VolTargetingModel",
            "vol_target_annualised": self.vol_target,
            "calibrated_mu": round(self._calibrated_mu, 6),
        }


class EnsembleModel:
    """
    Equal-weight blend of MomentumModel and VolTargetingModel.
    """

    def __init__(self, **kwargs):
        self.momentum = MomentumModel(**{k: v for k, v in kwargs.items() if k in
                                         ("ridge_lambda", "fisher_weights", "baseline_weights", "ewc_lambda")})
        self.vol_target = VolTargetingModel()

    def fit(self, X: List[List[float]], y: List[float]) -> "EnsembleModel":
        self.momentum.fit(X, y)
        self.vol_target.fit(X, y)
        return self

    def probability(self, x: List[float]) -> float:
        p1 = self.momentum.probability(x)
        p2 = self.vol_target.probability(x)
        return 0.5 * (p1 + p2)

    def predict(self, x: List[float]) -> float:
        p = self.probability(x)
        return 1.0 if p > 0.55 else (-1.0 if p < 0.45 else 0.0)

    def describe(self) -> Dict:
        return {
            "type": "EnsembleModel",
            "momentum": self.momentum.describe(),
            "vol_target": self.vol_target.describe(),
        }


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(model_type: Optional[str] = None, **kwargs) -> object:
    mt = (model_type or os.getenv("MODEL_TYPE", "momentum")).lower()
    if mt == "momentum":
        return MomentumModel(**kwargs)
    elif mt == "volatility":
        return VolTargetingModel(**kwargs)
    elif mt == "ensemble":
        return EnsembleModel(**kwargs)
    raise ValueError(f"Unknown MODEL_TYPE: {mt!r}. Choose momentum | volatility | ensemble.")
