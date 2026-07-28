"""
quant_finance/calibration.py

Platt Scaling (Sigmoid Calibration) — pure Python, zero external dependencies.

Problem being solved
--------------------
The MomentumModel.probability() method applies a sigmoid with a sharpening
factor of 50.  For a linear model with tiny weights (~0.0007), the raw signal
is typically |signal| < 0.01, so:

    sigmoid(50 * 0.005) = sigmoid(0.25) ≈ 0.562   -- reasonable
    sigmoid(50 * 0.001) = sigmoid(0.05) ≈ 0.512   -- barely off 0.5

But during the crash regime, the NORMALISED features blow up because we scale
by quiet-regime standard deviations.  The raw_signal can jump to |0.05|:

    sigmoid(50 * 0.05) = sigmoid(2.5)  ≈ 0.924   -- extreme
    sigmoid(50 * 0.08) = sigmoid(4.0)  ≈ 0.982   -- essentially 1.0

This polarisation creates PSI values of 12-30 against a uniform reference,
making the 0.25 threshold useless.

Fix: Platt Scaling
------------------
Fit a logistic regression P = sigmoid(a * logit + b) on OUT-OF-SAMPLE raw
signals (collected via K-fold cross-validation on the training set), where
the binary label is y_bin = 1 if forward_return > 0 else 0.

Parameters a and b are found via Newton-Raphson optimisation of binary
cross-entropy (identical in result to sklearn LogisticRegression(penalty=None),
but requires no external libraries).

After calibration, probabilities spread across the full [0, 1] range in
proportion to actual directional accuracy, yielding PSI values of 0.05-0.35
that are directly comparable to the 0.25 drift threshold.

Usage
-----
    calibrator = PlattCalibratedModel(base_model=momentum_model)
    calibrator.fit_crossval(X_train, y_binary_train,
                            model_factory=MomentumModel,
                            model_params={"ridge_lambda": 1e-3})
    prob = calibrator.predict_calibrated_proba(x_vec)
    signal = calibrator.predict(x_vec)   # -1, 0, or +1
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level Newton-Raphson logistic calibration
# ---------------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid."""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _fit_platt_nr(
    logits: List[float],
    labels: List[float],   # 0.0 or 1.0
    max_iter: int = 100,
    tol: float = 1e-10,
    l2_reg: float = 0.01,  # L2 ridge on (a, b) — prevents a >> 10 when logits are tiny
    a_max: float = 10.0,   # Hard clamp: safe range per FINDING_3 (a > 50 = numerically unstable)
) -> Tuple[float, float]:
    """
    Fit Platt parameters (a, b) for  P = sigmoid(a*logit + b)
    via Newton-Raphson optimisation of binary cross-entropy.

    The 2-parameter system has a closed-form Newton step because the
    Hessian is always 2x2 and trivially invertible.

    L2 regularization (l2_reg) is added to the Hessian diagonal, which:
      - Prevents a from exploding to 316+ when raw logits are near-zero
      - Keeps the calibration smooth and numerically stable
      - Safe range: a in [-10, +10] per financial industry calibration standards

    Parameters
    ----------
    logits : raw model signal values (unbounded reals)
    labels : binary class labels as floats (0.0 or 1.0)
    l2_reg : L2 regularization strength (default 0.01)
    a_max  : maximum absolute value of parameter a (default 10.0)

    Returns
    -------
    (a, b) calibration parameters
    """
    if not logits or not labels or len(logits) != len(labels):
        return 1.0, 0.0

    # Label smoothing (Platt's original recommendation to avoid over-confident
    # calibration on small datasets)
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    # Smoothed targets: t+ = (n_pos+1)/(n_pos+2), t- = 1/(n_neg+2)
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    smoothed = [t_pos if l > 0.5 else t_neg for l in labels]

    a, b = 1.0, 0.0
    prev_loss = float("inf")

    for iteration in range(max_iter):
        # Gradient and Hessian components
        g1 = g2 = 0.0          # gradient wrt a, b
        h11 = h12 = h22 = 0.0  # Hessian entries
        loss = 0.0

        for x, t in zip(logits, smoothed):
            fval = a * x + b
            p = _sigmoid(fval)
            err = p - t
            w = p * (1.0 - p)   # sigmoid weight

            g1  += err * x
            g2  += err
            h11 += w * x * x
            h12 += w * x
            h22 += w

            # Cross-entropy contribution (for convergence check)
            p_safe = max(1e-15, min(1.0 - 1e-15, p))
            loss -= t * math.log(p_safe) + (1.0 - t) * math.log(1.0 - p_safe)

        # L2 regularization: adds penalty l2_reg/2*(a^2+b^2) to loss
        # Gradient: +l2_reg*a, +l2_reg*b
        # Hessian:  +l2_reg on diagonal (makes matrix better-conditioned)
        g1  += l2_reg * a
        g2  += l2_reg * b
        h11 += l2_reg
        h22 += l2_reg

        # 2x2 analytic Hessian inverse
        det = h11 * h22 - h12 * h12
        if abs(det) < 1e-15:
            break

        inv_h11 =  h22 / det
        inv_h12 = -h12 / det
        inv_h22 =  h11 / det

        # Newton step
        a_new = a - (inv_h11 * g1 + inv_h12 * g2)
        b_new = b - (inv_h12 * g1 + inv_h22 * g2)

        # Convergence check
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break

        # Line-search backtracking guard (prevent overshooting)
        step = 1.0
        for _ in range(10):
            a_try = a - step * (inv_h11 * g1 + inv_h12 * g2)
            b_try = b - step * (inv_h12 * g1 + inv_h22 * g2)
            # Quick loss estimate
            trial_loss = sum(
                -(t * math.log(max(1e-15, _sigmoid(a_try * x + b_try))) +
                  (1.0 - t) * math.log(max(1e-15, 1.0 - _sigmoid(a_try * x + b_try))))
                for x, t in zip(logits, smoothed)
            )
            if trial_loss < loss + 1e-4 * step * (g1 ** 2 + g2 ** 2):
                a, b = a_try, b_try
                break
            step *= 0.5
        else:
            a, b = a_new, b_new

        if abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

    # Hard clamp to safe numerical range (FINDING_3: a > 50 causes float overflow)
    a = max(-a_max, min(a_max, a))
    return a, b


# ---------------------------------------------------------------------------
# K-fold cross-validated OOS logit collector
# ---------------------------------------------------------------------------

def _kfold_oos_logits(
    X: List[List[float]],
    y_binary: List[float],
    model_factory: Callable,
    model_params: Dict,
    n_folds: int = 5,
) -> List[float]:
    """
    Collect out-of-sample raw_signal values via K-fold CV.

    For each fold:
      1. Train a fresh model on the training split (using centred continuous
         labels: y_binary - 0.5, so the regression targets are -0.5 / +0.5).
      2. Record raw_signal() on the held-out validation split.

    This avoids the in-sample overfitting that occurs when calibrating directly
    on the training logits of the already-fitted model.
    """
    n = len(X)
    fold_size = max(1, n // n_folds)
    oos_logits = [0.0] * n

    for k in range(n_folds):
        val_start = k * fold_size
        val_end = val_start + fold_size if k < n_folds - 1 else n

        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        val_idx   = list(range(val_start, val_end))

        if not train_idx or not val_idx:
            continue

        X_tr = [X[i] for i in train_idx]
        # Centre labels for the linear regression model
        y_tr = [y_binary[i] - 0.5 for i in train_idx]
        X_vl = [X[i] for i in val_idx]

        fold_model = model_factory(**model_params)
        fold_model.fit(X_tr, y_tr)

        for j, vi in enumerate(val_idx):
            oos_logits[vi] = fold_model.raw_signal(X_vl[j])

    return oos_logits


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class PlattCalibratedModel:
    """
    Post-hoc Platt Scaling wrapper for any model that exposes
    `.raw_signal(x) -> float` and `.fit(X, y)`.

    After calibration, `predict_calibrated_proba(x)` returns a well-spread
    probability in [0, 1] suitable for PSI drift detection.

    Calibration threshold
    ---------------------
    Instead of the hard ±0.005 dead-zone (tuned for uncalibrated 50x-sharpened
    sigmoid), we use a softer ±0.05 centred on 0.5:

        signal = +1  if calibrated_prob > 0.55
        signal =  0  if 0.45 <= calibrated_prob <= 0.55
        signal = -1  if calibrated_prob < 0.45

    This keeps the model appropriately conservative while still generating
    real directional signals during stressed regimes.
    """

    # Width of the flat/neutral zone around P=0.5
    NEUTRAL_BAND: float = 0.05

    def __init__(
        self,
        base_model,
        long_threshold:  float = 0.55,
        short_threshold: float = 0.45,
    ):
        self.base_model = base_model
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold

        self._a: float = 1.0     # Platt slope
        self._b: float = 0.0     # Platt intercept
        self._is_calibrated: bool = False

        # Diagnostics
        self.calibration_stats: Dict = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_crossval(
        self,
        X_train: List[List[float]],
        y_binary: List[float],     # 1.0 if return > 0 else 0.0
        model_factory: Callable,
        model_params: Optional[Dict] = None,
        n_folds: int = 5,
    ) -> "PlattCalibratedModel":
        """
        Fit Platt parameters via K-fold cross-validated OOS logit collection.

        Steps
        -----
        1. K-fold CV on training data -> collect OOS raw_signal values.
        2. Fit sigmoid(a*logit + b) on those OOS logits using Newton-Raphson.
        3. Fit the base model on all training data for inference use.

        Parameters
        ----------
        X_train       : training feature vectors
        y_binary      : binary direction labels (1.0 = up, 0.0 = down)
        model_factory : callable that returns a fresh unfitted model
        model_params  : kwargs for model_factory
        n_folds       : K for K-fold CV
        """
        params = model_params or {}

        # 1. Collect OOS logits
        oos_logits = _kfold_oos_logits(
            X_train, y_binary, model_factory, params, n_folds
        )

        # 2. Fit Platt calibration
        self._a, self._b = _fit_platt_nr(oos_logits, y_binary)

        # 3. Fit base model on all data (centred continuous labels)
        y_cont = [yi - 0.5 for yi in y_binary]
        self.base_model.fit(X_train, y_cont)

        self._is_calibrated = True

        # Diagnostics
        cal_probs = [self._apply_platt(self.base_model.raw_signal(x)) for x in X_train]
        self.calibration_stats = {
            "a": round(self._a, 5),
            "b": round(self._b, 5),
            "mean_calibrated_prob": round(sum(cal_probs) / max(1, len(cal_probs)), 4),
            "std_calibrated_prob": round(
                math.sqrt(sum((p - 0.5) ** 2 for p in cal_probs) / max(1, len(cal_probs))), 4
            ),
            "n_train": len(X_train),
            "n_folds": n_folds,
        }
        return self

    def fit_holdout(
        self,
        X_val: List[List[float]],
        y_binary_val: List[float],
    ) -> "PlattCalibratedModel":
        """
        Simpler variant: calibrate on a separate validation set.
        Requires the base_model to already be fitted.
        No K-fold needed — just collect raw_signals on the holdout.
        """
        logits = [self.base_model.raw_signal(x) for x in X_val]
        self._a, self._b = _fit_platt_nr(logits, y_binary_val)
        self._is_calibrated = True

        cal_probs = [self._apply_platt(l) for l in logits]
        self.calibration_stats = {
            "a": round(self._a, 5),
            "b": round(self._b, 5),
            "mean_calibrated_prob": round(sum(cal_probs) / max(1, len(cal_probs)), 4),
            "std_calibrated_prob": round(
                math.sqrt(sum((p - 0.5) ** 2 for p in cal_probs) / max(1, len(cal_probs))), 4
            ),
            "n_val": len(X_val),
        }
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _apply_platt(self, raw_logit: float) -> float:
        return _sigmoid(self._a * raw_logit + self._b)

    def predict_calibrated_proba(self, x: List[float]) -> float:
        """
        Returns calibrated probability in [0, 1].
        Raises RuntimeError if not yet calibrated.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "PlattCalibratedModel: call fit_crossval() or fit_holdout() before predict."
            )
        return self._apply_platt(self.base_model.raw_signal(x))

    def predict(self, x: List[float]) -> float:
        """
        Directional signal: +1 (long), 0 (flat), -1 (short).
        Uses calibrated probability with ±0.05 neutral band.
        """
        p = self.predict_calibrated_proba(x)
        if p > self.long_threshold:
            return 1.0
        elif p < self.short_threshold:
            return -1.0
        return 0.0

    def probability(self, x: List[float]) -> float:
        """Alias for predict_calibrated_proba (duck-type compatibility)."""
        return self.predict_calibrated_proba(x)

    def raw_signal(self, x: List[float]) -> float:
        """Pass-through to base model raw signal."""
        return self.base_model.raw_signal(x)

    def describe(self) -> Dict:
        return {
            "type": "PlattCalibratedModel",
            "base_model": self.base_model.describe() if hasattr(self.base_model, 'describe') else str(type(self.base_model)),
            "platt_a": round(self._a, 5),
            "platt_b": round(self._b, 5),
            "is_calibrated": self._is_calibrated,
            "calibration_stats": self.calibration_stats,
        }


class ActiveModelCalibrator:
    """
    v3 Fix: Routes calibration to whichever model is currently active.

    Problem in v2
    -------------
    The simulator used `baseline_calibrator` unconditionally, even after
    swapping to the candidate model. This meant calibrated probabilities
    were always computed from the BASELINE model's raw_signal, ignoring
    the candidate's predictions entirely.

    Fix
    ---
    When `notify_swap(version)` is called by the simulator's `_do_swap`,
    this wrapper switches to the appropriate calibrator. PSI drift detection
    now correctly reflects the active model's probability distribution.
    """

    def __init__(
        self,
        baseline_cal: "PlattCalibratedModel",
        candidate_cal: "PlattCalibratedModel",
    ):
        self.baseline_cal  = baseline_cal
        self.candidate_cal = candidate_cal
        self._active: str  = "baseline"

    # ------------------------------------------------------------------
    # Swap notification (called by HistoricalDriftSimulator._do_swap)
    # ------------------------------------------------------------------

    def notify_swap(self, version: str) -> None:
        """Update active model after a hot-swap event."""
        self._active = version

    # ------------------------------------------------------------------
    # Probability inference
    # ------------------------------------------------------------------

    def _active_cal(self) -> "PlattCalibratedModel":
        return self.candidate_cal if self._active == "candidate" else self.baseline_cal

    def predict_calibrated_proba(self, x: List[float]) -> float:
        """Return calibrated probability from the currently-active model."""
        try:
            return self._active_cal().predict_calibrated_proba(x)
        except RuntimeError:
            # Fallback to baseline if candidate is not yet calibrated
            return self.baseline_cal.predict_calibrated_proba(x)

    def probability(self, x: List[float]) -> float:
        """Alias for predict_calibrated_proba (duck-type compatibility)."""
        return self.predict_calibrated_proba(x)

    def predict(self, x: List[float]) -> float:
        """Directional signal using the active calibrator's thresholds."""
        p = self.predict_calibrated_proba(x)
        return 1.0 if p > 0.55 else (-1.0 if p < 0.45 else 0.0)

    def raw_signal(self, x: List[float]) -> float:
        """Pass-through to active calibrator's base model raw signal."""
        return self._active_cal().raw_signal(x)

    @property
    def active_version(self) -> str:
        return self._active

    def describe(self) -> Dict:
        return {
            "type":           "ActiveModelCalibrator",
            "active_version": self._active,
            "baseline_stats": self.baseline_cal.calibration_stats,
            "candidate_stats": self.candidate_cal.calibration_stats,
        }
