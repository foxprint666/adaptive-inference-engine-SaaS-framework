"""
quant_finance/pretrained_baseline.py

Multi-Cycle Pre-Trained Baseline with Online EWC and PIT Boundary Enforcement.

Why pre-training matters for FIM stability
------------------------------------------
With only 979 quiet-regime samples, the Fisher Information Matrix (FIM)
diagonal computed by the baseline model captures localized sample noise
rather than structural market behaviour.  A feature that happens to be
predictive in this particular quiet window gets an artificially inflated
Fisher value, causing EWC to aggressively lock that weight — preventing
meaningful adaptation during a stress transition.

This module generates a synthetic MULTI-CYCLE historical baseline
(3 complete economic cycles = ~7,500 ticks) and trains on the first
two cycles (5,000 ticks).  The FIM computed over two full cycles is
dramatically more stable: it averages out sample noise and captures
genuine structural feature importances across regimes.

Point-in-Time (PIT) boundary enforcement
------------------------------------------
The simulation window (cycle 3, ticks 5000-7499) is strictly held out
from all pre-training.  The PIT split is recorded in the returned
PretrainedInfo dataclass and enforced at the caller level in
run_validation.py.

Online EWC Fisher decay
------------------------
Rather than computing a single FIM at the end of training, we process
the pre-training data in rolling batches and update Fisher values with
exponential decay:

    F_new = decay * F_old + (1 - decay) * F_batch

This produces a Fisher matrix that is:
  - Weighted towards RECENT historical patterns (more relevant)
  - Retaining memory of multi-cycle structural importances
  - Less likely to over-regularise noisy features
  - Less likely to over-protect features that were transiently important

decay = 0.90 means the half-life of historical Fisher information is
approximately 7 batches (~3,500 ticks at batch_size=500).
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from quant_finance.market_regime_generator import generate_market_data
from quant_finance.quant_model import (
    MomentumModel,
    build_features,
)

# Local regime params dict (mirrors market_regime_generator RegimeSpec values)
_REGIME_PARAMS: dict = {
    "quiet": {
        "mu": 0.08 / 252, "sigma": 0.012 * (252 ** 0.5),
        "nu": 30, "spread_bps": 5, "gap_prob": 0.0,
        "garch_alpha": 0.05, "garch_beta": 0.90,
    },
    "stress": {
        "mu": -0.05 / 252, "sigma": 0.022 * (252 ** 0.5),
        "nu": 10, "spread_bps": 15, "gap_prob": 0.005,
        "garch_alpha": 0.08, "garch_beta": 0.88,
    },
    "crash": {
        "mu": -0.35 / 252, "sigma": 0.060 * (252 ** 0.5),
        "nu": 3, "spread_bps": 50, "gap_prob": 0.04,
        "garch_alpha": 0.15, "garch_beta": 0.80,
    },
    "recovery": {
        "mu": 0.18 / 252, "sigma": 0.035 * (252 ** 0.5),
        "nu": 8, "spread_bps": 20, "gap_prob": 0.002,
        "garch_alpha": 0.06, "garch_beta": 0.87,
    },
}


# ---------------------------------------------------------------------------
# Dataclass: metadata returned to the caller
# ---------------------------------------------------------------------------

@dataclass
class PretrainedInfo:
    """
    Carries the pre-trained model and all associated metadata so the
    orchestrator can wire it into the simulation without re-training.
    """
    model:              MomentumModel
    fisher_weights:     List[float]       # Online-EWC decayed Fisher diagonal
    baseline_weights:   List[float]       # Final trained weights
    n_cycles:           int
    n_train_ticks:      int
    n_sim_ticks:        int
    pit_split_tick:     int               # First tick index belonging to OOS window
    train_csv_path:     str
    sim_csv_path:       str
    regime_composition: Dict[str, int]
    fisher_decay:       float
    batch_size:         int
    training_metadata:  Dict             = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic multi-cycle data generation
# ---------------------------------------------------------------------------

_MULTI_CYCLE_REGIMES: List[Tuple[str, int]] = [
    # 3 full economic cycles, each with the same regime sequence
    # but with different RNG seeds so statistics vary per-cycle
    ("quiet",    1000),
    ("stress",    500),
    ("crash",     300),
    ("recovery",  700),
    ("quiet",    1000),  # cycle 2 begins
    ("stress",    500),
    ("crash",     300),
    ("recovery",  700),
    ("quiet",    1000),  # cycle 3 = OOS simulation window
    ("stress",    500),
    ("crash",     300),
    ("recovery",  700),
]


def generate_multicycle_data(
    output_csv: str,
    seed: int = 7,
) -> Tuple[str, str, int]:
    """
    Generate 3 full economic cycles of synthetic tick data.

    Returns
    -------
    (full_csv_path, pit_tick_index, total_ticks)
    The PIT boundary sits at the start of cycle 3 (tick 5000).
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Use the existing single-regime generator repeatedly
    all_rows: List[Dict] = []
    rng = random.Random(seed)

    for regime_name, n_ticks in _MULTI_CYCLE_REGIMES:
        # Vary the seed per segment so each regime instance is statistically
        # distinct rather than a perfect repeat
        seg_seed = rng.randint(1, 99_999)

        # Build a single-regime spec for the shared generator
        spec: List[Tuple[str, int]] = [(regime_name, n_ticks)]
        seg_rows = _generate_regime_segment(regime_name, n_ticks, seg_seed,
                                            start_price=all_rows[-1]["price"] if all_rows else 100.0)
        all_rows.extend(seg_rows)

    # PIT boundary: cycle 3 starts at tick 5000
    pit_tick = 5000

    # Write full CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return output_csv, pit_tick, len(all_rows)


def _generate_regime_segment(
    regime: str,
    n_ticks: int,
    seed: int,
    start_price: float = 100.0,
) -> List[Dict]:
    """
    Generate a single-regime segment matching the REGIMES spec from
    market_regime_generator, but starting at a given price.
    """
    rng = random.Random(seed)

    params = _REGIME_PARAMS.get(regime, _REGIME_PARAMS["quiet"])
    mu    = params["mu"]
    sigma = params["sigma"]
    nu    = params.get("nu", 30)
    spread_bps = params.get("spread_bps", 5)
    gap_prob   = params.get("gap_prob", 0.0)
    garch_alpha = params.get("garch_alpha", 0.05)
    garch_beta  = params.get("garch_beta", 0.90)

    dt = 1.0 / 252.0  # one tick = 1 trading day
    price = start_price

    # GARCH(1,1) parameters
    omega = (sigma / (252 ** 0.5)) ** 2 * (1 - garch_alpha - garch_beta)
    alpha = garch_alpha
    beta  = garch_beta
    h = (sigma / (252 ** 0.5)) ** 2   # initial conditional variance

    rows: List[Dict] = []
    for _ in range(n_ticks):
        # GARCH update
        z = _student_t_sample(rng, nu)
        h = omega + alpha * (math.sqrt(h) * z) ** 2 + beta * h
        vol_t = math.sqrt(max(h, 1e-10))

        # Log-return
        lr = mu * dt + vol_t * math.sqrt(dt) * z

        # Gap event (crash regime only)
        if gap_prob > 0 and rng.random() < gap_prob:
            lr += rng.gauss(-0.04, 0.01)

        price = max(0.01, price * math.exp(lr))

        spread = price * spread_bps / 10_000.0
        volume = rng.lognormvariate(math.log(50_000), 0.5)
        imbalance = rng.gauss(0.0, 0.3)

        rows.append({
            "price":    round(price, 4),
            "bid":      round(price - spread / 2, 4),
            "ask":      round(price + spread / 2, 4),
            "volume":   round(volume, 0),
            "imbalance":round(imbalance, 4),
            "vol_ann":  round(vol_t * math.sqrt(252), 4),
            "regime":   regime,
        })

    return rows


def _student_t_sample(rng: random.Random, nu: float) -> float:
    """Sample from Student-t(nu) via ratio of normals method."""
    z = rng.gauss(0.0, 1.0)
    if nu >= 1000:
        return z
    v = 2.0 * rng.gammavariate(nu / 2.0, 1.0) / nu
    return z / math.sqrt(v)


# ---------------------------------------------------------------------------
# Online EWC Fisher update
# ---------------------------------------------------------------------------

def _compute_batch_fisher(
    model: MomentumModel,
    X_batch: List[List[float]],
    y_batch: List[float],
) -> List[float]:
    """
    Compute an approximation of the Fisher diagonal for one data batch.

    For a linear regression model, the Fisher diagonal is proportional to
    the feature-wise mean squared gradient of the log-likelihood:

        F_j approx= (1/n) * sum_i (residual_i * x_scaled_ij)^2

    This is equivalent to the feature-wise second moment of the score function,
    which is the standard Fisher approximation used in EWC.
    """
    if not X_batch or not model.weights:
        return [1.0] * 6

    n = len(X_batch)
    d = len(model.weights)
    f_diag = [0.0] * d

    for xi, yi in zip(X_batch, y_batch):
        xs = model._scale(xi)
        pred = sum(model.weights[j] * xs[j] for j in range(d)) + model.bias
        residual = pred - yi
        for j in range(d):
            f_diag[j] += (residual * xs[j]) ** 2

    # Average and take absolute value (Fisher is always positive)
    return [abs(f_diag[j]) / n for j in range(d)]


def online_ewc_train(
    model:         MomentumModel,
    X_train:       List[List[float]],
    y_train:       List[float],
    batch_size:    int   = 500,
    fisher_decay:  float = 0.90,
) -> List[float]:
    """
    Train the model in sequential batches, updating the Fisher diagonal
    with exponential decay after each batch.

    This implements Online EWC:
        F_t = decay * F_{t-1} + (1 - decay) * F_batch

    Returns the final decayed Fisher diagonal.
    """
    n = len(X_train)
    if n == 0:
        return [1.0] * 6

    # Initial full fit to get starting weights
    model.fit(X_train, y_train)
    fisher = list(model.fisher_weights) or [1.0] * 6

    # Sequential batch updates
    n_batches = max(1, n // batch_size)
    for b in range(n_batches):
        start = b * batch_size
        end   = start + batch_size if b < n_batches - 1 else n

        X_b = X_train[start:end]
        y_b = y_train[start:end]

        if len(X_b) < 10:
            continue

        # Re-fit model on this batch (with EWC anchoring to previous weights)
        model.fisher_weights   = fisher
        model.baseline_weights = list(model.weights or [0.0] * 6)
        model.fit(X_b, y_b)

        # Compute batch Fisher
        f_batch = _compute_batch_fisher(model, X_b, y_b)

        # EWC online update: F_new = decay * F_old + (1 - decay) * F_batch
        fisher = [
            fisher_decay * fisher[j] + (1.0 - fisher_decay) * f_batch[j]
            for j in range(len(fisher))
        ]

    # Normalize so Fisher values are on a sensible scale
    max_f = max(fisher) if any(f > 0 for f in fisher) else 1.0
    fisher = [f / max(max_f, 1e-9) for f in fisher]

    return fisher


# ---------------------------------------------------------------------------
# Main builder: load or build pre-trained baseline
# ---------------------------------------------------------------------------

def build_pretrained_baseline(
    data_dir: str,
    look_back: int = 20,
    batch_size: int = 500,
    fisher_decay: float = 0.90,
    seed: int = 7,
    force_rebuild: bool = False,
) -> PretrainedInfo:
    """
    Generate multi-cycle synthetic data, train the baseline on cycles 1-2,
    and return the PretrainedInfo with a strict PIT boundary at tick 5000.

    Parameters
    ----------
    data_dir      : Directory for data files
    look_back     : Feature window size
    batch_size    : Online EWC batch size
    fisher_decay  : EWC Fisher decay factor (0.90 = ~7-batch half-life)
    seed          : RNG seed for reproducibility
    force_rebuild : If True, regenerate data even if CSV exists
    """
    os.makedirs(data_dir, exist_ok=True)
    full_csv    = os.path.join(data_dir, "pretrain_multicycle_data.csv")
    sim_csv     = os.path.join(data_dir, "synthetic_market_data.csv")

    # ── Step 1: Generate multi-cycle data ─────────────────────────────────
    if not os.path.exists(full_csv) or force_rebuild:
        print(f"  [PRETRAIN] Generating 3-cycle market data -> {full_csv}")
        generate_multicycle_data(full_csv, seed=seed)
    else:
        print(f"  [PRETRAIN] Loaded existing multi-cycle data: {full_csv}")

    # ── Step 2: Load and split at PIT boundary ────────────────────────────
    all_rows: List[Dict] = []
    with open(full_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    pit_tick = 5000
    train_rows = all_rows[:pit_tick]   # cycles 1-2: pre-training
    sim_rows   = all_rows[pit_tick:]   # cycle 3: OOS simulation

    print(f"  [PRETRAIN] PIT split: {len(train_rows)} train ticks | "
          f"{len(sim_rows)} OOS simulation ticks")

    # Write the simulation CSV (used by HistoricalDriftSimulator)
    if sim_rows:
        fieldnames = list(sim_rows[0].keys())
        with open(sim_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sim_rows)
        print(f"  [PRETRAIN] Simulation CSV written: {sim_csv}")

    # ── Step 3: Build feature matrix from training rows ───────────────────
    prices     = [float(r["price"])    for r in train_rows]
    volumes    = [float(r.get("volume", "1000")) for r in train_rows]
    imbalances = [float(r.get("imbalance", "0")) for r in train_rows]

    X_train: List[List[float]] = []
    y_train: List[float]       = []

    for i in range(look_back, len(prices) - 1):
        feat = build_features(
            prices[:i+1], volumes[:i+1], imbalances[:i+1], look_back=look_back
        )
        if feat is None:
            continue
        fwd = math.log(prices[i+1] / prices[i]) if prices[i] > 0 else 0.0
        X_train.append(feat)
        y_train.append(fwd)

    print(f"  [PRETRAIN] Feature matrix: {len(X_train)} samples, "
          f"{len(X_train[0]) if X_train else 0} features")

    # ── Step 4: Online EWC training with Fisher decay ─────────────────────
    model = MomentumModel(ridge_lambda=1e-3, ewc_lambda=0.8)
    print(f"  [PRETRAIN] Running Online EWC training "
          f"(batch={batch_size}, decay={fisher_decay})...")

    fisher = online_ewc_train(model, X_train, y_train, batch_size, fisher_decay)

    # Attach the final Fisher to the model
    model.fisher_weights   = fisher
    model.baseline_weights = list(model.weights or [0.0] * 6)

    # ── Step 5: Compute regime composition ────────────────────────────────
    regime_composition: Dict[str, int] = {}
    for r in train_rows:
        reg = r.get("regime", "unknown")
        regime_composition[reg] = regime_composition.get(reg, 0) + 1

    # ── Step 6: Training metadata ─────────────────────────────────────────
    feature_names = ["mom_short", "mom_med", "vol", "vol_z", "ema_imb", "vol_of_vol"]
    metadata = {
        "weights":       [round(w, 6) for w in (model.weights or [])],
        "bias":          round(model.bias, 6),
        "fisher":        [round(f, 6) for f in fisher],
        "feature_names": feature_names,
        "fisher_stability": {
            name: round(f, 4)
            for name, f in zip(feature_names, fisher)
        },
        "top_ewc_locked_features": [
            feature_names[i]
            for i in sorted(range(len(fisher)), key=lambda j: fisher[j], reverse=True)[:3]
        ],
    }

    print(f"  [PRETRAIN] Done. Weights: "
          f"{[round(w, 5) for w in (model.weights or [])]}")
    print(f"  [PRETRAIN] Fisher diagonal (normalised): "
          f"{[round(f, 4) for f in fisher]}")
    print(f"  [PRETRAIN] Top EWC-locked features: "
          f"{metadata['top_ewc_locked_features']}")

    return PretrainedInfo(
        model              = model,
        fisher_weights     = fisher,
        baseline_weights   = list(model.weights or []),
        n_cycles           = 2,           # trained on 2 cycles
        n_train_ticks      = len(train_rows),
        n_sim_ticks        = len(sim_rows),
        pit_split_tick     = pit_tick,
        train_csv_path     = full_csv,
        sim_csv_path       = sim_csv,
        regime_composition = regime_composition,
        fisher_decay       = fisher_decay,
        batch_size         = batch_size,
        training_metadata  = metadata,
    )
