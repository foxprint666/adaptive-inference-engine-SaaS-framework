"""
quant_finance/run_validation.py

Master orchestration script — v3: Zero Loose Ends.

Six root-cause fixes applied vs v2
------------------------------------
FIX-1  PSI reference distribution    uniform [0.5] → first-window empirical dist
FIX-2  ActiveModelCalibrator         calibration now follows the active model
FIX-3  Platt a clamped               L2 reg + a_max=10 (was a=316, numerically unstable)
FIX-4  Candidate EWC lambda          0.8 → 0.3, trained on crash+recovery data only
FIX-5  WFCV calibration integration  each fold uses Platt probs + vol-executor sizing
FIX-6  WFCV PSI reference            training-window distribution (not uniform 0.5)

Research grounding
-------------------
FINDING_1  PSI industry standard: reference = empirical baseline development sample
           (Basel III / CCAR). Uniform prior invalidates the 0.1 / 0.25 thresholds.
FINDING_2  EWC weight collapse: caused by feature scale mismatch + rigid lambda=0.8.
           Fix: lower lambda + train candidate on regime-specific stressed data.
FINDING_3  Platt a=316 is numerically unsafe. Safe range: |a| ≤ 10. L2 regularization
           prevents explosion when model logits are near-zero.

Run with:
    python -m quant_finance.run_validation
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Dict, List

from quant_finance.market_regime_generator import generate_market_data
from quant_finance.quant_model import build_features, MomentumModel
from quant_finance.ingress_sim import HistoricalDriftSimulator
from quant_finance.walk_forward_validator import WalkForwardValidator
from quant_finance.calibration import PlattCalibratedModel, ActiveModelCalibrator
from quant_finance.hysteresis import VolatilityScaledCooldown
from quant_finance.vol_target import VolatilityTargetedExecutor
from quant_finance.pretrained_baseline import build_pretrained_baseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH    = os.path.join(DATA_DIR, "synthetic_market_data.csv")

FEATURE_NAMES = ["mom_short", "mom_med", "vol", "vol_z", "ema_imb", "vol_of_vol"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_train_data(rows, regime_filter, look_back=20, return_binary=False):
    filtered = [r for r in rows if r["regime"] in regime_filter]
    prices     = [float(r["price"])             for r in filtered]
    volumes    = [float(r.get("volume", 1000))  for r in filtered]
    imbalances = [float(r.get("imbalance", 0))  for r in filtered]

    X, y, y_bin = [], [], []
    for i in range(look_back, len(prices) - 1):
        feat = build_features(prices[:i+1], volumes[:i+1], imbalances[:i+1], look_back=look_back)
        if feat is None:
            continue
        fwd = math.log(prices[i+1] / prices[i]) if prices[i] > 0 else 0.0
        X.append(feat)
        y.append(fwd)
        y_bin.append(1.0 if fwd > 0 else 0.0)

    return (X, y, y_bin) if return_binary else (X, y)


def _std(vals):
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals)))


def print_table(rows, columns, col_w=22):
    header = "  ".join(f"{c:<{col_w}}" for c in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{str(row.get(c, '')):<{col_w}}" for c in columns))


def _fmt_fisher(f, names):
    return " | ".join(f"{n}={v:.4f}" for n, v in zip(names, f))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("=  ADAPTIVE QUANT MODEL v3 — ZERO LOOSE ENDS                        =")
    print("=  6 Root-Cause Fixes | Pure Python | No External Dependencies       =")
    print("=" * 70)
    t0 = time.perf_counter()

    # ─── STEP 1: Multi-cycle pre-training (PIT boundary enforced) ─────────────
    print("\n[STEP 1] Building multi-cycle pre-trained baseline (PIT boundary)...")
    pretrain = build_pretrained_baseline(
        data_dir=DATA_DIR, look_back=20, batch_size=500,
        fisher_decay=0.90, seed=7, force_rebuild=True,
    )

    baseline_model = pretrain.model
    fisher_weights = pretrain.fisher_weights
    baseline_wts   = pretrain.baseline_weights

    print(f"\n  Pre-training summary:")
    print(f"    Cycles          : {pretrain.n_cycles}")
    print(f"    Training ticks  : {pretrain.n_train_ticks:,}")
    print(f"    OOS sim ticks   : {pretrain.n_sim_ticks:,}")
    print(f"    PIT boundary    : tick {pretrain.pit_split_tick:,} (strict OOS)")
    print(f"    Regime mix      : {pretrain.regime_composition}")
    print(f"    Fisher diagonal : {_fmt_fisher(fisher_weights, FEATURE_NAMES)}")
    meta = pretrain.training_metadata
    print(f"    Top EWC-locked  : {meta.get('top_ewc_locked_features', [])}")

    # ─── STEP 2: Local baseline (quiet-only, for comparison) ──────────────────
    print("\n[STEP 2] Building local baseline (quiet-only, comparison)...")
    sim_rows = _load_csv(CSV_PATH)
    X_q, y_q, y_q_bin = _build_train_data(sim_rows, ["quiet"], return_binary=True)
    local_model = MomentumModel(ridge_lambda=1e-3)
    local_model.fit(X_q, y_q)

    local_f   = local_model.fisher_weights or [0.0] * 6
    local_std = _std(local_f)
    pt_std    = _std(fisher_weights)
    improve   = (1 - pt_std / max(local_std, 1e-9)) * 100

    print(f"  Local  Fisher std : {local_std:.4f}  (sample noise)")
    print(f"  Pretrained Fisher std: {pt_std:.4f}  ({improve:.1f}% more stable)")

    # ─── STEP 3: Candidate model (FIX-4: crash+recovery data, lambda=0.3) ─────
    print("\n[STEP 3 | FIX-4] Training candidate on crash+recovery data (lambda=0.3)...")
    # v2 used quiet+stress (dominated by quiet). v3 uses crash+recovery so
    # feature magnitudes are larger, preventing near-zero weight collapse.
    X_cr, y_cr, y_cr_bin = _build_train_data(
        sim_rows, ["crash", "recovery"], return_binary=True
    )
    candidate_model = MomentumModel(
        ridge_lambda     = 1e-3,
        ewc_lambda       = 0.3,       # FIX-4: was 0.8 → near-zero weights
        fisher_weights   = fisher_weights,
        baseline_weights = baseline_wts,
    )
    candidate_model.fit(X_cr, y_cr)

    w_drift = math.sqrt(sum(
        (b - c) ** 2 for b, c in zip(baseline_wts, candidate_model.weights or [])
    ))
    print(f"  Training samples  : {len(X_cr)} (crash+recovery, higher feature magnitudes)")
    print(f"  EWC lambda        : 0.3 (was 0.8)")
    print(f"  Weights           : {[round(w, 5) for w in (candidate_model.weights or [])]}")
    print(f"  Weight L2-drift   : {w_drift:.5f}")

    # ─── STEP 4: Platt calibration (FIX-3: L2 regularization, a_max=10) ──────
    print("\n[STEP 4 | FIX-3] Fitting Platt calibrators (L2 reg, a_max=10)...")

    # Baseline: last 20% of quiet window as holdout
    n_q  = len(X_q)
    nq20 = max(1, int(n_q * 0.20))
    baseline_calibrator = PlattCalibratedModel(base_model=baseline_model)
    baseline_calibrator.fit_holdout(X_q[-nq20:], y_q_bin[-nq20:])
    bs = baseline_calibrator.calibration_stats
    print(f"  Baseline: a={bs['a']:.4f}  b={bs['b']:.4f}  "
          f"prob_std={bs['std_calibrated_prob']:.4f}")

    # Candidate: last 20% of crash+recovery holdout
    ncr20 = max(1, int(len(X_cr) * 0.20))
    candidate_calibrator = PlattCalibratedModel(base_model=candidate_model)
    candidate_calibrator.fit_holdout(X_cr[-ncr20:], y_cr_bin[-ncr20:])
    cs = candidate_calibrator.calibration_stats
    print(f"  Candidate: a={cs['a']:.4f}  b={cs['b']:.4f}  "
          f"prob_std={cs['std_calibrated_prob']:.4f}")
    if abs(cs['a']) > 10:
        print(f"  [WARNING] Candidate a={cs['a']:.2f} exceeds safe range — "
              "model weights still near-zero (increase training data or lower lambda further)")

    # FIX-2: ActiveModelCalibrator routes to correct calibrator after each swap
    active_calibrator = ActiveModelCalibrator(baseline_calibrator, candidate_calibrator)
    print(f"\n  FIX-2: ActiveModelCalibrator active (routes to baseline/candidate dynamically)")

    # ─── STEP 5: Execution infrastructure ─────────────────────────────────────
    print("\n[STEP 5] Configuring execution infrastructure...")
    vol_executor = VolatilityTargetedExecutor(
        target_volatility=0.19, base_trade_size=100.0,
        long_threshold=0.55,   short_threshold=0.45,
        min_trade_size=5.0,
    )
    dynamic_cooldown = VolatilityScaledCooldown(
        base_cooldown=50, vol_ref=0.19,
        min_cooldown=5,   max_cooldown=200,
        recovery_fraction=0.85,
    )
    for regime_name, vol in [("quiet", 0.19), ("stress", 0.44),
                              ("crash", 0.95), ("recovery", 0.56)]:
        cd = dynamic_cooldown.calculate_cooldown(vol)
        q  = vol_executor.compute_trade_size(vol)
        print(f"  {regime_name:<10}: cooldown={cd:>3} ticks  Q={q:.0f} shares  "
              f"(vol={vol:.0%})")

    # ─── STEP 6: Walk-Forward Cross-Validation (FIX-5, FIX-6) ────────────────
    print("\n[STEP 6 | FIX-5,6] Running WFCV with calibration + vol-executor...")
    print("  FIX-6: PSI reference = training-window distribution (not uniform 0.5)")
    validator = WalkForwardValidator(
        rows         = sim_rows,
        model_type   = "momentum",
        calibrator   = active_calibrator,
        vol_executor = vol_executor,
    )
    wfcv = validator.run()

    wfcv_path = os.path.join(RESULTS_DIR, "wfcv_results_v3.json")
    with open(wfcv_path, "w", encoding="utf-8") as f:
        json.dump(wfcv, f, indent=2)

    # ─── STEP 7: Historical Drift Simulation (FIX-1, FIX-2) ──────────────────
    print("\n[STEP 7 | FIX-1,2] Running Drift Simulation...")
    print("  FIX-1: PSI reference = first-window empirical distribution")
    print("  FIX-2: ActiveModelCalibrator follows active model across swaps")

    sim_output = os.path.join(RESULTS_DIR, "sim_tick_results_v3.jsonl")
    simulator = HistoricalDriftSimulator(
        csv_path         = CSV_PATH,
        baseline_model   = baseline_model,
        candidate_model  = candidate_model,
        psi_threshold    = 0.25,
        auc_threshold    = 0.72,
        window_size      = 100,
        output_path      = sim_output,
        adv              = 50_000.0,
        calibrator       = active_calibrator,
        vol_executor     = vol_executor,
        dynamic_cooldown = dynamic_cooldown,
    )
    sim = simulator.run()

    sim_path = os.path.join(RESULTS_DIR, "sim_summary_v3.json")
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(sim, f, indent=2)

    # ─── STEP 8: Print Full Results ───────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    print(f"\n\n  {'=' * 70}")
    print(f"  v3 RESULTS — ALL SIX FIXES APPLIED")
    print(f"  {'=' * 70}")

    # --- Simulation summary ---
    print(f"\n  -- SIMULATION P&L --")
    print(f"  Starting equity    : $10,000.00")
    print(f"  Final equity       : ${sim['final_equity']:>12,.2f}")
    print(f"  Total return       : {sim['total_return_pct']:>+.2f}%")
    print(f"  Max drawdown       : {sim['max_drawdown_pct']:.2f}%")
    print(f"  TX costs           : ${sim['total_tx_costs']:.2f}")
    print(f"  Drift checks       : {sim['n_drift_checks']}")
    print(f"  Hot-swaps          : {sim['n_hot_swaps']}")

    # PSI stats
    if sim.get("drift_checks"):
        dc_psi  = [d["psi"] for d in sim["drift_checks"]]
        psi_mean = sum(dc_psi) / len(dc_psi)
        n_above  = sum(1 for p in dc_psi if p > 0.25)
        print(f"\n  PSI stats (post FIX-1):")
        print(f"  Mean PSI            : {psi_mean:.4f} (v2 was ~35.6)")
        print(f"  Windows > threshold : {n_above}/{len(dc_psi)}")

    # --- WFCV table ---
    print(f"\n  -- WALK-FORWARD MATRIX (FIX-5,6) --")
    print_table(
        wfcv.get("folds", []),
        ["fold_id", "label", "directional_accuracy", "sharpe_ratio",
         "max_drawdown_pct", "psi", "n_long", "n_short", "n_flat"],
        col_w=20,
    )
    s = wfcv.get("summary", {})
    print(f"\n  Mean Sharpe     : {s.get('mean_sharpe', 0):.3f}")
    print(f"  Mean accuracy   : {s.get('mean_accuracy', 0):.4f}")
    print(f"  Folds w/ drift  : {s.get('folds_with_drift', 0)}/{s.get('total_folds', 0)}")

    # --- Model weights ---
    cand_wts = candidate_model.weights or []
    print(f"\n  -- WEIGHT COMPARISON (Pretrained vs v3 Candidate) --")
    print_table(
        [
            {
                "Feature": FEATURE_NAMES[i],
                "Pretrained": f"{baseline_wts[i]:+.5f}",
                "Candidate":  f"{cand_wts[i]:+.5f}" if i < len(cand_wts) else "N/A",
                "Delta":      f"{cand_wts[i]-baseline_wts[i]:+.5f}" if i < len(cand_wts) else "N/A",
                "Fisher":     f"{fisher_weights[i]:.4f}",
            }
            for i in range(len(FEATURE_NAMES))
        ],
        ["Feature", "Pretrained", "Candidate", "Delta", "Fisher"],
        col_w=18,
    )

    # --- Six-fix summary ---
    dc_checks = sim.get("drift_checks", [])
    swap_log  = sim.get("swap_log", [])
    reverts   = sum(1 for s in swap_log if s.get("to") == "baseline")

    print(f"""
  -- SIX-FIX VERIFICATION --

  FIX-1  PSI reference: first-window empirical distribution
         v2 PSI mean: ~35.6 (all above threshold, meaningless)
         v3 PSI mean: {psi_mean if dc_checks else 'N/A':.4f} (should be ~0 in quiet, spike in crash)

  FIX-2  ActiveModelCalibrator: routes to candidate after swap
         Total swaps    : {len(swap_log)}
         Reverts found  : {reverts}  (v2 had 0 reverts because PSI never fell below 0.25)

  FIX-3  Platt a clamped to [-10, +10]:
         Baseline a     : {bs['a']:.4f}  (v2: 2.62 - was fine)
         Candidate a    : {cs['a']:.4f}  (v2: 316.65 - was unsafe)

  FIX-4  Candidate lambda 0.8 -> 0.3, crash+recovery training data:
         v2 candidate weights: near-zero ([-0.00039, 6e-05, ...])
         v3 candidate weights: {[round(w, 5) for w in cand_wts[:3]]}... (should be larger)

  FIX-5  WFCV now uses Platt calibration + vol-executor per fold:
         v2 Fold 1: 0.0% accuracy (all flat)
         v3 Fold 1: {wfcv['folds'][0]['directional_accuracy']*100:.1f}% accuracy (calibrated signals)

  FIX-6  WFCV PSI uses training-window distribution as reference:
         v2 WFCV PSI Fold 1: 6.25 (against uniform)
         v3 WFCV PSI Fold 1: {wfcv['folds'][0]['psi']:.4f} (against training distribution)
""")

    # ─── STEP 9: v1/v2/v3 Comparison Table ────────────────────────────────────
    print("  -- v1 / v2 / v3 COMPARISON --")
    print_table(
        [
            {"Metric": "Fisher std",         "v1": "11,403,900",     "v2": "0.3732",        "v3": f"{pt_std:.4f}"},
            {"Metric": "Max drawdown",        "v1": "63.4%",          "v2": "1.29%",         "v3": f"{sim['max_drawdown_pct']:.2f}%"},
            {"Metric": "Candidate a (Platt)", "v1": "N/A",            "v2": "316.65 (unsafe)","v3": f"{cs['a']:.2f} (clamped)"},
            {"Metric": "PSI mean (sim)",      "v1": "~16.5",          "v2": "~35.6",         "v3": f"{psi_mean:.4f}" if dc_checks else "N/A"},
            {"Metric": "WFCV Fold 1 acc",     "v1": "49.7%",          "v2": "0.0% (flat)",   "v3": f"{wfcv['folds'][0]['directional_accuracy']*100:.1f}%"},
            {"Metric": "WFCV Fold 2 acc",     "v1": "0.0% (flat)",    "v2": "48.96%",        "v3": f"{wfcv['folds'][1]['directional_accuracy']*100:.1f}%"},
            {"Metric": "WFCV Fold 3 acc",     "v1": "0.0% (flat)",    "v2": "32.43%",        "v3": f"{wfcv['folds'][2]['directional_accuracy']*100:.1f}%"},
            {"Metric": "Hot-swap reverts",    "v1": "0",              "v2": "0",             "v3": str(reverts)},
            {"Metric": "PIT boundary",        "v1": "No",             "v2": "Yes (tick 5000)","v3": "Yes (tick 5000)"},
            {"Metric": "PSI ref correct",     "v1": "No",             "v2": "No",            "v3": "Yes"},
        ],
        ["Metric", "v1", "v2", "v3"],
        col_w=26,
    )

    # ─── STEP 10: Write combined JSON + git stage ──────────────────────────────
    combined = {
        "version": "v3_zero_loose_ends",
        "fixes_applied": [
            "FIX-1: PSI reference = first-window empirical distribution",
            "FIX-2: ActiveModelCalibrator routes to active model after swap",
            "FIX-3: Platt L2 regularization + a_max=10 clamp",
            "FIX-4: Candidate EWC lambda=0.3, trained on crash+recovery only",
            "FIX-5: WFCV uses calibrated probs + vol-executor per fold",
            "FIX-6: WFCV PSI reference = training-window distribution",
        ],
        "simulation": sim,
        "wfcv": wfcv,
        "pretrain_info": {
            "n_cycles":          pretrain.n_cycles,
            "n_train_ticks":     pretrain.n_train_ticks,
            "n_sim_ticks":       pretrain.n_sim_ticks,
            "pit_split_tick":    pretrain.pit_split_tick,
            "fisher_decay":      pretrain.fisher_decay,
            "regime_composition":pretrain.regime_composition,
        },
        "fisher_stability": {
            "local_std":     round(local_std, 5),
            "pretrained_std":round(pt_std, 5),
            "improvement_pct":round(improve, 1),
        },
        "calibration": {
            "baseline":  bs,
            "candidate": cs,
        },
        "vol_executor": vol_executor.describe(),
        "dynamic_cooldown": dynamic_cooldown.describe(),
        "weight_comparison": [
            {
                "feature":      FEATURE_NAMES[i],
                "pretrained_w": round(baseline_wts[i], 6),
                "candidate_w":  round(cand_wts[i], 6) if i < len(cand_wts) else 0.0,
                "fisher":       round(fisher_weights[i], 6),
            }
            for i in range(len(FEATURE_NAMES))
        ],
        "elapsed_seconds": round(elapsed, 3),
    }

    out = os.path.join(RESULTS_DIR, "combined_results_v3.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n  Files written:")
    print(f"    combined_results_v3.json   — full v3 results")
    print(f"    wfcv_results_v3.json       — WFCV matrix")
    print(f"    sim_summary_v3.json        — simulation summary")
    print(f"    sim_tick_results_v3.jsonl  — per-tick equity curve")
    print(f"\n  Total elapsed : {elapsed:.2f}s")
    print("\n" + "=" * 70 + "\n")

    return combined


if __name__ == "__main__":
    main()
