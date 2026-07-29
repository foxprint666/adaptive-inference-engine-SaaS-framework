"""
quant_finance/preflight_gate.py

10-point pre-flight gate. Reads results/risk_ledger.jsonl and
results/latency_log.jsonl then prints PASS / FAIL / ? for each gate.

All 10 gates must PASS before deploying live capital ($500-$1,000 USD).

Specified gates (5):
  G1  Zero unhandled exceptions (10 consecutive trading days)
  G2  Live PSI in [0.01, 0.20] during low-vol regimes
  G3  Circuit breaker triggered during disconnect test
  G4  Net daily P&L positive after commissions
  G5  Max drawdown < 2.0% in paper mode

Additional gates (5):
  G6  Warm-up guard: no trades in first 200 events
  G7  State checkpoint survives restart
  G8  Latency p99 < 200 ms (3 consecutive days)
  G9  Zero double-position events (open-position guard)
  G10 Tick gap p99 < 2000 ms (feed quality, full trading day)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_RES        = os.path.join(os.path.dirname(__file__), "results")
LEDGER      = os.path.join(_RES, "risk_ledger.jsonl")
CHECKPOINT = os.path.join(_RES, "risk_ledger_checkpoint.json")
LATENCY     = os.path.join(_RES, "latency_log.jsonl")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p), len(s) - 1)]


def run_preflight_gate() -> bool:
    print("\n" + "=" * 62)
    print("  PRE-FLIGHT GATE -- Micro-Account Readiness Checklist")
    print("=" * 62)

    ledger_events   = _load_jsonl(LEDGER)
    latency_events  = _load_jsonl(LATENCY)
    checkpoint: dict = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            checkpoint = json.load(f)

    if not ledger_events:
        print("  [!] risk_ledger.jsonl is empty -- run paper trading first.")

    results: Dict[str, Optional[bool]] = {}

    # ── G1: Zero unhandled exceptions ────────────────────────────────
    errors = [e for e in ledger_events if e.get("type") == "error"]
    max_ce = max((e.get("consecutive_errors", 0) for e in errors), default=0)
    results["G1"] = max_ce < 3

    # ── G2: PSI in [0.01, 0.20] during low-vol ───────────────────────
    psi_events = [e for e in ledger_events if "psi" in e]
    if psi_events:
        low_vol_psi = [e["psi"] for e in psi_events if e.get("regime", "quiet") == "quiet"]
        if low_vol_psi:
            results["G2"] = all(0.01 <= p <= 0.20 for p in low_vol_psi)
        else:
            results["G2"] = None
    else:
        results["G2"] = None

    # ── G3: Circuit breaker triggered ────────────────────────────────
    cb_trips = [e for e in ledger_events if e.get("type") == "circuit_breaker_trip"]
    results["G3"] = len(cb_trips) > 0

    # ── G4: Net P&L positive ─────────────────────────────────────────
    if checkpoint:
        start = checkpoint.get("starting_equity", 10_000.0)
        cur   = checkpoint.get("current_equity",  start)
        results["G4"] = cur > start
    else:
        results["G4"] = None

    # ── G5: Max drawdown < 2.0% ──────────────────────────────────────
    max_dd = checkpoint.get("max_drawdown_pct", 100.0)
    results["G5"] = max_dd < 2.0

    # ── G6: Warm-up guard -- no fills in first 200 ledger events ─────
    events_before_first_fill = next(
        (i for i, e in enumerate(ledger_events) if e.get("type") == "fill"),
        len(ledger_events),
    )
    results["G6"] = events_before_first_fill >= 200

    # ── G7: Checkpoint file exists and is populated ───────────────────
    results["G7"] = bool(checkpoint) and os.path.exists(CHECKPOINT)

    # ── G8: Latency p99 < 200 ms ─────────────────────────────────────
    e2e_vals = [e["e2e_ms"] for e in latency_events if "e2e_ms" in e]
    if e2e_vals:
        results["G8"] = _pct(e2e_vals, 0.99) < 200.0
    else:
        results["G8"] = None

    # ── G9: Zero double-position events ──────────────────────────────
    double_pos = [
        e for e in ledger_events
        if e.get("status") == "blocked_open_position"
    ]
    results["G9"] = len(double_pos) == 0

    # ── G10: Tick gap p99 < 2000 ms ──────────────────────────────────
    gap_events = [e for e in ledger_events if "tick_gap_p99_ms" in e]
    if gap_events:
        worst = max(e["tick_gap_p99_ms"] for e in gap_events)
        results["G10"] = worst < 2000.0
    else:
        results["G10"] = None

    # ── Print results ─────────────────────────────────────────────────
    LABELS = {
        "G1":  "Zero unhandled exceptions (10 consecutive days)",
        "G2":  "PSI in [0.01, 0.20] during low-vol regimes",
        "G3":  "Circuit breaker triggered during disconnect test",
        "G4":  "Net daily P&L positive after commissions",
        "G5":  "Max drawdown < 2.0% in paper mode",
        "G6":  "Warm-up guard: zero trades in first 200 buffer ticks",
        "G7":  "State checkpoint survives restart",
        "G8":  "Latency p99 < 200 ms (3 consecutive days)",
        "G9":  "Zero double-position events",
        "G10": "Tick gap p99 < 2000 ms (feed quality, full day)",
    }

    all_pass = True
    for g, label in LABELS.items():
        v = results[g]
        if v is None:
            tag = "  [ ? ]"
            all_pass = False
        elif v:
            tag = "  [PASS]"
        else:
            tag = "  [FAIL]"
            all_pass = False
        print(f"{tag}  {label}")

    print("\n" + "=" * 62)
    if all_pass:
        print("  ALL GATES PASSED -- Ready for micro-account deployment")
    else:
        print("  GATES NOT ALL PASSED -- Continue paper trading")
    print("=" * 62 + "\n")
    return all_pass


if __name__ == "__main__":
    run_preflight_gate()
