"""
quant_finance/live_engine.py

Main async event loop.

State machine
-------------
  WARMUP       -> READY        (200-tick buffer filled)
  READY        -> CIRCUIT_OPEN (circuit breaker trips)
  READY        -> RECONNECTING (feed dead)
  RECONNECTING -> READY        (feed restored)
  CIRCUIT_OPEN -> READY        (daily reset at midnight)

Run modes
---------
  RUN_MODE=REPLAY   Synthetic CSV replay  (default, no credentials needed)
  RUN_MODE=LIVE     Live Alpaca WebSocket  (requires env vars below)

Required env vars for LIVE mode
--------------------------------
  APCA_API_KEY_ID        Alpaca paper trading API key
  APCA_API_SECRET_KEY    Alpaca paper trading secret
  APCA_SYMBOL            Ticker (default: SPY)

Latency budget (Phase 3)
------------------------
  p50 e2e < 50 ms   (pre-staged limit orders bring this to ~25 ms)
  p99 e2e < 200 ms  (REST round-trip dominates; limit orders help)
  All deltas use time.perf_counter_ns() -- NTP-safe monotonic clock.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from enum import Enum


class EngineState(Enum):
    WARMUP       = "WARMUP"
    READY        = "READY"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    RECONNECTING = "RECONNECTING"


async def run_engine(
    symbol:          str   = "SPY",
    starting_equity: float = 10_000.0,
    run_mode:        str   = None,
    replay_csv:      str   = None,
) -> None:
    """
    Main live engine entry point.

    Parameters
    ----------
    symbol          : Ticker to trade (default SPY)
    starting_equity : Paper account starting equity
    run_mode        : "LIVE" or "REPLAY" (overrides RUN_MODE env var)
    replay_csv      : Optional CSV path for replay mode
    """
    run_mode   = (run_mode or os.environ.get("RUN_MODE", "REPLAY")).upper()
    api_key    = os.environ.get("APCA_API_KEY_ID",     "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")

    print("\n" + "=" * 62)
    print(f"  ADAPTIVE QUANT ENGINE  v3 | {run_mode} mode | {symbol}")
    print(f"  Equity: ${starting_equity:,.0f}")
    print(f"  IEX WebSocket feed  (~2-3% of US consolidated volume)")
    print(f"  Limit orders: ask+$0.01 buy | bid-$0.01 sell")
    print(f"  Latency clock: time.perf_counter_ns() (NTP-safe)")
    print("=" * 62)

    # ── Load pre-trained model ─────────────────────────────────────
    from quant_finance.pretrained_baseline import build_pretrained_baseline
    from quant_finance.calibration import PlattCalibratedModel, ActiveModelCalibrator
    from quant_finance.vol_target import VolatilityTargetedExecutor
    from quant_finance.quant_model import MomentumModel
    from quant_finance.run_validation import _load_csv, _build_train_data

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    CSV_PATH = os.path.join(DATA_DIR, "synthetic_market_data.csv")

    print("\n[INIT] Loading pre-trained model...")
    pretrain    = build_pretrained_baseline(data_dir=DATA_DIR, look_back=20, force_rebuild=False)
    base_model  = pretrain.model
    fisher_wts  = pretrain.fisher_weights
    base_wts    = pretrain.baseline_weights

    sim_rows = _load_csv(CSV_PATH)
    X_q, y_q, y_q_bin = _build_train_data(sim_rows, ["quiet"], return_binary=True)
    nq = max(1, int(len(X_q) * 0.20))
    base_cal = PlattCalibratedModel(base_model=base_model)
    base_cal.fit_holdout(X_q[-nq:], y_q_bin[-nq:])

    X_cr, y_cr, y_cr_bin = _build_train_data(
        sim_rows, ["crash", "recovery"], return_binary=True
    )
    cand_model = MomentumModel(
        ridge_lambda=1e-3, ewc_lambda=0.3,
        fisher_weights=fisher_wts, baseline_weights=base_wts,
    )
    cand_model.fit(X_cr, y_cr)
    ncr = max(1, int(len(X_cr) * 0.20))
    cand_cal = PlattCalibratedModel(base_model=cand_model)
    cand_cal.fit_holdout(X_cr[-ncr:], y_cr_bin[-ncr:])

    active_cal = ActiveModelCalibrator(base_cal, cand_cal)
    vol_exec   = VolatilityTargetedExecutor(
        target_volatility=0.19, base_trade_size=100.0,
        long_threshold=0.55,    short_threshold=0.45,
        min_trade_size=1.0,
    )
    print("[INIT] Model ready.")

    # ── Initialise components ──────────────────────────────────────
    from quant_finance.circuit_breaker import CircuitBreaker, OrderProposal
    from quant_finance.feature_buffer  import FeatureBuffer
    from quant_finance.muse_router     import MuseRouter
    from quant_finance.risk_ledger     import RiskLedger
    from quant_finance.latency_logger  import LatencyLogger

    buffer  = FeatureBuffer(window=200)
    breaker = CircuitBreaker(
        starting_equity=starting_equity, dd_limit_pct=0.02, max_qty=100
    )
    router  = MuseRouter(active_cal, vol_exec, symbol=symbol)
    ledger  = RiskLedger(starting_equity=starting_equity)
    latlog  = LatencyLogger()

    state = EngineState.WARMUP

    # ── Gateway ────────────────────────────────────────────────────
    gateway = None
    if run_mode == "LIVE":
        from quant_finance.broker_gateway import BrokerGateway
        gateway = BrokerGateway(api_key=api_key, api_secret=api_secret)

    async def _on_dead_feed() -> None:
        nonlocal state
        state = EngineState.RECONNECTING
        breaker.set_position_open(False)
        if gateway:
            await gateway.cancel_all_orders()
        print("  [ENGINE] Dead feed handled. Waiting for reconnect.")

    # ── Feed ───────────────────────────────────────────────────────
    if run_mode == "LIVE":
        from quant_finance.live_feed import LiveFeed
        feed   = LiveFeed(
            symbol=symbol, api_key=api_key, api_secret=api_secret,
            on_dead_feed=_on_dead_feed,
        )
        stream = feed.stream()
    else:
        from quant_finance.tick_replay import TickReplay
        csv_r  = replay_csv or CSV_PATH
        replay = TickReplay(csv_path=csv_r, speed=0.0, symbol=symbol)
        stream = replay.stream()

    # ── Main loop ──────────────────────────────────────────────────
    print(f"\n[ENGINE] Running {'LIVE' if run_mode == 'LIVE' else 'REPLAY'}...\n")
    tick_n       = 0
    position_side: str   = ""
    entry_price:   float = 0.0

    async for tick in stream:
        tick_n += 1
        t0_ns = tick.timestamp_ns   # t0: set by live_feed/tick_replay

        # Feed restored after reconnect
        if state == EngineState.RECONNECTING:
            state = EngineState.READY
            print(f"  [ENGINE] Feed restored at tick {tick_n}")

        # ── Feature extraction (t1) ────────────────────────────────
        feat = buffer.push(tick)
        t1_ns = time.perf_counter_ns()   # t1 checkpoint

        if feat is None:
            if tick_n % 50 == 0:
                print(f"  [ENGINE] Warm-up {buffer.fill_pct:.0f}% "
                      f"({tick_n}/200 ticks)")
            continue

        # Transition: WARMUP -> READY
        if state == EngineState.WARMUP:
            state = EngineState.READY
            breaker.set_buffer_ready()
            print(f"  [ENGINE] READY -- warm-up complete at tick {tick_n}")

        # ── Model prediction (t2) ──────────────────────────────────
        vol_ann = buffer.latest_vol_ann
        order   = await router.route(tick, feat, vol_ann)
        t2_ns   = time.perf_counter_ns() if not order else order.signal_time_ns

        if order is None:
            latlog.record(t0_ns, t1_ns, time.perf_counter_ns(), had_order=False)
            continue

        # ── Circuit breaker ────────────────────────────────────────
        proposal = OrderProposal(
            symbol=order.symbol, side=order.side, qty=order.qty,
            limit_price=order.limit_price,
            signal_time_ns=order.signal_time_ns,
            model_prob=order.signal_prob,
        )
        ok, reason = breaker.approve(proposal)
        if not ok:
            if tick_n % 500 == 0:  # Throttled console log (every 500 ticks)
                print(f"  [BREAKER] {reason}")
            if "TRIPPED" in reason:
                state = EngineState.CIRCUIT_OPEN
                ledger.record_circuit_trip(reason)
            latlog.record(t0_ns, t1_ns, t2_ns, had_order=False)
            continue

        mid = (tick.bid + tick.ask) / 2.0 if tick.ask > 0 else tick.price

        # ── Execute ────────────────────────────────────────────────
        if run_mode == "LIVE" and gateway:
            # --- Live mode: Alpaca paper REST ---
            fill = await gateway.submit(order, mid_at_signal=mid)
            t3   = fill.fill_time_ns

            if fill.status not in ("error", "blocked_open_position",) and \
               not fill.status.startswith("error"):

                if position_side and position_side != order.side:
                    # Position exit / reversal filled!
                    print(f"  [LIVE] EXIT/REVERSE order filled | {order.side.upper()} {order.qty}x {symbol} @ ${fill.fill_price:.2f}")
                    position_side = ""
                    entry_price   = 0.0
                    breaker.set_position_open(False)
                else:
                    # New position entry filled!
                    position_side = order.side
                    entry_price   = fill.fill_price or mid
                    breaker.set_position_open(True, side=order.side)
                    print(f"  [LIVE] ENTRY order filled | {order.side.upper()} {order.qty}x {symbol} @ ${entry_price:.2f}")

                ledger.record_fill(fill, vol_ann=vol_ann)

                # Update equity from account (every order)
                try:
                    acct   = await gateway.get_account()
                    equity = float(acct.get("equity", starting_equity))
                    ledger.update_equity(equity)
                    breaker.update_equity(equity)
                except Exception as exc:
                    ledger.record_error()
                    breaker.record_error()
                    print(f"  [ENGINE] Account query failed: {exc}")

            if fill.status.startswith("error"):
                breaker.record_error()
                ledger.record_error()
            else:
                breaker.clear_errors()
                ledger.clear_errors()

        else:
            # --- Replay mode: simulate fill ---
            slip      = 0.01 if order.side == "buy" else -0.01
            sim_price = mid + slip
            t3        = time.perf_counter_ns()

            # Simple P&L simulation
            if position_side and position_side != order.side:
                # Closing position
                direction = 1.0 if position_side == "buy" else -1.0
                pnl       = (sim_price - entry_price) * direction * order.qty
                # Apply fee: max($1, $0.005/share)
                fee       = max(1.0, order.qty * 0.005)
                net_pnl   = pnl - fee
                ledger.update_equity(ledger._cur_eq + net_pnl)
                breaker.update_equity(ledger._cur_eq)
                position_side = ""
                entry_price   = 0.0
                breaker.set_position_open(False)
                print(f"  [REPLAY] tick={tick_n:>5} | CLOSE "
                      f"{order.qty}x {symbol} @ ${sim_price:.2f} "
                      f"| P&L ${net_pnl:+.2f} "
                      f"| equity ${ledger._cur_eq:.2f}")
            else:
                position_side = order.side
                entry_price   = sim_price
                breaker.set_position_open(True)
                print(f"  [REPLAY] tick={tick_n:>5} | "
                      f"{order.side.upper():4} {order.qty}x "
                      f"{symbol} @ ${sim_price:.2f} "
                      f"| prob={order.signal_prob:.3f} "
                      f"| vol={vol_ann:.1%}")

        latlog.record(t0_ns, t1_ns, t2_ns, t3_ns=t3, had_order=True)

    # ── Session report ─────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Session complete. {tick_n:,} ticks processed.")
    print(f"  Risk:     {ledger.describe()}")
    print(f"  Latency:  {latlog.summary()}")
    print(f"  Router:   {router.describe()}")
    print(f"  Breaker:  {breaker.describe()}")
    if run_mode != "LIVE":
        print(f"\n  [NEXT STEP]")
        print(f"  Set env vars: APCA_API_KEY_ID, APCA_API_SECRET_KEY")
        print(f"  Then run:     $env:RUN_MODE='LIVE'; python -m quant_finance.live_engine")
    print("=" * 62)


def main() -> None:
    asyncio.run(run_engine(
        symbol          = os.environ.get("APCA_SYMBOL", "SPY"),
        starting_equity = 10_000.0,
        run_mode        = os.environ.get("RUN_MODE", "REPLAY"),
    ))


if __name__ == "__main__":
    main()
