"""
backtester.py - Strategy Backtesting Engine
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from data_fetcher import fetch_ohlcv
from indicators import calculate_all_indicators, get_latest_values
from strategies import check_all_strategies, generate_trade_plan
from utils import calculate_max_drawdown

logger = logging.getLogger(__name__)


def run_backtest(
    symbol: str,
    strategy_key: str,
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 100000,
) -> dict:
    """
    Run backtest for a symbol + strategy over a date range.

    Returns:
        dict with metrics + trades list + equity_curve
    """
    # Fetch data
    df = fetch_ohlcv(symbol, period="2y")
    if df is None or len(df) < 60:
        return {"error": f"Insufficient data for {symbol}"}

    # Date filtering
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if len(df) < 60:
        return {"error": "Date range too short for backtesting (min 60 trading days)"}

    # Calculate indicators on full history, then iterate
    df_full = calculate_all_indicators(df.copy())
    if df_full is None or df_full.empty:
        return {"error": "Failed to compute indicators"}

    trades = []
    equity = initial_capital
    equity_curve = [initial_capital]
    in_trade = False
    entry_price = 0
    stop_loss = 0
    target1 = 0
    target2 = 0

    STRATEGY_MAP = {
        "MOMENTUM_BREAKOUT": 0,
        "PULLBACK_UPTREND": 1,
        "VOLATILITY_SQUEEZE": 2,
        "REVERSAL_SWING": 3,
    }

    min_lookback = 50

    for i in range(min_lookback, len(df_full) - 1):
        df_slice = df_full.iloc[:i + 1].copy()
        ind = get_latest_values(df_slice)
        current_price = ind.get("Close", 0)
        current_date = str(df_full.index[i].date())

        if in_trade:
            # Check exit conditions
            next_open = df_full["Open"].iloc[i + 1]
            next_high = df_full["High"].iloc[i + 1]
            next_low = df_full["Low"].iloc[i + 1]
            exit_price = None
            exit_reason = None

            # Stop loss hit
            if next_low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "Stop Loss"
            # Target 1 hit
            elif next_high >= target1:
                exit_price = target1
                exit_reason = "Target 1"
            # Check for strategy exit signals (bearish reversal)
            elif ind.get("RSI", 50) > 75:
                exit_price = current_price
                exit_reason = "RSI Overbought"
            elif ind.get("MACD", 0) < ind.get("MACD_Signal", 0) and ind.get("Close", 0) < ind.get("EMA20", 0):
                exit_price = current_price
                exit_reason = "Trend Reversal"

            if exit_price:
                position_size = equity * 0.1 / entry_price
                pnl = (exit_price - entry_price) * position_size
                ret_pct = ((exit_price - entry_price) / entry_price) * 100
                equity += pnl
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "return_pct": round(ret_pct, 2),
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "result": "WIN" if pnl > 0 else "LOSS",
                })
                in_trade = False
                equity_curve.append(round(equity, 2))

        else:
            # Check entry signals
            triggered = check_all_strategies(df_slice, ind)
            strat_idx = STRATEGY_MAP.get(strategy_key, -1)

            entry_signal = False
            if strategy_key == "ALL":
                entry_signal = len(triggered) > 0
            elif strat_idx >= 0:
                entry_signal = any(s["key"] == strategy_key for s in triggered)

            if entry_signal:
                plan = generate_trade_plan(df_slice, ind, strategy_key)
                entry_price = plan.get("entry", current_price)
                stop_loss = plan.get("stop_loss", current_price * 0.97)
                target1 = plan.get("target1", current_price * 1.05)
                target2 = plan.get("target2", current_price * 1.08)
                entry_date = current_date
                in_trade = True

    # Close any open trade at end
    if in_trade and len(df_full) > 0:
        final_price = df_full["Close"].iloc[-1]
        position_size = equity * 0.1 / entry_price
        pnl = (final_price - entry_price) * position_size
        ret_pct = ((final_price - entry_price) / entry_price) * 100
        equity += pnl
        trades.append({
            "entry_date": entry_date,
            "exit_date": str(df_full.index[-1].date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(final_price, 2),
            "return_pct": round(ret_pct, 2),
            "pnl": round(pnl, 2),
            "exit_reason": "End of Period",
            "result": "WIN" if pnl > 0 else "LOSS",
        })
        equity_curve.append(round(equity, 2))

    # Compute metrics
    metrics = _compute_metrics(trades, equity_curve, initial_capital)
    metrics["symbol"] = symbol
    metrics["strategy"] = strategy_key
    metrics["start_date"] = start_date or str(df.index[0].date())
    metrics["end_date"] = end_date or str(df.index[-1].date())
    metrics["equity_curve"] = equity_curve

    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _compute_metrics(trades: list, equity_curve: list, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_return": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "total_return_pct": 0,
            "final_equity": initial_capital,
        }

    total = len(trades)
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    win_rate = len(wins) / total * 100

    returns = [t["return_pct"] for t in trades]
    avg_return = np.mean(returns)

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / max(gross_loss, 0.01)

    eq_series = pd.Series(equity_curve)
    max_dd = calculate_max_drawdown(eq_series) * 100

    final_equity = equity_curve[-1] if equity_curve else initial_capital
    total_return = ((final_equity - initial_capital) / initial_capital) * 100

    avg_win = np.mean([t["return_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["return_pct"] for t in losses]) if losses else 0

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "avg_return": round(avg_return, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "total_return_pct": round(total_return, 2),
        "final_equity": round(final_equity, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }
