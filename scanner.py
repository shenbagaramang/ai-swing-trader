"""
scanner.py - High-Performance Parallel Stock Scanner
Analyzes 1500+ NSE stocks using multiprocessing + caching
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from data_fetcher import fetch_ohlcv, fetch_batch_ohlcv
from indicators import calculate_all_indicators, get_latest_values
from strategies import check_all_strategies, generate_trade_plan
from ai_model import predict_move_probability
from ranking_engine import calculate_ai_score
from utils import chunk_list, get_sector, safe_float

logger = logging.getLogger(__name__)

# Global cache: symbol -> (timestamp, df_with_indicators)
_price_cache = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Get data from cache or fetch fresh."""
    now = time.time()
    if symbol in _price_cache:
        ts, df = _price_cache[symbol]
        if now - ts < CACHE_TTL_SECONDS and df is not None and not df.empty:
            return df

    df = fetch_ohlcv(symbol, period=period)
    if df is not None and not df.empty:
        df_ind = calculate_all_indicators(df)
        _price_cache[symbol] = (now, df_ind)
        return df_ind
    return pd.DataFrame()


def analyze_single_stock(symbol: str, fii_sentiment: str = "neutral") -> dict:
    """
    Full analysis pipeline for a single stock.
    Returns structured result dict or None.
    """
    try:
        df = _get_cached_data(symbol)
        if df is None or len(df) < 50:
            return None

        ind = get_latest_values(df)
        if not ind or ind.get("Close", 0) <= 0:
            return None

        price = ind["Close"]
        rsi = safe_float(ind.get("RSI", 0))
        vol_spike = safe_float(ind.get("Vol_Spike", 1.0))

        # Run strategies
        triggered_strategies = check_all_strategies(df, ind)

        # ML prediction
        ml_probs = predict_move_probability(df)

        # AI score
        score_dict = calculate_ai_score(
            indicators=ind,
            strategies_triggered=triggered_strategies,
            ml_probs=ml_probs,
            fii_sentiment=fii_sentiment,
        )

        # Trade plan (use first triggered strategy or default)
        strategy_name = triggered_strategies[0]["key"] if triggered_strategies else "DEFAULT"
        trade_plan = generate_trade_plan(df, ind, strategy_name)

        # Strategy names for display
        strategy_labels = [s["name"] for s in triggered_strategies]

        result = {
            "symbol": symbol,
            "display_symbol": symbol.replace(".NS", ""),
            "price": round(price, 2),
            "sector": get_sector(symbol),
            "rsi": round(rsi, 1),
            "volume_spike": round(vol_spike, 2),
            "strategy": ", ".join(strategy_labels) if strategy_labels else "—",
            "strategies_count": len(triggered_strategies),
            "ai_score": score_dict["total_score"],
            "score_breakdown": score_dict,
            "prob_5pct": ml_probs.get("prob_5pct", 0),
            "prob_8pct": ml_probs.get("prob_8pct", 0),
            "prob_10pct": ml_probs.get("prob_10pct", 0),
            "model_used": ml_probs.get("model_used", "fallback"),
            "entry": trade_plan.get("entry", price),
            "stop_loss": trade_plan.get("stop_loss", 0),
            "target1": trade_plan.get("target1", 0),
            "target2": trade_plan.get("target2", 0),
            "risk_reward": trade_plan.get("risk_reward", 0),
            "atr": trade_plan.get("atr", 0),
            "ema20": round(safe_float(ind.get("EMA20", 0)), 2),
            "ema50": round(safe_float(ind.get("EMA50", 0)), 2),
            "ema200": round(safe_float(ind.get("EMA200", 0)), 2),
            "adx": round(safe_float(ind.get("ADX", 0)), 1),
            "macd": round(safe_float(ind.get("MACD", 0)), 4),
            "macd_signal": round(safe_float(ind.get("MACD_Signal", 0)), 4),
            "supertrend_dir": int(ind.get("Supertrend_Dir", 0)),
            "bb_pct": round(safe_float(ind.get("BB_Percent", 0.5)), 3),
            "ema_aligned": int(ind.get("EMA_Aligned", 0)),
            "breakout_20d": int(ind.get("Breakout_20d", 0)),
            "scan_time": datetime.now().isoformat(),
        }
        return result

    except Exception as e:
        logger.debug(f"Error analyzing {symbol}: {e}")
        return None


def _worker_analyze(args):
    """Worker function for multiprocessing."""
    symbol, fii_sentiment = args
    try:
        return analyze_single_stock(symbol, fii_sentiment)
    except Exception:
        return None


def run_scanner(
    symbols: list,
    fii_sentiment: str = "neutral",
    min_score: float = 0,
    strategies_filter: list = None,
    progress_callback=None,
) -> list:
    """
    Run the scanner on a list of symbols.
    Uses batch fetching + multiprocessing for speed.

    Returns list of result dicts, sorted by AI score.
    """
    logger.info(f"Starting scan of {len(symbols)} stocks...")
    start_time = time.time()
    results = []

    # Pre-fetch in batches using yfinance batch download
    chunk_size = 50
    chunks = chunk_list(symbols, chunk_size)

    fetched_data = {}
    for i, chunk in enumerate(chunks):
        try:
            batch_data = fetch_batch_ohlcv(chunk, period="1y")
            fetched_data.update(batch_data)
            if progress_callback:
                progress_callback(int((i + 1) / len(chunks) * 40))  # 0-40%
        except Exception as e:
            logger.debug(f"Batch fetch error: {e}")

    # Pre-populate cache
    now = time.time()
    for sym, df in fetched_data.items():
        if df is not None and not df.empty and len(df) >= 50:
            try:
                df_ind = calculate_all_indicators(df)
                _price_cache[sym] = (now, df_ind)
            except Exception:
                pass

    # Analyze all symbols (multiprocessing for large lists)
    args_list = [(sym, fii_sentiment) for sym in symbols]

    if len(symbols) > 50:
        workers = min(cpu_count(), 4)
        try:
            with Pool(workers) as pool:
                batch_results = []
                for i, res in enumerate(pool.imap_unordered(_worker_analyze, args_list, chunksize=10)):
                    if res:
                        batch_results.append(res)
                    if progress_callback and i % 20 == 0:
                        pct = 40 + int(i / len(symbols) * 55)
                        progress_callback(min(95, pct))
            results = batch_results
        except Exception as e:
            logger.warning(f"Multiprocessing failed, falling back to sequential: {e}")
            for i, (sym, sent) in enumerate(args_list):
                res = analyze_single_stock(sym, sent)
                if res:
                    results.append(res)
                if progress_callback and i % 10 == 0:
                    progress_callback(40 + int(i / len(symbols) * 55))
    else:
        for i, (sym, sent) in enumerate(args_list):
            res = analyze_single_stock(sym, sent)
            if res:
                results.append(res)
            if progress_callback:
                progress_callback(40 + int((i + 1) / len(symbols) * 55))

    # Apply filters
    if min_score > 0:
        results = [r for r in results if r.get("ai_score", 0) >= min_score]

    if strategies_filter:
        results = [r for r in results
                   if any(sf.lower() in r.get("strategy", "").lower() for sf in strategies_filter)]

    # Sort by AI score
    results.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

    elapsed = time.time() - start_time
    logger.info(f"Scan complete: {len(results)} results from {len(symbols)} stocks in {elapsed:.1f}s")

    if progress_callback:
        progress_callback(100)

    return results


def run_watchlist_scan(watchlist: list, fii_sentiment: str = "neutral") -> list:
    """Scan only watchlist stocks."""
    symbols = [item["symbol"] if isinstance(item, dict) else item for item in watchlist]
    symbols = [s if ".NS" in s else f"{s}.NS" for s in symbols]
    return run_scanner(symbols, fii_sentiment=fii_sentiment)


def get_scan_summary(results: list) -> dict:
    """Compute summary statistics from scan results."""
    if not results:
        return {}
    scores = [r.get("ai_score", 0) for r in results]
    strong_buys = sum(1 for s in scores if s >= 80)
    buys = sum(1 for s in scores if 65 <= s < 80)
    watches = sum(1 for s in scores if 50 <= s < 65)

    return {
        "total_scanned": len(results),
        "strong_buys": strong_buys,
        "buys": buys,
        "watches": watches,
        "avg_score": round(np.mean(scores), 1) if scores else 0,
        "max_score": round(max(scores), 1) if scores else 0,
        "strategies_dist": _count_strategies(results),
    }


def _count_strategies(results: list) -> dict:
    counts = {}
    for r in results:
        strats = r.get("strategy", "").split(", ")
        for s in strats:
            if s and s != "—":
                counts[s] = counts.get(s, 0) + 1
    return counts
