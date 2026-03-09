"""
options_analyzer.py - Options Chain Analysis for NSE stocks
"""

import pandas as pd
import numpy as np
import logging
from data_fetcher import fetch_option_chain

logger = logging.getLogger(__name__)

_oc_cache = {}


def get_option_analysis(symbol: str, force_refresh: bool = False) -> dict:
    """Get full option chain analysis for a symbol."""
    import time
    now = time.time()
    if not force_refresh and symbol in _oc_cache:
        ts, data = _oc_cache[symbol]
        if now - ts < 900:  # 15 min cache
            return data

    raw = fetch_option_chain(symbol)
    analysis = analyze_option_chain(raw)
    _oc_cache[symbol] = (now, analysis)
    return analysis


def analyze_option_chain(raw: dict) -> dict:
    """
    Full options analysis:
    - PCR, Max Pain, OI buildup, sentiment
    - Short covering / long buildup detection
    """
    result = raw.copy()
    strikes = raw.get("strikes", [])
    current_price = raw.get("current_price", 0)

    if not strikes or not current_price:
        return result

    # ATM analysis (nearest 5 strikes to current price)
    sorted_strikes = sorted(strikes, key=lambda x: abs(x["strike"] - current_price))
    atm_strikes = sorted_strikes[:5]

    result["atm_call_oi"] = sum(s["call_oi"] for s in atm_strikes)
    result["atm_put_oi"] = sum(s["put_oi"] for s in atm_strikes)
    result["atm_pcr"] = round(result["atm_put_oi"] / max(result["atm_call_oi"], 1), 2)

    # OI change analysis
    total_call_chg = sum(s.get("call_chg_oi", 0) for s in strikes)
    total_put_chg = sum(s.get("put_chg_oi", 0) for s in strikes)
    result["call_oi_change"] = total_call_chg
    result["put_oi_change"] = total_put_chg

    # Detect market conditions
    price_up = True  # Assume — will be set by price data
    result["market_condition"] = _detect_market_condition(
        total_call_chg, total_put_chg, current_price, raw.get("pcr", 1)
    )

    # Highlight strong signals
    result["bullish_signal"] = _is_bullish_signal(result)
    result["bearish_signal"] = _is_bearish_signal(result)

    # OI concentration
    result["resistance_levels"] = _find_resistance_from_oi(strikes, current_price)
    result["support_levels"] = _find_support_from_oi(strikes, current_price)

    return result


def _detect_market_condition(call_chg: int, put_chg: int, price: float, pcr: float) -> str:
    """Classify market condition based on OI changes."""
    if call_chg < 0 and put_chg > 0:
        return "short_covering"  # Calls unwinding, puts building
    elif call_chg > 0 and put_chg > 0:
        if pcr > 1.1:
            return "long_buildup_bullish"
        else:
            return "long_buildup_neutral"
    elif call_chg > 0 and put_chg < 0:
        return "short_buildup_bearish"  # Calls building, puts unwinding
    elif call_chg < 0 and put_chg < 0:
        return "long_unwinding"
    return "neutral"


def _is_bullish_signal(analysis: dict) -> bool:
    """Check for strong bullish options signal."""
    pcr = analysis.get("pcr", 1)
    condition = analysis.get("market_condition", "")
    oi_buildup = analysis.get("oi_buildup", "")

    return (
        (pcr > 1.2) or
        (condition in ["short_covering", "long_buildup_bullish"]) or
        (oi_buildup == "put_buildup_bullish")
    )


def _is_bearish_signal(analysis: dict) -> bool:
    """Check for strong bearish options signal."""
    pcr = analysis.get("pcr", 1)
    condition = analysis.get("market_condition", "")

    return pcr < 0.7 or condition in ["short_buildup_bearish", "long_unwinding"]


def _find_resistance_from_oi(strikes: list, current_price: float) -> list:
    """Find resistance levels from high call OI concentrations."""
    above = [s for s in strikes if s["strike"] > current_price]
    if not above:
        return []
    sorted_above = sorted(above, key=lambda x: x["call_oi"], reverse=True)
    return [s["strike"] for s in sorted_above[:3]]


def _find_support_from_oi(strikes: list, current_price: float) -> list:
    """Find support levels from high put OI concentrations."""
    below = [s for s in strikes if s["strike"] <= current_price]
    if not below:
        return []
    sorted_below = sorted(below, key=lambda x: x["put_oi"], reverse=True)
    return [s["strike"] for s in sorted_below[:3]]


def format_oi_table(analysis: dict) -> pd.DataFrame:
    """Format option chain as DataFrame for display."""
    strikes = analysis.get("strikes", [])
    if not strikes:
        return pd.DataFrame()

    current = analysis.get("current_price", 0)
    rows = []
    for s in strikes:
        rows.append({
            "Strike": s["strike"],
            "Call OI": s["call_oi"],
            "Call Chg OI": s.get("call_chg_oi", 0),
            "Call IV": f"{s.get('call_iv', 0):.1f}%",
            "ATM": "⭐" if abs(s["strike"] - current) < current * 0.01 else "",
            "Put OI": s["put_oi"],
            "Put Chg OI": s.get("put_chg_oi", 0),
            "Put IV": f"{s.get('put_iv', 0):.1f}%",
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Strike", ascending=False).reset_index(drop=True)
