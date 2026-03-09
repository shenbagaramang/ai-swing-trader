"""
institutional_tracker.py - FII/DII Institutional Flow Tracker
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from data_fetcher import fetch_fii_dii_data

logger = logging.getLogger(__name__)

_fii_dii_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600  # 1 hour


def get_fii_dii_data(force_refresh: bool = False) -> dict:
    """Get FII/DII data with caching."""
    import time
    now = time.time()
    if not force_refresh and _fii_dii_cache["data"] and (now - _fii_dii_cache["ts"] < CACHE_TTL):
        return _fii_dii_cache["data"]
    data = fetch_fii_dii_data()
    _fii_dii_cache["data"] = data
    _fii_dii_cache["ts"] = now
    return data


def get_institutional_sentiment(data: dict = None) -> str:
    """
    Determine overall institutional sentiment.
    Returns 'bullish', 'bearish', or 'neutral'.
    """
    if data is None:
        data = get_fii_dii_data()

    fii = data.get("fii", [])
    if not fii:
        return "neutral"

    # Last 5 days FII net
    recent = fii[:5]
    net_sum = sum(d.get("net", 0) for d in recent)

    if net_sum > 2000:  # Net buyers
        return "bullish"
    elif net_sum < -2000:  # Net sellers
        return "bearish"
    else:
        return "neutral"


def get_fii_trend(data: dict = None) -> dict:
    """Analyze FII trend over last 10/20 days."""
    if data is None:
        data = get_fii_dii_data()

    fii = data.get("fii", [])
    dii = data.get("dii", [])

    def analyze_series(series, name):
        if not series:
            return {}
        nets = [d.get("net", 0) for d in series]
        return {
            f"{name}_last5_net": sum(nets[:5]),
            f"{name}_last10_net": sum(nets[:10]),
            f"{name}_last20_net": sum(nets[:20]),
            f"{name}_positive_days": sum(1 for n in nets[:10] if n > 0),
            f"{name}_trend": "buying" if sum(nets[:5]) > 0 else "selling",
        }

    result = {}
    result.update(analyze_series(fii, "fii"))
    result.update(analyze_series(dii, "dii"))
    return result


def to_dataframe(data: dict) -> tuple:
    """Convert FII/DII data to DataFrames."""
    fii = data.get("fii", [])
    dii = data.get("dii", [])

    fii_df = pd.DataFrame(fii) if fii else pd.DataFrame(columns=["date", "net", "buy", "sell"])
    dii_df = pd.DataFrame(dii) if dii else pd.DataFrame(columns=["date", "net", "buy", "sell"])

    return fii_df, dii_df


def get_combined_flow_df(data: dict = None) -> pd.DataFrame:
    """Get combined FII+DII flow DataFrame."""
    if data is None:
        data = get_fii_dii_data()

    fii_df, dii_df = to_dataframe(data)
    if fii_df.empty:
        return pd.DataFrame()

    combined = fii_df[["date", "net"]].copy().rename(columns={"net": "FII_Net"})
    if not dii_df.empty:
        combined = combined.merge(
            dii_df[["date", "net"]].rename(columns={"net": "DII_Net"}),
            on="date", how="outer"
        )
    combined["Total_Net"] = combined.get("FII_Net", 0) + combined.get("DII_Net", 0)
    return combined


def flag_institutional_stocks(scan_results: list, fii_data: dict = None) -> list:
    """
    Flag scan results with institutional participation indicators.
    Stocks with breakout + FII buying get an institutional flag.
    """
    if fii_data is None:
        fii_data = get_fii_dii_data()

    sentiment = get_institutional_sentiment(fii_data)
    flagged = []
    for stock in scan_results:
        stock = stock.copy()
        inst_flag = False
        inst_note = ""

        # If FII is bullish and stock is breaking out
        if sentiment == "bullish" and stock.get("breakout_20d", 0) == 1:
            inst_flag = True
            inst_note = "FII buying + breakout"
        elif sentiment == "bullish" and stock.get("ai_score", 0) >= 70:
            inst_flag = True
            inst_note = "FII buying support"
        elif sentiment == "bearish":
            inst_note = "FII selling pressure"

        stock["inst_flag"] = inst_flag
        stock["inst_note"] = inst_note
        flagged.append(stock)
    return flagged
