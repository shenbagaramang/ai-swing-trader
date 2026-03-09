"""
data_fetcher.py - Market data fetching for AI Swing Trader
Fetches OHLCV, NSE FII/DII flows, option chain data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# NSE API headers (mimicking browser)
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}

_nse_session = None


def get_nse_session() -> requests.Session:
    global _nse_session
    if _nse_session is None:
        _nse_session = requests.Session()
        _nse_session.headers.update(NSE_HEADERS)
        try:
            _nse_session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)
        except Exception:
            pass
    return _nse_session


# ─── OHLCV Data ───────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol using yfinance.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception as e:
        logger.debug(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_batch_ohlcv(symbols: list, period: str = "1y") -> dict:
    """Fetch OHLCV for multiple symbols using yfinance batch download."""
    result = {}
    try:
        # yfinance batch download
        tickers_str = " ".join(symbols)
        data = yf.download(
            tickers_str,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if len(symbols) == 1:
            sym = symbols[0]
            if not data.empty:
                df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.dropna(subset=["Close"], inplace=True)
                result[sym] = df
        else:
            for sym in symbols:
                try:
                    if sym in data.columns.get_level_values(0):
                        df = data[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
                    elif hasattr(data, 'columns') and isinstance(data.columns, pd.MultiIndex):
                        df = data.xs(sym, axis=1, level=0)[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        continue
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df.dropna(subset=["Close"], inplace=True)
                    if not df.empty:
                        result[sym] = df
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Batch download error: {e}")
        # Fallback: individual fetch
        for sym in symbols:
            df = fetch_ohlcv(sym, period=period)
            if not df.empty:
                result[sym] = df
    return result


def fetch_stock_info(symbol: str) -> dict:
    """Fetch stock info (name, sector, market cap etc)."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "eps": info.get("trailingEps", 0),
        }
    except Exception:
        return {"symbol": symbol, "name": symbol, "sector": "Unknown"}


# ─── NSE FII / DII Data ──────────────────────────────────────────────────────

def fetch_fii_dii_data() -> dict:
    """
    Fetch FII/DII institutional flow data from NSE.
    Falls back to generated mock data if NSE blocks request.
    """
    try:
        session = get_nse_session()
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return _parse_fii_dii(data)
    except Exception as e:
        logger.warning(f"NSE FII/DII fetch failed: {e}")

    # Fallback: simulate realistic data
    return _generate_mock_fii_dii()


def _parse_fii_dii(raw: list) -> dict:
    """Parse NSE FII/DII API response."""
    result = {"fii": [], "dii": [], "date_range": []}
    try:
        for item in raw[:20]:  # Last 20 trading days
            date = item.get("date", "")
            fii_net = float(item.get("fiiNet", 0))
            dii_net = float(item.get("diiNet", 0))
            result["fii"].append({"date": date, "net": fii_net,
                                   "buy": float(item.get("fiiBuy", 0)),
                                   "sell": float(item.get("fiiSell", 0))})
            result["dii"].append({"date": date, "net": dii_net,
                                   "buy": float(item.get("diiBuy", 0)),
                                   "sell": float(item.get("diiSell", 0))})
            result["date_range"].append(date)
    except Exception as e:
        logger.debug(f"FII/DII parse error: {e}")
    return result


def _generate_mock_fii_dii() -> dict:
    """Generate realistic mock FII/DII data."""
    import random
    dates = [(datetime.now() - timedelta(days=i)).strftime("%d-%b-%Y") for i in range(20)]
    fii = []
    dii = []
    for d in dates:
        fii_net = random.gauss(500, 1500)
        dii_net = random.gauss(300, 1000)
        fii.append({
            "date": d,
            "net": round(fii_net, 2),
            "buy": round(abs(fii_net) + random.uniform(3000, 8000), 2),
            "sell": round(abs(fii_net) + random.uniform(2000, 7000), 2),
        })
        dii.append({
            "date": d,
            "net": round(dii_net, 2),
            "buy": round(abs(dii_net) + random.uniform(2000, 5000), 2),
            "sell": round(abs(dii_net) + random.uniform(1500, 4500), 2),
        })
    return {"fii": fii, "dii": dii, "date_range": dates}


# ─── Option Chain ─────────────────────────────────────────────────────────────

def fetch_option_chain(symbol: str) -> dict:
    """
    Fetch option chain data from NSE for a given symbol.
    Returns parsed option chain with calls/puts, OI, PCR, max pain.
    """
    # Remove .NS suffix for NSE API
    nse_symbol = symbol.replace(".NS", "")

    try:
        session = get_nse_session()
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={nse_symbol}"
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return _parse_option_chain(data, nse_symbol)
    except Exception as e:
        logger.warning(f"Option chain fetch failed for {nse_symbol}: {e}")

    return _generate_mock_option_chain(nse_symbol)


def _parse_option_chain(raw: dict, symbol: str) -> dict:
    """Parse NSE option chain response."""
    result = {
        "symbol": symbol,
        "expiry_dates": [],
        "current_price": 0,
        "pcr": 0,
        "max_pain": 0,
        "call_oi_total": 0,
        "put_oi_total": 0,
        "strikes": [],
        "oi_buildup": "neutral",
        "sentiment": "neutral",
    }
    try:
        records = raw.get("records", {})
        data = records.get("data", [])
        result["expiry_dates"] = records.get("expiryDates", [])
        result["current_price"] = records.get("underlyingValue", 0)

        strikes_data = {}
        for item in data:
            strike = item.get("strikePrice", 0)
            if strike not in strikes_data:
                strikes_data[strike] = {"strike": strike, "call_oi": 0, "put_oi": 0,
                                         "call_chg_oi": 0, "put_chg_oi": 0,
                                         "call_iv": 0, "put_iv": 0}
            if "CE" in item:
                ce = item["CE"]
                strikes_data[strike]["call_oi"] = ce.get("openInterest", 0)
                strikes_data[strike]["call_chg_oi"] = ce.get("changeinOpenInterest", 0)
                strikes_data[strike]["call_iv"] = ce.get("impliedVolatility", 0)
            if "PE" in item:
                pe = item["PE"]
                strikes_data[strike]["put_oi"] = pe.get("openInterest", 0)
                strikes_data[strike]["put_chg_oi"] = pe.get("changeinOpenInterest", 0)
                strikes_data[strike]["put_iv"] = pe.get("impliedVolatility", 0)

        result["strikes"] = sorted(strikes_data.values(), key=lambda x: x["strike"])
        result["call_oi_total"] = sum(s["call_oi"] for s in result["strikes"])
        result["put_oi_total"] = sum(s["put_oi"] for s in result["strikes"])

        if result["call_oi_total"] > 0:
            result["pcr"] = round(result["put_oi_total"] / result["call_oi_total"], 2)

        # Max pain calculation
        result["max_pain"] = _calculate_max_pain(result["strikes"])

        # Sentiment
        if result["pcr"] > 1.2:
            result["sentiment"] = "bullish"
        elif result["pcr"] < 0.7:
            result["sentiment"] = "bearish"
        else:
            result["sentiment"] = "neutral"

        # OI buildup detection
        call_chg = sum(s["call_chg_oi"] for s in result["strikes"])
        put_chg = sum(s["put_chg_oi"] for s in result["strikes"])
        if put_chg > call_chg * 1.5:
            result["oi_buildup"] = "put_buildup_bullish"
        elif call_chg > put_chg * 1.5:
            result["oi_buildup"] = "call_buildup_bearish"
        else:
            result["oi_buildup"] = "neutral"

    except Exception as e:
        logger.debug(f"Option chain parse error: {e}")

    return result


def _calculate_max_pain(strikes: list) -> float:
    """Calculate max pain strike price."""
    if not strikes:
        return 0
    max_pain_strike = 0
    min_loss = float("inf")
    for target in strikes:
        t = target["strike"]
        loss = sum(
            max(0, t - s["strike"]) * s["call_oi"] +
            max(0, s["strike"] - t) * s["put_oi"]
            for s in strikes
        )
        if loss < min_loss:
            min_loss = loss
            max_pain_strike = t
    return max_pain_strike


def _generate_mock_option_chain(symbol: str) -> dict:
    """Generate mock option chain for demonstration."""
    import random
    current_price = random.uniform(500, 3000)
    atm = round(current_price / 50) * 50
    strikes = []
    for i in range(-10, 11):
        strike = atm + i * 50
        call_oi = max(0, int(random.gauss(5000 - abs(i) * 400, 500)))
        put_oi = max(0, int(random.gauss(4500 - abs(i) * 400, 500)))
        strikes.append({
            "strike": strike,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_chg_oi": random.randint(-500, 1000),
            "put_chg_oi": random.randint(-500, 1000),
            "call_iv": random.uniform(15, 35),
            "put_iv": random.uniform(15, 35),
        })

    call_oi_total = sum(s["call_oi"] for s in strikes)
    put_oi_total = sum(s["put_oi"] for s in strikes)
    pcr = round(put_oi_total / max(call_oi_total, 1), 2)

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "pcr": pcr,
        "max_pain": _calculate_max_pain(strikes),
        "call_oi_total": call_oi_total,
        "put_oi_total": put_oi_total,
        "strikes": strikes,
        "expiry_dates": ["current", "next_month"],
        "oi_buildup": "neutral",
        "sentiment": "bullish" if pcr > 1.2 else ("bearish" if pcr < 0.7 else "neutral"),
    }


# ─── Sector Indices ───────────────────────────────────────────────────────────

def fetch_sector_indices() -> dict:
    """Fetch sector index performance from yfinance."""
    sector_symbols = {
        "NIFTY50": "^NSEI",
        "NIFTY Bank": "^NSEBANK",
        "NIFTY IT": "^CNXIT",
        "NIFTY Pharma": "^CNXPHARMA",
        "NIFTY Auto": "^CNXAUTO",
        "NIFTY FMCG": "^CNXFMCG",
        "NIFTY Metal": "^CNXMETAL",
        "NIFTY Energy": "^CNXENERGY",
        "NIFTY Realty": "^CNXREALTY",
        "NIFTY Infra": "^CNXINFRA",
    }
    results = {}
    for name, sym in sector_symbols.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty and len(hist) >= 2:
                latest = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                chg_pct = ((latest - prev) / prev) * 100
                week_chg = ((latest - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
                results[name] = {
                    "price": round(latest, 2),
                    "change_pct": round(chg_pct, 2),
                    "week_change_pct": round(week_chg, 2),
                }
        except Exception:
            pass
    return results
