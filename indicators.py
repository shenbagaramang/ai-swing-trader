"""
indicators.py - Technical Indicators Calculator for AI Swing Trader
Uses pandas-ta for indicator computation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators on the given OHLCV DataFrame.
    Returns the DataFrame with all indicator columns added.
    """
    if df is None or len(df) < 30:
        return df

    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]

    try:
        # ── Trend Indicators ──────────────────────────────────────────────────
        df["EMA20"] = _ema(df["Close"], 20)
        df["EMA50"] = _ema(df["Close"], 50)
        df["EMA200"] = _ema(df["Close"], 200)

        # Supertrend
        df = _supertrend(df, period=10, multiplier=3.0)

        # ADX
        df = _adx(df, period=14)

        # ── Momentum Indicators ────────────────────────────────────────────────
        df["RSI"] = _rsi(df["Close"], period=14)
        df = _macd(df)
        df = _stoch_rsi(df)
        df["MOM"] = df["Close"].diff(10)  # 10-period momentum

        # ── Volatility Indicators ──────────────────────────────────────────────
        df = _atr(df, period=14)
        df = _bollinger_bands(df, period=20, std=2.0)
        df = _keltner_channel(df, period=20, multiplier=1.5)

        # ── Volume Indicators ──────────────────────────────────────────────────
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["Vol_Spike"] = df["Volume"] / df["Vol_SMA20"].replace(0, np.nan)
        df = _vwap(df)

        # ── Pattern Detection ──────────────────────────────────────────────────
        df = _detect_candle_patterns(df)

        # ── Derived Signals ────────────────────────────────────────────────────
        df["Above_EMA20"] = (df["Close"] > df["EMA20"]).astype(int)
        df["Above_EMA50"] = (df["Close"] > df["EMA50"]).astype(int)
        df["Above_EMA200"] = (df["Close"] > df["EMA200"]).astype(int)
        df["EMA_Aligned"] = ((df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])).astype(int)

        df["High20"] = df["High"].rolling(20).max()
        df["Low20"] = df["Low"].rolling(20).min()
        df["Breakout_20d"] = (df["Close"] >= df["High20"].shift(1)).astype(int)

        # BB squeeze
        if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
            df["BB_Squeeze"] = (df["BB_Width"] < df["BB_Width"].rolling(20).mean() * 0.8).astype(int)

        df.dropna(subset=["EMA20", "RSI"], inplace=True)

    except Exception as e:
        logger.error(f"Indicator calculation error: {e}")

    return df


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(df: pd.DataFrame) -> pd.DataFrame:
    fast = _ema(df["Close"], 12)
    slow = _ema(df["Close"], 26)
    df["MACD"] = fast - slow
    df["MACD_Signal"] = _ema(df["MACD"], 9)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Cross_Up"] = (
        (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    ).astype(int)
    return df


def _stoch_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    rsi = _rsi(df["Close"], period)
    rsi_min = rsi.rolling(period).min()
    rsi_max = rsi.rolling(period).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    df["Stoch_RSI_K"] = stoch.rolling(3).mean() * 100
    df["Stoch_RSI_D"] = df["Stoch_RSI_K"].rolling(3).mean()
    return df


def _atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100
    return df


def _bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(period).mean()
    std_dev = df["Close"].rolling(period).std()
    df["BB_Upper"] = sma + std * std_dev
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - std * std_dev
    df["BB_Percent"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)
    return df


def _keltner_channel(df: pd.DataFrame, period: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
    ema = _ema(df["Close"], period)
    if "ATR" not in df.columns:
        df = _atr(df)
    df["KC_Upper"] = ema + multiplier * df["ATR"]
    df["KC_Middle"] = ema
    df["KC_Lower"] = ema - multiplier * df["ATR"]
    return df


def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    if "ATR" not in df.columns:
        df = _atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * df["ATR"]
    lower_band = hl2 - multiplier * df["ATR"]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Finalize bands
        if df["Close"].iloc[i - 1] <= upper_band.iloc[i - 1]:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])
        if df["Close"].iloc[i - 1] >= lower_band.iloc[i - 1]:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])

        if i == 1:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        elif supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            if df["Close"].iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        else:
            if df["Close"].iloc[i] >= lower_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

    df["Supertrend"] = supertrend
    df["Supertrend_Dir"] = direction
    return df


def _adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["ADX"] = dx.ewm(span=period, adjust=False).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    return df


def _vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    return df


def _detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    body = (c - o).abs()
    total = h - l + 1e-10

    # Hammer: small body at top, long lower wick
    lower_wick = pd.Series(np.where(c > o, o - l, c - l), index=df.index)
    upper_wick = pd.Series(np.where(c > o, h - c, h - o), index=df.index)
    df["Hammer"] = ((body / total < 0.3) & (lower_wick > body * 2) & (upper_wick < body * 0.5)).astype(int)

    # Bullish Engulfing
    df["Bull_Engulf"] = (
        (c.shift(1) < o.shift(1)) &  # prev candle bearish
        (c > o) &  # current candle bullish
        (o <= c.shift(1)) &  # current open <= prev close
        (c >= o.shift(1))  # current close >= prev open
    ).astype(int)

    # Bearish Engulfing
    df["Bear_Engulf"] = (
        (c.shift(1) > o.shift(1)) &
        (c < o) &
        (o >= c.shift(1)) &
        (c <= o.shift(1))
    ).astype(int)

    # Doji
    df["Doji"] = (body / total < 0.1).astype(int)

    # Shooting Star
    df["Shooting_Star"] = ((body / total < 0.3) & (upper_wick > body * 2) & (lower_wick < body * 0.5)).astype(int)

    return df


def get_latest_values(df: pd.DataFrame) -> dict:
    """Extract latest indicator values as a flat dictionary."""
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    vals = {}
    for col in df.columns:
        try:
            v = row[col]
            vals[col] = float(v) if not pd.isna(v) else 0.0
        except Exception:
            vals[col] = 0.0
    return vals
