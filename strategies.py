"""
strategies.py - Swing Trading Strategy Engine
Implements 4 core strategies with signal detection
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

STRATEGIES = {
    "MOMENTUM_BREAKOUT": "Momentum Breakout",
    "PULLBACK_UPTREND": "Pullback in Uptrend",
    "VOLATILITY_SQUEEZE": "Volatility Squeeze Breakout",
    "REVERSAL_SWING": "Reversal Swing",
}


def check_all_strategies(df: pd.DataFrame, indicators: dict) -> list:
    """
    Run all strategies on the given DataFrame + indicators dict.
    Returns list of triggered strategy names.
    """
    triggered = []
    strategies = [
        ("MOMENTUM_BREAKOUT", check_momentum_breakout),
        ("PULLBACK_UPTREND", check_pullback_uptrend),
        ("VOLATILITY_SQUEEZE", check_volatility_squeeze),
        ("REVERSAL_SWING", check_reversal_swing),
    ]
    for key, func in strategies:
        try:
            signal = func(df, indicators)
            if signal["triggered"]:
                triggered.append({
                    "key": key,
                    "name": STRATEGIES[key],
                    "strength": signal.get("strength", 50),
                    "details": signal.get("details", {}),
                    "description": signal.get("description", ""),
                })
        except Exception as e:
            logger.debug(f"Strategy {key} error: {e}")
    return triggered


def check_momentum_breakout(df: pd.DataFrame, ind: dict) -> dict:
    """
    Strategy 1: Momentum Breakout
    - Price > EMA20
    - EMA20 > EMA50
    - RSI 55-70
    - MACD bullish crossover (recent)
    - Volume > 1.5x average
    - Breakout above 20-day high
    """
    try:
        price = ind.get("Close", 0)
        ema20 = ind.get("EMA20", 0)
        ema50 = ind.get("EMA50", 0)
        rsi = ind.get("RSI", 50)
        macd = ind.get("MACD", 0)
        macd_signal = ind.get("MACD_Signal", 0)
        vol_spike = ind.get("Vol_Spike", 1.0)
        breakout = ind.get("Breakout_20d", 0)
        adx = ind.get("ADX", 20)

        # Check for recent MACD crossover (last 3 bars)
        recent_macd_cross = False
        if len(df) >= 3:
            for i in range(-3, 0):
                if df.iloc[i].get("MACD_Cross_Up", 0) == 1 if "MACD_Cross_Up" in df.columns else False:
                    recent_macd_cross = True
                    break
            # Also check if MACD > Signal
            if "MACD" in df.columns and "MACD_Signal" in df.columns:
                recent_macd_cross = recent_macd_cross or (macd > macd_signal and df["MACD"].iloc[-2] <= df["MACD_Signal"].iloc[-2])

        conditions = {
            "price_above_ema20": price > ema20 > 0,
            "ema20_above_ema50": ema20 > ema50 > 0,
            "rsi_in_range": 55 <= rsi <= 75,
            "macd_bullish": macd > macd_signal,
            "volume_expansion": vol_spike >= 1.5,
            "breakout_20d": breakout == 1 or (price > 0 and price >= df["High"].rolling(20).max().iloc[-2] if "High" in df.columns else False),
        }

        met = sum(conditions.values())
        triggered = met >= 5  # Need at least 5/6 conditions

        strength = min(100, int(met / 6 * 100) + (10 if adx > 25 else 0))

        return {
            "triggered": triggered,
            "strength": strength,
            "details": conditions,
            "description": f"Momentum breakout with {met}/6 conditions met. ADX: {adx:.1f}",
        }
    except Exception as e:
        return {"triggered": False, "strength": 0, "details": {}, "description": str(e)}


def check_pullback_uptrend(df: pd.DataFrame, ind: dict) -> dict:
    """
    Strategy 2: Pullback in Uptrend
    - Price above EMA50
    - Pullback near EMA20 (within 2%)
    - RSI bouncing from 40 (was below 45, now above 45)
    - Bullish candle
    """
    try:
        price = ind.get("Close", 0)
        ema20 = ind.get("EMA20", 0)
        ema50 = ind.get("EMA50", 0)
        rsi = ind.get("RSI", 50)
        hammer = ind.get("Hammer", 0)
        bull_engulf = ind.get("Bull_Engulf", 0)
        adx = ind.get("ADX", 20)

        # RSI bounce: was below 48, now above 45
        rsi_bounce = False
        if "RSI" in df.columns and len(df) >= 3:
            prev_rsi = df["RSI"].iloc[-3:-1].min() if len(df) >= 3 else 50
            rsi_bounce = prev_rsi < 48 and rsi > 45

        near_ema20 = abs(price - ema20) / (ema20 + 1e-10) < 0.02 if ema20 > 0 else False

        conditions = {
            "price_above_ema50": price > ema50 > 0,
            "near_ema20": near_ema20,
            "rsi_bounce": rsi_bounce or (40 <= rsi <= 55),
            "bullish_candle": (hammer == 1 or bull_engulf == 1 or
                               (ind.get("Close", 0) > ind.get("Open", 0))),
            "uptrend": ind.get("EMA20", 0) > ind.get("EMA50", 0),
        }

        met = sum(conditions.values())
        triggered = met >= 4

        strength = min(100, int(met / 5 * 100) + (5 if adx > 20 else 0))

        return {
            "triggered": triggered,
            "strength": strength,
            "details": conditions,
            "description": f"Pullback in uptrend with {met}/5 conditions met.",
        }
    except Exception as e:
        return {"triggered": False, "strength": 0, "details": {}, "description": str(e)}


def check_volatility_squeeze(df: pd.DataFrame, ind: dict) -> dict:
    """
    Strategy 3: Volatility Squeeze Breakout
    - Bollinger Band squeeze (width < 20-period avg)
    - Volume expansion on breakout candle
    - Price breaking out of range
    """
    try:
        bb_squeeze = ind.get("BB_Squeeze", 0)
        vol_spike = ind.get("Vol_Spike", 1.0)
        price = ind.get("Close", 0)
        bb_upper = ind.get("BB_Upper", 0)
        bb_lower = ind.get("BB_Lower", 0)
        bb_middle = ind.get("BB_Middle", 0)
        bb_pct = ind.get("BB_Percent", 0.5)

        # Recent squeeze: BB was narrow in last 5 bars
        recent_squeeze = False
        if "BB_Width" in df.columns and len(df) >= 6:
            recent_bw = df["BB_Width"].iloc[-6:-1]
            recent_squeeze = (recent_bw < recent_bw.rolling(20, min_periods=3).mean()).any()

        # Breakout direction
        breakout_up = price > bb_upper if bb_upper > 0 else False
        approaching_upper = bb_pct > 0.85 if bb_pct else False

        conditions = {
            "bb_squeeze": bb_squeeze == 1 or recent_squeeze,
            "volume_expansion": vol_spike >= 1.3,
            "price_breakout": breakout_up or approaching_upper,
            "momentum_up": ind.get("RSI", 50) > 50,
        }

        met = sum(conditions.values())
        triggered = met >= 3

        strength = min(100, int(met / 4 * 100) + (int((vol_spike - 1) * 20) if vol_spike > 1 else 0))

        return {
            "triggered": triggered,
            "strength": strength,
            "details": conditions,
            "description": f"Volatility squeeze breakout. BB Squeeze: {bb_squeeze}, Vol: {vol_spike:.1f}x",
        }
    except Exception as e:
        return {"triggered": False, "strength": 0, "details": {}, "description": str(e)}


def check_reversal_swing(df: pd.DataFrame, ind: dict) -> dict:
    """
    Strategy 4: Reversal Swing
    - RSI < 30 then crosses above 35
    - MACD crossover
    - Hammer or bullish engulfing candle
    """
    try:
        rsi = ind.get("RSI", 50)
        macd = ind.get("MACD", 0)
        macd_signal = ind.get("MACD_Signal", 0)
        hammer = ind.get("Hammer", 0)
        bull_engulf = ind.get("Bull_Engulf", 0)

        # RSI oversold recovery
        rsi_recovery = False
        if "RSI" in df.columns and len(df) >= 5:
            min_rsi_recent = df["RSI"].iloc[-5:-1].min()
            rsi_recovery = min_rsi_recent < 35 and rsi > 35

        # MACD crossover (recent)
        macd_cross = False
        if "MACD" in df.columns and "MACD_Signal" in df.columns and len(df) >= 2:
            macd_cross = (macd > macd_signal and
                          df["MACD"].iloc[-2] <= df["MACD_Signal"].iloc[-2])

        conditions = {
            "rsi_oversold_recovery": rsi_recovery,
            "rsi_range": 30 <= rsi <= 50,
            "macd_cross": macd_cross or (macd > macd_signal),
            "bullish_reversal_candle": hammer == 1 or bull_engulf == 1,
        }

        met = sum(conditions.values())
        triggered = met >= 3 and rsi_recovery  # RSI recovery is mandatory

        strength = min(100, int(met / 4 * 100) + (20 if rsi_recovery and macd_cross else 0))

        return {
            "triggered": triggered,
            "strength": strength,
            "details": conditions,
            "description": f"Reversal swing. RSI: {rsi:.1f}, MACD cross: {macd_cross}",
        }
    except Exception as e:
        return {"triggered": False, "strength": 0, "details": {}, "description": str(e)}


def generate_trade_plan(df: pd.DataFrame, ind: dict, strategy_name: str) -> dict:
    """
    Generate a trade plan: entry, stop loss, targets, risk reward.
    """
    try:
        price = ind.get("Close", 0)
        atr = ind.get("ATR", price * 0.02)
        high20 = df["High"].rolling(20).max().iloc[-1] if "High" in df.columns else price * 1.05
        low20 = df["Low"].rolling(20).min().iloc[-1] if "Low" in df.columns else price * 0.95

        if price <= 0:
            return {}

        if "BREAKOUT" in strategy_name.upper():
            entry = round(high20 * 1.001, 2)  # Just above resistance
            stop_loss = round(entry - 1.5 * atr, 2)
        elif "PULLBACK" in strategy_name.upper():
            entry = round(price * 1.001, 2)
            stop_loss = round(entry - 1.2 * atr, 2)
        elif "SQUEEZE" in strategy_name.upper():
            entry = round(price * 1.002, 2)
            stop_loss = round(entry - 1.5 * atr, 2)
        else:  # Reversal
            entry = round(price * 1.001, 2)
            stop_loss = round(entry - 1.0 * atr, 2)

        risk = entry - stop_loss
        if risk <= 0:
            risk = atr

        target1 = round(entry + 2 * risk, 2)  # 2R
        target2 = round(entry + 3 * risk, 2)  # 3R

        rr = round((target1 - entry) / risk, 2) if risk > 0 else 0

        return {
            "entry": entry,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "risk_reward": rr,
            "risk_amount": round(risk, 2),
            "atr": round(atr, 2),
        }
    except Exception as e:
        logger.debug(f"Trade plan error: {e}")
        return {
            "entry": price,
            "stop_loss": round(price * 0.97, 2),
            "target1": round(price * 1.05, 2),
            "target2": round(price * 1.08, 2),
            "risk_reward": 1.5,
            "risk_amount": round(price * 0.03, 2),
            "atr": round(price * 0.02, 2),
        }
