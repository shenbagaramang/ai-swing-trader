"""
ranking_engine.py - AI Trade Scoring and Ranking Engine
Scores stocks from 0-100 based on multiple factors
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Score weights (must sum to 100)
SCORE_WEIGHTS = {
    "trend_strength": 25,
    "momentum_strength": 25,
    "volume_expansion": 20,
    "breakout_quality": 15,
    "institutional": 15,
}


def calculate_ai_score(
    indicators: dict,
    strategies_triggered: list,
    ml_probs: dict,
    fii_sentiment: str = "neutral",
) -> dict:
    """
    Calculate composite AI score (0-100) for a stock.

    Parameters:
        indicators: dict of latest indicator values
        strategies_triggered: list of triggered strategies
        ml_probs: dict with prob_5pct, prob_8pct, prob_10pct
        fii_sentiment: 'bullish', 'bearish', 'neutral'

    Returns:
        dict with total_score and component scores
    """
    scores = {}

    # ── 1. Trend Strength (25%) ───────────────────────────────────────────────
    trend_score = 0
    ema_aligned = indicators.get("EMA_Aligned", 0)
    above_ema20 = indicators.get("Above_EMA20", 0)
    above_ema50 = indicators.get("Above_EMA50", 0)
    above_ema200 = indicators.get("Above_EMA200", 0)
    adx = indicators.get("ADX", 20)
    supertrend_dir = indicators.get("Supertrend_Dir", 0)

    # EMA alignment
    if ema_aligned:
        trend_score += 40
    elif above_ema50:
        trend_score += 20
    elif above_ema20:
        trend_score += 10

    # Position relative to EMAs
    trend_score += above_ema200 * 20

    # ADX strength
    if adx >= 30:
        trend_score += 25
    elif adx >= 25:
        trend_score += 15
    elif adx >= 20:
        trend_score += 8

    # Supertrend
    if supertrend_dir == 1:
        trend_score += 15

    scores["trend_strength"] = min(100, trend_score)

    # ── 2. Momentum Strength (25%) ────────────────────────────────────────────
    momentum_score = 0
    rsi = indicators.get("RSI", 50)
    macd = indicators.get("MACD", 0)
    macd_signal = indicators.get("MACD_Signal", 0)
    macd_hist = indicators.get("MACD_Hist", 0)
    stoch_k = indicators.get("Stoch_RSI_K", 50)

    # RSI scoring
    if 55 <= rsi <= 70:
        momentum_score += 35
    elif 50 <= rsi < 55:
        momentum_score += 20
    elif 70 < rsi <= 75:
        momentum_score += 15  # Strong but getting overbought
    elif 40 <= rsi < 50:
        momentum_score += 10  # Recovering
    elif rsi < 35:
        momentum_score += 5   # Potential reversal

    # MACD
    if macd > macd_signal:
        momentum_score += 25
        if macd_hist > 0 and macd_hist > indicators.get("MACD_Hist_prev", macd_hist * 0.9):
            momentum_score += 10  # Expanding histogram

    # Stochastic RSI
    if 50 <= stoch_k <= 80:
        momentum_score += 20
    elif stoch_k > 80:
        momentum_score += 10
    elif 20 <= stoch_k < 50:
        momentum_score += 8

    # Momentum oscillator
    mom = indicators.get("MOM", 0)
    if mom > 0:
        momentum_score += 10

    scores["momentum_strength"] = min(100, momentum_score)

    # ── 3. Volume Expansion (20%) ─────────────────────────────────────────────
    volume_score = 0
    vol_spike = indicators.get("Vol_Spike", 1.0)

    if vol_spike >= 3.0:
        volume_score = 100
    elif vol_spike >= 2.0:
        volume_score = 80
    elif vol_spike >= 1.5:
        volume_score = 60
    elif vol_spike >= 1.2:
        volume_score = 40
    elif vol_spike >= 1.0:
        volume_score = 25
    else:
        volume_score = 10

    scores["volume_expansion"] = volume_score

    # ── 4. Breakout Quality (15%) ─────────────────────────────────────────────
    breakout_score = 0
    breakout_20d = indicators.get("Breakout_20d", 0)
    bb_pct = indicators.get("BB_Percent", 0.5)
    bb_squeeze = indicators.get("BB_Squeeze", 0)

    if breakout_20d == 1:
        breakout_score += 50
    if bb_pct > 0.85:
        breakout_score += 20
    elif bb_pct > 0.7:
        breakout_score += 10
    if bb_squeeze:
        breakout_score += 15  # Recent squeeze adds quality
    if vol_spike >= 1.5 and breakout_20d:
        breakout_score += 15  # Volume-confirmed breakout

    scores["breakout_quality"] = min(100, breakout_score)

    # ── 5. Institutional Participation (15%) ──────────────────────────────────
    inst_score = 50  # Neutral default
    if fii_sentiment == "bullish":
        inst_score = 90
    elif fii_sentiment == "bearish":
        inst_score = 20
    else:
        inst_score = 50

    scores["institutional"] = inst_score

    # ── Compute Weighted Total ────────────────────────────────────────────────
    total = sum(
        scores[k] * SCORE_WEIGHTS[k] / 100
        for k in SCORE_WEIGHTS
    )
    total = min(100, max(0, total))

    # Strategy bonus
    strategy_bonus = min(10, len(strategies_triggered) * 3)
    total = min(100, total + strategy_bonus)

    # ML probability bonus
    prob5 = ml_probs.get("prob_5pct", 0)
    prob_bonus = min(5, prob5 * 10)
    total = min(100, total + prob_bonus)

    return {
        "total_score": round(total, 1),
        "trend_strength": round(scores["trend_strength"], 1),
        "momentum_strength": round(scores["momentum_strength"], 1),
        "volume_expansion": round(scores["volume_expansion"], 1),
        "breakout_quality": round(scores["breakout_quality"], 1),
        "institutional": round(scores["institutional"], 1),
        "strategy_bonus": strategy_bonus,
    }


def rank_stocks(scan_results: list, top_n: int = 20) -> list:
    """
    Rank and return top N stocks by AI score.
    """
    if not scan_results:
        return []
    sorted_results = sorted(
        scan_results,
        key=lambda x: x.get("ai_score", 0),
        reverse=True
    )
    return sorted_results[:top_n]


def get_score_label(score: float) -> tuple:
    """Return (label, color) for a given score."""
    if score >= 80:
        return "🔥 Strong Buy", "#00C853"
    elif score >= 65:
        return "✅ Buy", "#69F0AE"
    elif score >= 50:
        return "⚡ Watch", "#FFD740"
    elif score >= 35:
        return "⚠️ Weak", "#FF6D00"
    else:
        return "❌ Avoid", "#F44336"


def create_score_breakdown_text(score_dict: dict) -> str:
    """Create formatted breakdown of AI score components."""
    lines = [f"**Total AI Score: {score_dict.get('total_score', 0):.1f}/100**", ""]
    component_labels = {
        "trend_strength": f"Trend Strength (25%)",
        "momentum_strength": f"Momentum (25%)",
        "volume_expansion": f"Volume (20%)",
        "breakout_quality": f"Breakout Quality (15%)",
        "institutional": f"Institutional (15%)",
    }
    for key, label in component_labels.items():
        val = score_dict.get(key, 0)
        bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
        lines.append(f"{label}: {bar} {val:.0f}")
    return "\n".join(lines)
