"""
alerts.py - Alert System for AI Swing Trader
Supports Telegram bot and desktop notifications
"""

import logging
import os
import json
from datetime import datetime
import requests
from database import get_active_alerts, mark_alert_triggered, save_alert

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ─── Alert Checking ───────────────────────────────────────────────────────────

def check_alerts(scan_results: list) -> list:
    """
    Check all active alerts against current scan results.
    Returns list of triggered alerts.
    """
    active_alerts = get_active_alerts()
    if not active_alerts:
        return []

    price_map = {r["display_symbol"]: r for r in scan_results}
    triggered = []

    for alert in active_alerts:
        symbol = alert.get("symbol", "")
        alert_type = alert.get("alert_type", "")
        target = float(alert.get("target_value", 0) or 0)

        stock_data = price_map.get(symbol) or price_map.get(symbol.replace(".NS", ""))
        if not stock_data:
            continue

        fired = False
        msg = ""

        if alert_type == "PRICE_ABOVE":
            if stock_data.get("price", 0) >= target:
                fired = True
                msg = f"🔔 {symbol} price {stock_data['price']:.2f} crossed above {target:.2f}"

        elif alert_type == "PRICE_BELOW":
            if stock_data.get("price", 0) <= target:
                fired = True
                msg = f"🔔 {symbol} price {stock_data['price']:.2f} dropped below {target:.2f}"

        elif alert_type == "AI_SCORE":
            if stock_data.get("ai_score", 0) >= target:
                fired = True
                msg = f"🤖 {symbol} AI Score {stock_data['ai_score']:.0f} exceeded {target:.0f}"

        elif alert_type == "BREAKOUT":
            if stock_data.get("breakout_20d", 0) == 1:
                fired = True
                msg = f"📈 {symbol} BREAKOUT! Price: {stock_data['price']:.2f}"

        elif alert_type == "STRATEGY":
            if stock_data.get("strategies_count", 0) > 0:
                fired = True
                msg = f"⚡ {symbol} triggered strategy: {stock_data['strategy']}"

        if fired:
            mark_alert_triggered(alert["id"])
            alert_result = {
                "symbol": symbol,
                "alert_type": alert_type,
                "message": msg,
                "stock_data": stock_data,
                "triggered_at": datetime.now().isoformat(),
            }
            triggered.append(alert_result)

            # Fire notifications
            _send_notifications(msg, stock_data)

    return triggered


def _send_notifications(message: str, stock_data: dict):
    """Send notifications via all configured channels."""
    _send_desktop_notification(message)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        _send_telegram(message, stock_data)


def _send_desktop_notification(message: str):
    """Send desktop notification."""
    try:
        from plyer import notification
        notification.notify(
            title="AI Swing Trader Alert",
            message=message[:200],
            app_name="AI Swing Trader",
            timeout=10,
        )
    except Exception as e:
        logger.debug(f"Desktop notification failed: {e}")


def _send_telegram(message: str, stock_data: dict = None):
    """Send Telegram message."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        full_msg = message
        if stock_data:
            full_msg += f"\n\n📊 Details:"
            full_msg += f"\nPrice: ₹{stock_data.get('price', 0):.2f}"
            full_msg += f"\nRSI: {stock_data.get('rsi', 0):.1f}"
            full_msg += f"\nAI Score: {stock_data.get('ai_score', 0):.0f}/100"
            full_msg += f"\nEntry: ₹{stock_data.get('entry', 0):.2f}"
            full_msg += f"\nStop Loss: ₹{stock_data.get('stop_loss', 0):.2f}"
            full_msg += f"\nTarget 1: ₹{stock_data.get('target1', 0):.2f}"
            full_msg += f"\nRisk/Reward: {stock_data.get('risk_reward', 0):.1f}x"

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": full_msg,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


def send_daily_summary(top_stocks: list):
    """Send daily summary of top swing trade opportunities."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    msg = f"📊 <b>AI Swing Trader - Daily Summary</b>\n"
    msg += f"🕐 {datetime.now().strftime('%d %b %Y %H:%M')}\n\n"
    msg += f"🔥 <b>Top 5 Swing Opportunities:</b>\n\n"

    for i, stock in enumerate(top_stocks[:5], 1):
        msg += f"{i}. <b>{stock.get('display_symbol', '')}</b>\n"
        msg += f"   Price: ₹{stock.get('price', 0):.2f} | Score: {stock.get('ai_score', 0):.0f}/100\n"
        msg += f"   Strategy: {stock.get('strategy', '')}\n"
        msg += f"   Target: ₹{stock.get('target1', 0):.2f} | SL: ₹{stock.get('stop_loss', 0):.2f}\n\n"

    _send_telegram(msg)


def add_price_alert(symbol: str, price: float, direction: str = "above"):
    alert_type = "PRICE_ABOVE" if direction == "above" else "PRICE_BELOW"
    condition = f"Price {direction} {price}"
    save_alert(symbol, alert_type, condition, price)


def add_score_alert(symbol: str, score_threshold: float = 80):
    save_alert(symbol, "AI_SCORE", f"AI Score >= {score_threshold}", score_threshold)


def add_breakout_alert(symbol: str):
    save_alert(symbol, "BREAKOUT", "20-day breakout", 0)


def add_strategy_alert(symbol: str):
    save_alert(symbol, "STRATEGY", "Any strategy triggered", 0)


def test_telegram_connection() -> bool:
    """Test if Telegram credentials are valid."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        resp = requests.get(url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
