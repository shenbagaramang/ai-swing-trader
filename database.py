"""
database.py - SQLite database management for AI Swing Trader
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "swing_trader.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize all database tables."""
    conn = get_connection()
    c = conn.cursor()

    # Stock universe table
    c.execute("""
        CREATE TABLE IF NOT EXISTS stock_universe (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            industry TEXT,
            market_cap REAL,
            index_membership TEXT,
            last_updated TEXT
        )
    """)

    # Watchlist table
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE,
            added_date TEXT,
            notes TEXT
        )
    """)

    # Scan results table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            symbol TEXT,
            price REAL,
            rsi REAL,
            volume_spike REAL,
            strategy TEXT,
            ai_score REAL,
            prob_5pct REAL,
            prob_8pct REAL,
            prob_10pct REAL,
            entry REAL,
            stop_loss REAL,
            target1 REAL,
            target2 REAL,
            risk_reward REAL,
            raw_data TEXT
        )
    """)

    # Alerts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            alert_type TEXT,
            condition TEXT,
            target_value REAL,
            is_active INTEGER DEFAULT 1,
            created_date TEXT,
            triggered_date TEXT
        )
    """)

    # Price cache table
    c.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # Backtests table
    c.execute("""
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            symbol TEXT,
            strategy TEXT,
            start_date TEXT,
            end_date TEXT,
            win_rate REAL,
            avg_return REAL,
            profit_factor REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            results_json TEXT
        )
    """)

    conn.commit()
    conn.close()


# ─── Watchlist ────────────────────────────────────────────────────────────────

def add_to_watchlist(symbol: str, notes: str = "") -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, added_date, notes) VALUES (?, ?, ?)",
            (symbol.upper(), datetime.now().isoformat(), notes)
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def remove_from_watchlist(symbol: str) -> bool:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def get_watchlist() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT symbol, added_date, notes FROM watchlist ORDER BY added_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Scan Results ─────────────────────────────────────────────────────────────

def save_scan_results(results: list):
    if not results:
        return
    conn = get_connection()
    scan_date = datetime.now().isoformat()
    for r in results:
        conn.execute("""
            INSERT INTO scan_results
            (scan_date, symbol, price, rsi, volume_spike, strategy, ai_score,
             prob_5pct, prob_8pct, prob_10pct, entry, stop_loss, target1, target2,
             risk_reward, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_date,
            r.get("symbol", ""),
            r.get("price", 0),
            r.get("rsi", 0),
            r.get("volume_spike", 0),
            r.get("strategy", ""),
            r.get("ai_score", 0),
            r.get("prob_5pct", 0),
            r.get("prob_8pct", 0),
            r.get("prob_10pct", 0),
            r.get("entry", 0),
            r.get("stop_loss", 0),
            r.get("target1", 0),
            r.get("target2", 0),
            r.get("risk_reward", 0),
            json.dumps(r)
        ))
    conn.commit()
    conn.close()


def get_latest_scan() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT * FROM scan_results
        WHERE scan_date = (SELECT MAX(scan_date) FROM scan_results)
        ORDER BY ai_score DESC
    """, conn)
    conn.close()
    return df


# ─── Stock Universe ───────────────────────────────────────────────────────────

def save_stock_universe(stocks: list):
    conn = get_connection()
    now = datetime.now().isoformat()
    for s in stocks:
        conn.execute("""
            INSERT OR REPLACE INTO stock_universe
            (symbol, name, sector, industry, market_cap, index_membership, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            s.get("symbol", ""),
            s.get("name", ""),
            s.get("sector", ""),
            s.get("industry", ""),
            s.get("market_cap", 0),
            s.get("index_membership", ""),
            now
        ))
    conn.commit()
    conn.close()


def get_stock_universe(index_filter: str = "ALL") -> pd.DataFrame:
    conn = get_connection()
    if index_filter == "ALL":
        df = pd.read_sql_query("SELECT * FROM stock_universe ORDER BY symbol", conn)
    else:
        df = pd.read_sql_query(
            "SELECT * FROM stock_universe WHERE index_membership LIKE ? ORDER BY symbol",
            conn, params=(f"%{index_filter}%",)
        )
    conn.close()
    return df


# ─── Alerts ───────────────────────────────────────────────────────────────────

def save_alert(symbol: str, alert_type: str, condition: str, target_value: float):
    conn = get_connection()
    conn.execute("""
        INSERT INTO alerts (symbol, alert_type, condition, target_value, is_active, created_date)
        VALUES (?, ?, ?, ?, 1, ?)
    """, (symbol.upper(), alert_type, condition, target_value, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_active_alerts() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM alerts WHERE is_active = 1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_alert_triggered(alert_id: int):
    conn = get_connection()
    conn.execute(
        "UPDATE alerts SET is_active = 0, triggered_date = ? WHERE id = ?",
        (datetime.now().isoformat(), alert_id)
    )
    conn.commit()
    conn.close()


# ─── Price Cache ──────────────────────────────────────────────────────────────

def cache_prices(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    conn = get_connection()
    for date, row in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO price_cache (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            str(date.date()) if hasattr(date, 'date') else str(date),
            float(row.get("Open", 0)),
            float(row.get("High", 0)),
            float(row.get("Low", 0)),
            float(row.get("Close", 0)),
            float(row.get("Volume", 0))
        ))
    conn.commit()
    conn.close()


def get_cached_prices(symbol: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM price_cache WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,)
    )
    conn.close()
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


# ─── Backtests ────────────────────────────────────────────────────────────────

def save_backtest(symbol, strategy, start_date, end_date, metrics: dict, trades: list):
    conn = get_connection()
    conn.execute("""
        INSERT INTO backtests
        (run_date, symbol, strategy, start_date, end_date,
         win_rate, avg_return, profit_factor, max_drawdown, total_trades, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        symbol, strategy, start_date, end_date,
        metrics.get("win_rate", 0),
        metrics.get("avg_return", 0),
        metrics.get("profit_factor", 0),
        metrics.get("max_drawdown", 0),
        metrics.get("total_trades", 0),
        json.dumps({"metrics": metrics, "trades": trades})
    ))
    conn.commit()
    conn.close()


def get_backtest_history() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM backtests ORDER BY run_date DESC LIMIT 50", conn
    )
    conn.close()
    return df


# Initialize on import
init_database()
