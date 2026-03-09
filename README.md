# 📈 AI Swing Trader Pro — NSE India

> Professional AI-powered swing trading platform for the Indian stock market.
> Runs completely **locally** on your computer.

---

## 🚀 Quick Start

### 1. Install Python 3.9+
Download from https://python.org

### 2. Install Dependencies
```bash
cd ai_swing_trader
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## 🔑 Features

| Feature | Description |
|---------|-------------|
| 🔍 **Scanner** | Scan NIFTY50/100/500/All NSE stocks |
| 📊 **Stock Analysis** | Candlestick + EMA + BB + RSI + MACD + Supertrend |
| 🤖 **AI Scoring** | 0-100 score based on 5 weighted factors |
| 🔮 **ML Prediction** | Random Forest — prob of 5%/8%/10% move |
| 🏦 **Institutional Flow** | FII/DII tracking with sentiment indicator |
| ⚙️ **Options Chain** | PCR, Max Pain, OI buildup, support/resistance |
| 📋 **Backtesting** | Strategy backtester with equity curve |
| 👁️ **Watchlist** | Save and scan custom stock lists |
| 🔔 **Alerts** | Telegram bot + desktop notifications |

---

## 📐 Strategies

### Strategy 1: Momentum Breakout
- Price > EMA20 > EMA50
- RSI 55–70
- Volume > 1.5× average
- 20-day high breakout

### Strategy 2: Pullback in Uptrend
- Price above EMA50
- Retest of EMA20
- RSI bounce from 40
- Bullish candle pattern

### Strategy 3: Volatility Squeeze Breakout
- Bollinger Band squeeze
- Volume expansion
- Price breaking upper band

### Strategy 4: Reversal Swing
- RSI < 30 → cross above 35
- MACD bullish crossover
- Hammer / bullish engulfing

---

## 🤖 AI Score Components

| Factor | Weight |
|--------|--------|
| Trend Strength (EMA alignment) | 25% |
| Momentum Strength (RSI/MACD) | 25% |
| Volume Expansion | 20% |
| Breakout Quality | 15% |
| Institutional Participation (FII) | 15% |

---

## 📱 Telegram Alerts Setup

1. Message @BotFather on Telegram
2. Create a new bot: `/newbot`
3. Copy your bot token
4. Get your chat ID from @userinfobot
5. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token_here"
   export TELEGRAM_CHAT_ID="your_chat_id_here"
   ```

---

## 🗂️ Project Structure

```
ai_swing_trader/
├── app.py                   # Main Streamlit dashboard
├── data_fetcher.py          # yfinance + NSE API data fetching
├── indicators.py            # 14+ technical indicators
├── strategies.py            # 4 swing trading strategies
├── scanner.py               # Parallel stock scanner
├── ai_model.py              # ML prediction engine (Random Forest)
├── ranking_engine.py        # AI scoring system
├── backtester.py            # Strategy backtesting
├── options_analyzer.py      # Options chain analysis
├── institutional_tracker.py # FII/DII flow tracker
├── alerts.py                # Telegram + desktop alerts
├── database.py              # SQLite local database
├── utils.py                 # Utilities + NSE symbol lists
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚡ Performance Notes

- Scanner uses **multiprocessing** for parallel analysis
- Data cached in-memory (5 min TTL) + SQLite
- Nifty 50: ~10-15 seconds
- Nifty 100: ~20-30 seconds
- Nifty 500: ~2-3 minutes
- All NSE (2000+): ~10-15 minutes

---

## ⚠️ Disclaimer

This platform is for **educational and research purposes only**.
It does not constitute financial advice. Always do your own research
before making investment decisions. Past performance does not guarantee future results.

---

## 📦 Dependencies

```
streamlit >= 1.28
pandas >= 2.0
numpy >= 1.24
pandas-ta >= 0.3.14b
yfinance >= 0.2.31
plotly >= 5.17
requests >= 2.31
scikit-learn >= 1.3
joblib >= 1.3
scipy >= 1.11
plyer >= 2.1        (desktop notifications)
python-telegram-bot (Telegram alerts, optional)
sqlalchemy >= 2.0
```
# ai-swing-trader
