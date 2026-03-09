"""
ai_model.py - AI/ML Price Move Prediction Engine
Uses Random Forest to predict probability of 5%/8%/10% moves
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "swing_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Hist", "ADX",
    "EMA_Aligned", "Above_EMA20", "Above_EMA50",
    "Vol_Spike", "ATR_Pct",
    "BB_Percent", "BB_Squeeze",
    "Stoch_RSI_K", "Hammer", "Bull_Engulf",
    "Breakout_20d", "Supertrend_Dir",
    "MOM",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    "Vol_Ratio_5d", "Price_to_High52w",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from indicator DataFrame."""
    df = df.copy()

    # Price returns
    df["Return_1d"] = df["Close"].pct_change(1) * 100
    df["Return_3d"] = df["Close"].pct_change(3) * 100
    df["Return_5d"] = df["Close"].pct_change(5) * 100
    df["Return_10d"] = df["Close"].pct_change(10) * 100

    # Volume ratio
    df["Vol_Ratio_5d"] = df["Volume"] / df["Volume"].rolling(5).mean().replace(0, np.nan)

    # Price to 52w high
    rolling_high = df["High"].rolling(252).max() if "High" in df.columns else df["Close"].rolling(252).max()
    df["Price_to_High52w"] = df["Close"] / rolling_high.replace(0, np.nan)

    # Fill missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df


def build_labels(df: pd.DataFrame, forward_days: int = 10) -> pd.Series:
    """
    Create binary label: did price move up by threshold% in next N days?
    Uses 5% threshold by default.
    """
    future_max = df["High"].rolling(forward_days).max().shift(-forward_days) if "High" in df.columns else \
                 df["Close"].shift(-forward_days)
    future_return = (future_max - df["Close"]) / df["Close"] * 100
    return future_return


def train_model(data_dict: dict) -> dict:
    """
    Train the AI model using historical data from multiple stocks.
    data_dict: {symbol: DataFrame with indicators}
    Returns training metrics.
    """
    logger.info("Starting AI model training...")

    all_features = []
    all_labels_5 = []
    all_labels_8 = []
    all_labels_10 = []

    for symbol, df in data_dict.items():
        if df is None or len(df) < 60:
            continue
        try:
            df_feat = build_features(df)
            labels = build_labels(df_feat)

            # Build feature rows (drop last 10 rows — no future data)
            feat_df = df_feat[FEATURE_COLS].iloc[:-10]
            lbl = labels.iloc[:-10]

            if len(feat_df) < 20:
                continue

            feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            all_features.append(feat_df)
            all_labels_5.append((lbl >= 5).astype(int))
            all_labels_8.append((lbl >= 8).astype(int))
            all_labels_10.append((lbl >= 10).astype(int))
        except Exception as e:
            logger.debug(f"Feature build error for {symbol}: {e}")

    if not all_features:
        logger.warning("No training data available. Using fallback model.")
        return _create_fallback_model()

    X = pd.concat(all_features, ignore_index=True)
    y5 = pd.concat(all_labels_5, ignore_index=True)
    y8 = pd.concat(all_labels_8, ignore_index=True)
    y10 = pd.concat(all_labels_10, ignore_index=True)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    logger.info(f"Training on {len(X)} samples from {len(data_dict)} stocks")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train separate models for each target
    models = {}
    metrics = {}
    for target_name, y in [("5pct", y5), ("8pct", y8), ("10pct", y10)]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            models[target_name] = model
            metrics[target_name] = {
                "accuracy": round(acc, 3),
                "samples": len(X),
                "positive_rate": float(y.mean()),
            }
            logger.info(f"Model {target_name}: accuracy={acc:.3f}")
        except Exception as e:
            logger.error(f"Model training error {target_name}: {e}")

    # Save models
    joblib.dump({"models": models, "scaler": scaler, "features": FEATURE_COLS}, MODEL_PATH)
    logger.info(f"Models saved to {MODEL_PATH}")

    return metrics


def predict_move_probability(df: pd.DataFrame) -> dict:
    """
    Predict probability of 5%, 8%, 10% move for a single stock.
    Returns dict with probability scores.
    """
    default = {"prob_5pct": 0.35, "prob_8pct": 0.25, "prob_10pct": 0.15, "model_used": "fallback"}

    try:
        if not os.path.exists(MODEL_PATH):
            return _rule_based_prediction(df)

        saved = joblib.load(MODEL_PATH)
        models = saved["models"]
        scaler = saved["scaler"]

        df_feat = build_features(df)
        row = df_feat[FEATURE_COLS].iloc[[-1]]
        row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(row)

        probs = {}
        for target_name in ["5pct", "8pct", "10pct"]:
            if target_name in models:
                prob = models[target_name].predict_proba(X_scaled)[0]
                probs[f"prob_{target_name}"] = round(float(prob[1] if len(prob) > 1 else prob[0]), 3)
            else:
                probs[f"prob_{target_name}"] = 0.25

        probs["model_used"] = "random_forest"
        return probs

    except Exception as e:
        logger.debug(f"Prediction error: {e}")
        return _rule_based_prediction(df)


def _rule_based_prediction(df: pd.DataFrame) -> dict:
    """
    Rule-based prediction when ML model is not available.
    Uses indicator combinations to estimate probability.
    """
    try:
        df_feat = build_features(df)
        ind = df_feat.iloc[-1]

        score = 0.0
        max_score = 0.0

        def add(condition, weight):
            nonlocal score, max_score
            max_score += weight
            if condition:
                score += weight

        rsi = float(ind.get("RSI", 50) or 50)
        add(55 <= rsi <= 70, 2)
        add(ind.get("EMA_Aligned", 0) == 1, 2)
        add(ind.get("Vol_Spike", 1) >= 1.5, 2)
        add(ind.get("MACD", 0) > ind.get("MACD_Signal", 0), 1.5)
        add(ind.get("ADX", 20) > 25, 1.5)
        add(ind.get("Breakout_20d", 0) == 1, 2)
        add(ind.get("Supertrend_Dir", 0) == 1, 1.5)
        add(ind.get("BB_Percent", 0.5) > 0.7, 1)

        base_prob = score / max_score if max_score > 0 else 0.3
        base_prob = min(0.85, max(0.1, base_prob))

        return {
            "prob_5pct": round(base_prob * 0.9, 3),
            "prob_8pct": round(base_prob * 0.65, 3),
            "prob_10pct": round(base_prob * 0.45, 3),
            "model_used": "rule_based",
        }
    except Exception:
        return {"prob_5pct": 0.35, "prob_8pct": 0.25, "prob_10pct": 0.15, "model_used": "fallback"}


def _create_fallback_model() -> dict:
    """Create a simple fallback model when training data is insufficient."""
    return {
        "5pct": {"accuracy": 0.0, "samples": 0, "positive_rate": 0},
        "8pct": {"accuracy": 0.0, "samples": 0, "positive_rate": 0},
        "10pct": {"accuracy": 0.0, "samples": 0, "positive_rate": 0},
    }


def get_feature_importance() -> dict:
    """Get feature importance from trained model."""
    try:
        if not os.path.exists(MODEL_PATH):
            return {}
        saved = joblib.load(MODEL_PATH)
        models = saved["models"]
        features = saved.get("features", FEATURE_COLS)
        if "5pct" in models:
            importances = models["5pct"].feature_importances_
            return dict(sorted(
                zip(features, importances),
                key=lambda x: x[1], reverse=True
            ))
    except Exception:
        pass
    return {}


def is_model_trained() -> bool:
    return os.path.exists(MODEL_PATH)
