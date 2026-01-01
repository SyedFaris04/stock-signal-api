from typing import Any, Dict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# load saved models
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("rf_model.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

# Original features (preserved)
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d",
    "ma_10", "ma_20", "ma_ratio",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_up", "bb_mid", "bb_low", "bb_width",
    "atr_14",
    "volume_z",
]

# Enhanced features (optional - only use if you retrain with these)
ENHANCED_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d",
    "ma_10", "ma_20", "ma_ratio",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_up", "bb_mid", "bb_low", "bb_width",
    "atr_14",
    "volume_z",
    # New features for better prediction
    "stoch_k", "stoch_d",
    "roc_10",
    "momentum_10",
    "williams_r",
    "volume_ratio",
    "dist_from_ma10", "dist_from_ma20",
    "ma_slope_10",
]

# Use original features by default (change to ENHANCED_FEATURES if you retrain)
ACTIVE_FEATURES = FEATURES

LOOKBACK_DAYS = 60

# Ensemble weights - optimized for better performance
# LSTM is typically better at time series patterns
WEIGHT_RF = 0.3  # Random Forest weight
WEIGHT_LSTM = 0.7  # LSTM weight


# ---- feature helpers ----
def compute_rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger(series, window=20, num_std=2):
    """Bollinger Bands"""
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-9)
    return upper, mid, lower, width


def compute_atr(df, window=14):
    """Average True Range"""
    high = df["High"]
    low = df["Low"]
    close = df["price"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_stochastic(df, window=14):
    """Stochastic Oscillator (%K and %D)"""
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    stoch_k = 100 * (df['price'] - low_min) / (high_max - low_min + 1e-9)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d


def compute_williams_r(df, window=14):
    """Williams %R Momentum Indicator"""
    high_max = df['High'].rolling(window=window).max()
    low_min = df['Low'].rolling(window=window).min()
    williams_r = -100 * (high_max - df['price']) / (high_max - low_min + 1e-9)
    return williams_r


# ---- request/response models ----
class SignalRequest(BaseModel):
    ticker: str


class SignalResponse(BaseModel):
    ticker: str
    date: str
    price: float
    proba: float
    signal: int
    action: str
    explanation: str


app = FastAPI(title="Stock Signal API - Enhanced")

# ---- CORS configuration ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def last_lookback_window(ticker: str, enhanced=False):
    """
    Fetch and compute technical indicators for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        enhanced: If True, compute additional features (requires model retraining)
    
    Returns:
        DataFrame with computed features or None if insufficient data
    """
    end = datetime.utcnow()
    start = end - timedelta(days=730)  # 2 years for better indicator calculation
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["price"] = df["Close"]

    # Basic returns and volatility (PRESERVED)
    df["ret_1d"] = df["price"].pct_change()
    df["ret_5d"] = df["price"].pct_change(5)
    df["ret_20d"] = df["price"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    
    # Moving averages (PRESERVED)
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["ma_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # RSI (PRESERVED)
    df["rsi"] = compute_rsi(df["price"], window=14)

    # MACD (PRESERVED)
    macd, macd_sig, macd_hist = compute_macd(df["price"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    # Bollinger Bands (PRESERVED)
    bb_up, bb_mid, bb_low, bb_width = compute_bollinger(df["price"])
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low
    df["bb_width"] = bb_width

    # ATR (PRESERVED)
    df["atr_14"] = compute_atr(df, window=14)
    
    # Volume Z-score (PRESERVED)
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std() + 1e-9
    )

    # Enhanced features (NEW - only computed if enhanced=True)
    if enhanced:
        # Stochastic Oscillator
        stoch_k, stoch_d = compute_stochastic(df, window=14)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        
        # Rate of Change
        df["roc_10"] = ((df["price"] - df["price"].shift(10)) / (df["price"].shift(10) + 1e-9)) * 100
        
        # Momentum
        df["momentum_10"] = df["price"] - df["price"].shift(10)
        
        # Williams %R
        df["williams_r"] = compute_williams_r(df, window=14)
        
        # Volume ratio
        df["volume_ma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma_20"] + 1e-9)
        
        # Distance from moving averages
        df["dist_from_ma10"] = (df["price"] - df["ma_10"]) / (df["ma_10"] + 1e-9)
        df["dist_from_ma20"] = (df["price"] - df["ma_20"]) / (df["ma_20"] + 1e-9)
        
        # MA slope (trend strength)
        df["ma_slope_10"] = df["ma_10"].diff(5) / (df["ma_10"] + 1e-9)

    df = df.dropna()
    if len(df) < LOOKBACK_DAYS:
        return None

    return df.iloc[-LOOKBACK_DAYS:]


def action_from_signal(sig: int) -> str:
    return "BUY" if sig == 1 else "NO_POSITION"


def build_explanation(row: pd.Series) -> str:
    """
    Generate a human-readable explanation based on technical indicators.
    Enhanced with additional context.
    """
    msgs = []

    # RSI Analysis
    rsi = float(row.get("rsi", np.nan))
    if not np.isnan(rsi):
        if rsi > 70:
            msgs.append(f"RSI is {rsi:.1f}, indicating overbought conditions - caution advised.")
        elif rsi < 30:
            msgs.append(f"RSI is {rsi:.1f}, indicating oversold conditions - potential buying opportunity.")
        else:
            msgs.append(f"RSI is {rsi:.1f}, showing neutral momentum.")

    # Moving Average Trend
    ma10 = float(row.get("ma_10", np.nan))
    ma20 = float(row.get("ma_20", np.nan))
    price = float(row.get("price", np.nan))
    
    if not np.isnan(ma10) and not np.isnan(ma20) and not np.isnan(price):
        if ma10 > ma20 and price > ma10:
            msgs.append("Strong bullish trend: price above both 10-day and 20-day moving averages.")
        elif ma10 > ma20:
            msgs.append("Short-term uptrend detected with 10-day MA above 20-day MA.")
        elif ma10 < ma20 and price < ma10:
            msgs.append("Strong bearish trend: price below both moving averages.")
        elif ma10 < ma20:
            msgs.append("Short-term downtrend with 10-day MA below 20-day MA.")

    # Volatility Assessment
    vol20 = float(row.get("vol_20d", np.nan))
    if not np.isnan(vol20):
        if vol20 > 0.03:
            msgs.append("High volatility detected - larger price swings expected, higher risk.")
        elif vol20 < 0.015:
            msgs.append("Low volatility environment - more stable price action.")

    # MACD Momentum
    macd_val = float(row.get("macd", np.nan))
    macd_sig = float(row.get("macd_signal", np.nan))
    if not np.isnan(macd_val) and not np.isnan(macd_sig):
        if macd_val > macd_sig and macd_val > 0:
            msgs.append("MACD shows strong bullish momentum with positive crossover.")
        elif macd_val > macd_sig:
            msgs.append("MACD above signal line - bullish momentum building.")
        elif macd_val < macd_sig and macd_val < 0:
            msgs.append("MACD shows strong bearish momentum with negative crossover.")
        elif macd_val < macd_sig:
            msgs.append("MACD below signal line - bearish momentum detected.")

    # Bollinger Bands Position
    bb_up = float(row.get("bb_up", np.nan))
    bb_low = float(row.get("bb_low", np.nan))
    if not np.isnan(bb_up) and not np.isnan(bb_low) and not np.isnan(price):
        if price > bb_up:
            msgs.append("Price breaking above upper Bollinger Band - potential overbought.")
        elif price < bb_low:
            msgs.append("Price breaking below lower Bollinger Band - potential oversold.")

    if not msgs:
        return "Technical indicators show mixed signals. Consider multiple factors before trading."

    return " ".join(msgs)


@app.post("/signal", response_model=SignalResponse)
def get_signal(req: SignalRequest):
    """
    POST endpoint for signal prediction (original format).
    """
    window = last_lookback_window(req.ticker, enhanced=False)
    if window is None:
        return SignalResponse(
            ticker=req.ticker.upper(),
            date="N/A",
            price=0.0,
            proba=0.0,
            signal=0,
            action="NO_DATA",
            explanation="Not enough history to compute indicators.",
        )

    X_tab = scaler.transform(window[ACTIVE_FEATURES])
    proba_rf = rf_model.predict_proba(
        X_tab[-1:].astype(np.float32)
    )[:, 1][0]

    X_seq = window[ACTIVE_FEATURES].values.astype(np.float32)[np.newaxis, ...]
    proba_lstm = float(
        lstm_model.predict(X_seq, verbose=0).ravel()[0]
    )

    # Optimized ensemble weights
    proba_ens = WEIGHT_RF * proba_rf + WEIGHT_LSTM * proba_lstm
    signal = int(proba_ens >= 0.5)
    action = action_from_signal(signal)

    last_row = window.iloc[-1]
    explanation = build_explanation(last_row)

    return SignalResponse(
        ticker=req.ticker.upper(),
        date=str(last_row.name.date()),
        price=float(last_row["price"]),
        proba=float(proba_ens),
        signal=signal,
        action=action,
        explanation=explanation,
    )


@app.get("/predict")
def get_signal_get(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    GET endpoint for signal prediction (dashboard integration).
    Enhanced with confidence metrics.
    """
    window = last_lookback_window(ticker, enhanced=False)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "last_date": "N/A",
            "last_price": 0.0,
            "probability": 0.0,
            "signal": 0,
            "action": "NO_DATA",
            "explanation": "Not enough history to compute indicators.",
            "recent_mae": None,
            "confidence": "LOW"
        }

    X_tab = scaler.transform(window[ACTIVE_FEATURES])
    proba_rf = rf_model.predict_proba(
        X_tab[-1:].astype(np.float32)
    )[:, 1][0]

    X_seq = window[ACTIVE_FEATURES].values.astype(np.float32)[np.newaxis, ...]
    proba_lstm = float(
        lstm_model.predict(X_seq, verbose=0).ravel()[0]
    )

    # Optimized ensemble
    proba_ens = WEIGHT_RF * proba_rf + WEIGHT_LSTM * proba_lstm
    signal = int(proba_ens >= 0.5)
    action = action_from_signal(signal)

    last_row = window.iloc[-1]
    explanation = build_explanation(last_row)
    
    # Calculate confidence based on model agreement
    model_agreement = 1 - abs(proba_rf - proba_lstm)
    if model_agreement > 0.8 and (proba_ens > 0.65 or proba_ens < 0.35):
        confidence = "HIGH"
    elif model_agreement > 0.6:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Calculate recent MAE for validation
    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)
    
    mask = ~actual_label.isna()
    if mask.sum() > 0:
        recent_window = min(20, mask.sum())
        X_tab_all = scaler.transform(window[ACTIVE_FEATURES])
        proba_rf_all = rf_model.predict_proba(X_tab_all.astype(np.float32))[:, 1]
        
        proba_lstm_single = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
        proba_lstm_all = np.full(len(window), proba_lstm_single)
        
        proba_ens_all = WEIGHT_RF * proba_rf_all + WEIGHT_LSTM * proba_lstm_all
        
        recent_proba = proba_ens_all[mask][-recent_window:]
        recent_label = actual_label[mask].iloc[-recent_window:]
        recent_mae = float(np.abs(recent_label.values - recent_proba).mean())
    else:
        recent_mae = None

    return {
        "ticker": ticker.upper(),
        "last_date": str(last_row.name.date()),
        "last_price": float(last_row["price"]),
        "probability": float(proba_ens),
        "signal": signal,
        "action": action,
        "explanation": explanation,
        "recent_mae": recent_mae,
        "confidence": confidence,
        "model_agreement": float(model_agreement)
    }


@app.get("/history")
def get_history(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Return historical data with predictions for charting.
    """
    window = last_lookback_window(ticker, enhanced=False)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "history": [],
            "dates": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "ma10": [],
            "ma20": [],
            "proba": [],
            "proba_low": [],
            "proba_high": [],
            "error": [],
        }

    # RF probabilities for all rows
    X_tab = scaler.transform(window[ACTIVE_FEATURES])
    proba_rf_all = rf_model.predict_proba(
        X_tab.astype(np.float32)
    )[:, 1]

    # LSTM probability
    arr = window[ACTIVE_FEATURES].values.astype(np.float32)
    X_seq = arr[np.newaxis, ...]
    proba_lstm_all = lstm_model.predict(X_seq, verbose=0).ravel()
    if proba_lstm_all.shape[0] == 1:
        proba_lstm_all = np.repeat(proba_lstm_all[0], len(window))

    # Optimized ensemble
    proba_ens_all = WEIGHT_RF * proba_rf_all + WEIGHT_LSTM * proba_lstm_all[: len(window)]

    # Confidence band
    proba_low = np.clip(proba_ens_all - 0.1, 0.0, 1.0)
    proba_high = np.clip(proba_ens_all + 0.1, 0.0, 1.0)

    # Actual outcomes for error calculation
    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)
    error = (actual_label - proba_ens_all).abs()

    # Build history array
    history_data = []
    for i in range(len(window)):
        fp = future_price.iloc[i]
        history_data.append({
            "date": str(window.index[i].date()),
            "price": float(window["price"].iloc[i]),
            "probability": float(proba_ens_all[i]),
            "action": "BUY" if proba_ens_all[i] >= 0.5 else "NO_POSITION",
            "future_price": float(fp) if not np.isnan(fp) else None
        })

    return {
        "ticker": ticker.upper(),
        "history": history_data,
        "dates": [str(idx.date()) for idx in window.index],
        "open": window["Open"].round(2).tolist(),
        "high": window["High"].round(2).tolist(),
        "low": window["Low"].round(2).tolist(),
        "close": window["price"].round(2).tolist(),
        "ma10": window["ma_10"].round(2).tolist(),
        "ma20": window["ma_20"].round(2).tolist(),
        "proba": proba_ens_all.round(4).tolist(),
        "proba_low": proba_low.round(4).tolist(),
        "proba_high": proba_high.round(4).tolist(),
        "error": error.round(4).fillna(0).tolist(),
    }


@app.get("/metrics")
def get_metrics(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Performance metrics with enhanced statistics.
    """
    window = last_lookback_window(ticker, enhanced=False)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "hit_rate": None,
            "mae": None,
            "avg_ret_buy": None,
            "n_signals": 0,
            "precision": None,
            "recall": None
        }

    X_tab = scaler.transform(window[ACTIVE_FEATURES])
    proba_rf_all = rf_model.predict_proba(
        X_tab.astype(np.float32)
    )[:, 1]

    arr = window[ACTIVE_FEATURES].values.astype(np.float32)
    X_seq = arr[np.newaxis, ...]
    proba_lstm_all = lstm_model.predict(X_seq, verbose=0).ravel()
    if proba_lstm_all.shape[0] == 1:
        proba_lstm_all = np.repeat(proba_lstm_all[0], len(window))

    proba_ens_all = WEIGHT_RF * proba_rf_all + WEIGHT_LSTM * proba_lstm_all[: len(window)]

    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)

    mask = ~actual_label.isna()
    if mask.sum() == 0:
        return {
            "ticker": ticker.upper(),
            "hit_rate": None,
            "mae": None,
            "avg_ret_buy": None,
            "n_signals": 0,
            "precision": None,
            "recall": None
        }

    proba_valid = proba_ens_all[mask.values]
    label_valid = actual_label[mask]
    
    # Predictions
    pred_buy = proba_valid >= 0.5

    # Calculate metrics
    buy_mask = pred_buy
    n_buy = int(buy_mask.sum())
    
    if n_buy > 0:
        # Hit rate (accuracy on BUY signals)
        hits = (label_valid[buy_mask] == 1.0).sum()
        hit_rate = float(hits) / n_buy
        avg_ret_buy = float(actual_ret_5d[mask][buy_mask].mean())
        
        # Precision: of all BUYs predicted, how many were correct
        precision = hit_rate
    else:
        hit_rate = None
        avg_ret_buy = None
        precision = None
    
    # Recall: of all actual good buys, how many did we catch
    actual_good_buys = label_valid == 1.0
    if actual_good_buys.sum() > 0:
        recall = float((pred_buy & actual_good_buys).sum() / actual_good_buys.sum())
    else:
        recall = None

    mae = float(np.abs(label_valid.values - proba_valid).mean())

    return {
        "ticker": ticker.upper(),
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "mae": mae,
        "avg_ret_buy": avg_ret_buy,
        "n_signals": int(mask.sum()),
        "model_weights": {
            "random_forest": float(WEIGHT_RF),
            "lstm": float(WEIGHT_LSTM)
        }
    }


@app.get("/")
def root():
    """API health check and info."""
    return {
        "status": "online",
        "version": "2.0-enhanced",
        "features": {
            "total_indicators": len(ACTIVE_FEATURES),
            "ensemble_weights": {
                "random_forest": WEIGHT_RF,
                "lstm": WEIGHT_LSTM
            },
            "enhanced_mode": ACTIVE_FEATURES == ENHANCED_FEATURES
        },
        "endpoints": ["/predict", "/signal", "/history", "/metrics"]
    }
