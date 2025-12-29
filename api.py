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
LOOKBACK_DAYS = 60


# ---- feature helpers ----
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger(series, window=20, num_std=2):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-9)
    return upper, mid, lower, width


def compute_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["price"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


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


app = FastAPI(title="Stock Signal API")

# ---- CORS configuration ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo/FYP; later restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def last_lookback_window(ticker: str):
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["price"] = df["Close"]

    df["ret_1d"] = df["price"].pct_change()
    df["ret_5d"] = df["price"].pct_change(5)
    df["ret_20d"] = df["price"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["ma_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    df["rsi"] = compute_rsi(df["price"], window=14)

    macd, macd_sig, macd_hist = compute_macd(df["price"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    bb_up, bb_mid, bb_low, bb_width = compute_bollinger(df["price"])
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low
    df["bb_width"] = bb_width

    df["atr_14"] = compute_atr(df, window=14)
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std() + 1e-9
    )

    df = df.dropna()
    if len(df) < LOOKBACK_DAYS:
        return None

    return df.iloc[-LOOKBACK_DAYS:]


def action_from_signal(sig: int) -> str:
    return "BUY" if sig == 1 else "NO_POSITION"


def build_explanation(row: pd.Series) -> str:
    """
    Generate a short human-readable explanation based on
    the latest feature values.
    """
    msgs = []

    rsi = float(row.get("rsi", np.nan))
    if not np.isnan(rsi):
        if rsi > 70:
            msgs.append(f"RSI is {rsi:.1f}, which is in overbought territory.")
        elif rsi < 30:
            msgs.append(f"RSI is {rsi:.1f}, which is in oversold territory.")
        else:
            msgs.append(f"RSI is {rsi:.1f}, a neutral momentum zone.")

    ma10 = float(row.get("ma_10", np.nan))
    ma20 = float(row.get("ma_20", np.nan))
    if not np.isnan(ma10) and not np.isnan(ma20):
        if ma10 > ma20:
            msgs.append("Short-term trend is above the 20-day average, indicating upward momentum.")
        elif ma10 < ma20:
            msgs.append("Short-term trend is below the 20-day average, indicating downward momentum.")

    vol20 = float(row.get("vol_20d", np.nan))
    if not np.isnan(vol20):
        if vol20 > 0.03:
            msgs.append("Recent volatility is relatively high, so risk is elevated.")
        elif vol20 < 0.015:
            msgs.append("Recent volatility is relatively low, moves may be smaller.")

    macd_val = float(row.get("macd", np.nan))
    macd_sig = float(row.get("macd_signal", np.nan))
    if not np.isnan(macd_val) and not np.isnan(macd_sig):
        if macd_val > macd_sig:
            msgs.append("MACD is above its signal line, a bullish momentum sign.")
        elif macd_val < macd_sig:
            msgs.append("MACD is below its signal line, a bearish momentum sign.")

    if not msgs:
        return "Technical indicators are in a mixed or neutral zone."

    return " ".join(msgs)


@app.post("/signal", response_model=SignalResponse)
def get_signal(req: SignalRequest):
    window = last_lookback_window(req.ticker)
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

    X_tab = scaler.transform(window[FEATURES])
    proba_rf = rf_model.predict_proba(
        X_tab[-1:].astype(np.float32)
    )[:, 1][0]

    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]
    proba_lstm = float(
        lstm_model.predict(X_seq, verbose=0).ravel()[0]
    )

    proba_ens = 0.5 * proba_rf + 0.5 * proba_lstm
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


# ---- NEW: GET /predict endpoint for dashboard ----
@app.get("/predict")
def get_signal_get(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    GET version of /signal for easier dashboard integration.
    Returns prediction for a given ticker.
    """
    window = last_lookback_window(ticker)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "last_date": "N/A",
            "last_price": 0.0,
            "probability": 0.0,
            "signal": 0,
            "action": "NO_DATA",
            "explanation": "Not enough history to compute indicators.",
            "recent_mae": None
        }

    X_tab = scaler.transform(window[FEATURES])
    proba_rf = rf_model.predict_proba(
        X_tab[-1:].astype(np.float32)
    )[:, 1][0]

    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]
    proba_lstm = float(
        lstm_model.predict(X_seq, verbose=0).ravel()[0]
    )

    proba_ens = 0.5 * proba_rf + 0.5 * proba_lstm
    signal = int(proba_ens >= 0.5)
    action = action_from_signal(signal)

    last_row = window.iloc[-1]
    explanation = build_explanation(last_row)
    
    # Calculate recent MAE for confidence
    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)
    
    mask = ~actual_label.isna()
    if mask.sum() > 0:
        recent_window = min(20, mask.sum())
        X_tab_all = scaler.transform(window[FEATURES])
        proba_rf_all = rf_model.predict_proba(X_tab_all.astype(np.float32))[:, 1]
        
        # For LSTM, use the same sequence approach
        proba_lstm_single = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
        proba_lstm_all = np.full(len(window), proba_lstm_single)
        
        proba_ens_all = 0.5 * proba_rf_all + 0.5 * proba_lstm_all
        
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
        "recent_mae": recent_mae
    }


# ---- history endpoint for charts ----
@app.get("/history")
def get_history(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Return the last LOOKBACK_DAYS of data for charting:
    OHLC prices, MA10, MA20, model probability and
    a simple confidence band and error.
    """
    window = last_lookback_window(ticker)
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
    X_tab = scaler.transform(window[FEATURES])
    proba_rf_all = rf_model.predict_proba(
        X_tab.astype(np.float32)
    )[:, 1]

    # LSTM probability: use same sequence for all rows (last LOOKBACK_DAYS)
    arr = window[FEATURES].values.astype(np.float32)
    X_seq = arr[np.newaxis, ...]
    proba_lstm_all = lstm_model.predict(X_seq, verbose=0).ravel()
    if proba_lstm_all.shape[0] == 1:
        proba_lstm_all = np.repeat(proba_lstm_all[0], len(window))

    proba_ens_all = 0.5 * proba_rf_all + 0.5 * proba_lstm_all[: len(window)]

    # Simple confidence band: +/- 0.1 around ensemble, clipped to [0,1]
    proba_low = np.clip(proba_ens_all - 0.1, 0.0, 1.0)
    proba_high = np.clip(proba_ens_all + 0.1, 0.0, 1.0)

    # Actual 5‑day forward label for rough error metric
    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)
    error = (actual_label - proba_ens_all).abs()

    # Build history array for dashboard
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


# ---- metrics endpoint ----
@app.get("/metrics")
def get_metrics(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    Simple performance metrics for the last LOOKBACK_DAYS window:
    hit-rate, mean absolute error, and average 5-day return after BUY.
    """
    window = last_lookback_window(ticker)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "hit_rate": None,
            "mae": None,
            "avg_ret_buy": None,
            "n_signals": 0,
        }

    X_tab = scaler.transform(window[FEATURES])
    proba_rf_all = rf_model.predict_proba(
        X_tab.astype(np.float32)
    )[:, 1]

    arr = window[FEATURES].values.astype(np.float32)
    X_seq = arr[np.newaxis, ...]
    proba_lstm_all = lstm_model.predict(X_seq, verbose=0).ravel()
    if proba_lstm_all.shape[0] == 1:
        proba_lstm_all = np.repeat(proba_lstm_all[0], len(window))

    proba_ens_all = 0.5 * proba_rf_all + 0.5 * proba_lstm_all[: len(window)]

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
        }

    proba_valid = proba_ens_all[mask.values]
    label_valid = actual_label[mask]

    buy_mask = proba_valid >= 0.5
    n_buy = int(buy_mask.sum())
    if n_buy > 0:
        hits = (label_valid[buy_mask] == 1.0).sum()
        hit_rate = float(hits) / n_buy
        avg_ret_buy = float(actual_ret_5d[mask][buy_mask].mean())
    else:
        hit_rate = None
        avg_ret_buy = None

    mae = float(np.abs(label_valid.values - proba_valid).mean())

    return {
        "ticker": ticker.upper(),
        "hit_rate": hit_rate,
        "mae": mae,
        "avg_ret_buy": avg_ret_buy,
        "n_signals": int(mask.sum()),
    }
