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

# =========================================================
# LOAD SAVED MODELS
# =========================================================
print("🔄 Loading improved ensemble models...")

scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("rf_model.pkl")
logreg_model = joblib.load("logreg_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")

lstm_model = tf.keras.models.load_model("lstm_model.h5")
transformer_model = tf.keras.models.load_model("transformer_model.h5")

# Load ensemble weights
ensemble_config = joblib.load("ensemble_weights.pkl")

print(f"✅ Loaded 6 models with {len(ensemble_config.get('features', []))} features")
print(f"📊 Ensemble weights: {ensemble_config}")

# =========================================================
# FEATURE CONFIGURATION
# =========================================================
# 46 enhanced features
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "vol_20d",
    "ma_10", "ma_20", "ma_50", "ma_ratio",
    "rsi", "rsi_sma",
    "macd", "macd_signal", "macd_hist",
    "bb_up", "bb_mid", "bb_low", "bb_width", "bb_pos",
    "atr_14", "atr_pct",
    "volume_z", "volume_ratio",
    "stoch_k", "stoch_d",
    "adx",
    "cci",
    "roc_10", "roc_20",
    "momentum_10",
    "williams_r",
    "obv_norm",
    "mfi",
    "trix",
    "dist_from_ma10", "dist_from_ma20", "dist_from_ma50",
    "ma_slope_10", "ma_slope_20",
    "price_to_bb_range",
    "volume_price_trend",
    "ease_of_movement",
    "keltner_upper", "keltner_lower", "keltner_width",
    "vwap", "vwap_dist",
]

LOOKBACK_DAYS = 60

# =========================================================
# TECHNICAL INDICATOR FUNCTIONS
# =========================================================

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
    low_min = df["Low"].rolling(window=window).min()
    high_max = df["High"].rolling(window=window).max()
    stoch_k = 100 * (df["price"] - low_min) / (high_max - low_min + 1e-9)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d


def compute_williams_r(df, window=14):
    """Williams %R"""
    high_max = df["High"].rolling(window=window).max()
    low_min = df["Low"].rolling(window=window).min()
    williams_r = -100 * (high_max - df["price"]) / (high_max - low_min + 1e-9)
    return williams_r


def compute_adx(df, window=14):
    """Average Directional Index"""
    high = df["High"]
    low = df["Low"]
    close = df["price"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = compute_atr(df, window=1)
    atr = tr.rolling(window).mean()

    plus_di = 100 * (plus_dm.rolling(window).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (atr + 1e-9))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(window).mean()
    return adx


def compute_cci(df, window=20):
    """Commodity Channel Index"""
    tp = (df["High"] + df["Low"] + df["price"]) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad + 1e-9)
    return cci


def compute_obv(df):
    """On Balance Volume"""
    obv = (np.sign(df["price"].diff()) * df["Volume"]).fillna(0).cumsum()
    return obv


def compute_mfi(df, window=14):
    """Money Flow Index"""
    tp = (df["High"] + df["Low"] + df["price"]) / 3
    mf = tp * df["Volume"]

    mf_pos = mf.where(tp.diff() > 0, 0).rolling(window).sum()
    mf_neg = mf.where(tp.diff() < 0, 0).rolling(window).sum()

    mfi = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-9)))
    return mfi


def compute_trix(series, window=15):
    """TRIX - Triple Exponential Average"""
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    trix = ema3.pct_change() * 100
    return trix


def compute_vwap(df):
    """Volume Weighted Average Price"""
    tp = (df["High"] + df["Low"] + df["price"]) / 3
    vwap = (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)
    return vwap


def compute_keltner(df, window=20, atr_multiplier=2):
    """Keltner Channels"""
    mid = df["price"].ewm(span=window, adjust=False).mean()
    atr = compute_atr(df, window=window)
    upper = mid + atr_multiplier * atr
    lower = mid - atr_multiplier * atr
    width = (upper - lower) / (mid + 1e-9)
    return upper, lower, width


# =========================================================
# DATA FETCHING AND FEATURE ENGINEERING
# =========================================================

def last_lookback_window(ticker: str):
    """
    Fetch stock data and compute all 46 technical indicators.
    FIXED: Handles yfinance MultiIndex columns properly.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=730)  # 2 years for indicator calculation

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None

    # ⚠️ FIX: Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Select required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols].copy()

    # ⚠️ FIX: Ensure columns are Series, not DataFrames
    for col in required_cols:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    df["price"] = df["Close"]

    # ========== BASIC FEATURES ==========
    df["ret_1d"] = df["price"].pct_change()
    df["ret_5d"] = df["price"].pct_change(5)
    df["ret_20d"] = df["price"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # ========== MOVING AVERAGES ==========
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["ma_50"] = df["price"].rolling(50).mean()
    df["ma_ratio"] = df["ma_10"] / (df["ma_20"] + 1e-9)

    # ========== RSI ==========
    df["rsi"] = compute_rsi(df["price"], window=14)
    df["rsi_sma"] = df["rsi"].rolling(14).mean()

    # ========== MACD ==========
    macd, macd_sig, macd_hist = compute_macd(df["price"])
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    # ========== BOLLINGER BANDS ==========
    bb_up, bb_mid, bb_low, bb_width = compute_bollinger(df["price"])
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low
    df["bb_width"] = bb_width
    df["bb_pos"] = (df["price"] - bb_low) / (bb_up - bb_low + 1e-9)

    # ========== ATR ==========
    df["atr_14"] = compute_atr(df, window=14)
    df["atr_pct"] = df["atr_14"] / (df["price"] + 1e-9) * 100

    # ========== VOLUME ==========
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (
        df["Volume"].rolling(20).std() + 1e-9
    )
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ma_20"] + 1e-9)

    # ========== STOCHASTIC ==========
    stoch_k, stoch_d = compute_stochastic(df, window=14)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # ========== ADX ==========
    df["adx"] = compute_adx(df, window=14)

    # ========== CCI ==========
    df["cci"] = compute_cci(df, window=20)

    # ========== RATE OF CHANGE ==========
    df["roc_10"] = ((df["price"] - df["price"].shift(10)) / (df["price"].shift(10) + 1e-9)) * 100
    df["roc_20"] = ((df["price"] - df["price"].shift(20)) / (df["price"].shift(20) + 1e-9)) * 100

    # ========== MOMENTUM ==========
    df["momentum_10"] = df["price"] - df["price"].shift(10)

    # ========== WILLIAMS %R ==========
    df["williams_r"] = compute_williams_r(df, window=14)

    # ========== ON BALANCE VOLUME ==========
    df["obv"] = compute_obv(df)
    df["obv_norm"] = (df["obv"] - df["obv"].rolling(20).mean()) / (df["obv"].rolling(20).std() + 1e-9)

    # ========== MONEY FLOW INDEX ==========
    df["mfi"] = compute_mfi(df, window=14)

    # ========== TRIX ==========
    df["trix"] = compute_trix(df["price"], window=15)

    # ========== DISTANCE FROM MA ==========
    df["dist_from_ma10"] = (df["price"] - df["ma_10"]) / (df["ma_10"] + 1e-9)
    df["dist_from_ma20"] = (df["price"] - df["ma_20"]) / (df["ma_20"] + 1e-9)
    df["dist_from_ma50"] = (df["price"] - df["ma_50"]) / (df["ma_50"] + 1e-9)

    # ========== MA SLOPE ==========
    df["ma_slope_10"] = df["ma_10"].diff(5) / (df["ma_10"] + 1e-9)
    df["ma_slope_20"] = df["ma_20"].diff(5) / (df["ma_20"] + 1e-9)

    # ========== ADVANCED ==========
    df["price_to_bb_range"] = (df["price"] - df["bb_low"]) / (df["bb_up"] - df["bb_low"] + 1e-9)
    df["volume_price_trend"] = df["Volume"] * df["ret_1d"]

    # Ease of Movement
    distance_moved = ((df["High"] + df["Low"]) / 2) - ((df["High"].shift(1) + df["Low"].shift(1)) / 2)
    box_ratio = (df["Volume"] / 1e6) / (df["High"] - df["Low"] + 1e-9)
    df["ease_of_movement"] = distance_moved / (box_ratio + 1e-9)

    # ========== KELTNER CHANNELS ==========
    k_upper, k_lower, k_width = compute_keltner(df, window=20, atr_multiplier=2)
    df["keltner_upper"] = k_upper
    df["keltner_lower"] = k_lower
    df["keltner_width"] = k_width

    # ========== VWAP ==========
    df["vwap"] = compute_vwap(df)
    df["vwap_dist"] = (df["price"] - df["vwap"]) / (df["vwap"] + 1e-9)

    # Drop NaN rows
    df = df.dropna()

    if len(df) < LOOKBACK_DAYS:
        return None

    return df.iloc[-LOOKBACK_DAYS:]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def action_from_signal(sig: int) -> str:
    """Convert signal to action"""
    return "BUY" if sig == 1 else "NO_POSITION"


def build_explanation(row: pd.Series) -> str:
    """Generate human-readable explanation"""
    msgs = []

    # RSI
    rsi = float(row.get("rsi", np.nan))
    if not np.isnan(rsi):
        if rsi > 70:
            msgs.append(f"RSI is {rsi:.1f} (overbought - caution).")
        elif rsi < 30:
            msgs.append(f"RSI is {rsi:.1f} (oversold - potential buy).")
        else:
            msgs.append(f"RSI is {rsi:.1f} (neutral).")

    # Moving Average Trend
    ma10 = float(row.get("ma_10", np.nan))
    ma20 = float(row.get("ma_20", np.nan))
    price = float(row.get("price", np.nan))

    if not np.isnan(ma10) and not np.isnan(ma20):
        if ma10 > ma20 and price > ma10:
            msgs.append("Strong uptrend (price > MA10 > MA20).")
        elif ma10 < ma20 and price < ma10:
            msgs.append("Strong downtrend (price < MA10 < MA20).")

    # Volatility
    vol20 = float(row.get("vol_20d", np.nan))
    if not np.isnan(vol20):
        if vol20 > 0.03:
            msgs.append("High volatility - higher risk.")
        elif vol20 < 0.015:
            msgs.append("Low volatility - stable.")

    # MACD
    macd_val = float(row.get("macd", np.nan))
    macd_sig = float(row.get("macd_signal", np.nan))
    if not np.isnan(macd_val) and not np.isnan(macd_sig):
        if macd_val > macd_sig:
            msgs.append("MACD bullish crossover.")
        else:
            msgs.append("MACD bearish signal.")

    if not msgs:
        return "Mixed signals. Use additional analysis."

    return " ".join(msgs)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="Stock Signal API - 6-Model Ensemble", version="3.0")

# ⚠️ CORS MIDDLEWARE - MUST BE HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST/RESPONSE MODELS
# =========================================================

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


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "online",
        "version": "3.0-improved",
        "models": ["LSTM", "Transformer", "RF", "LogReg", "XGB", "LGB"],
        "features": len(FEATURES),
        "ensemble_weights": ensemble_config,
        "endpoints": ["/predict", "/signal", "/history", "/metrics"]
    }


@app.get("/predict")
def get_signal_get(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """
    GET endpoint for predictions (used by dashboard).
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
            "explanation": "Insufficient data.",
            "confidence": "LOW"
        }

    # Prepare features
    X_tab = scaler.transform(window[FEATURES])
    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]

    # Get predictions from all 6 models
    proba_lstm = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
    proba_transformer = float(transformer_model.predict(X_seq, verbose=0).ravel()[0])
    proba_rf = rf_model.predict_proba(X_tab[-1:].astype(np.float32))[:, 1][0]
    proba_logreg = logreg_model.predict_proba(X_tab[-1:].astype(np.float32))[:, 1][0]
    proba_xgb = xgb_model.predict_proba(X_tab[-1:].astype(np.float32))[:, 1][0]
    proba_lgb = lgb_model.predict_proba(X_tab[-1:].astype(np.float32))[:, 1][0]

    # Weighted ensemble
    proba_ens = (
        ensemble_config["lstm_weight"] * proba_lstm +
        ensemble_config["transformer_weight"] * proba_transformer +
        ensemble_config["rf_weight"] * proba_rf +
        ensemble_config["logreg_weight"] * proba_logreg +
        ensemble_config["xgb_weight"] * proba_xgb +
        ensemble_config["lgb_weight"] * proba_lgb
    )

    signal = int(proba_ens >= 0.5)
    action = action_from_signal(signal)

    last_row = window.iloc[-1]
    explanation = build_explanation(last_row)

    # Confidence
    model_std = np.std([proba_lstm, proba_transformer, proba_rf, proba_logreg, proba_xgb, proba_lgb])
    if model_std < 0.1 and (proba_ens > 0.6 or proba_ens < 0.4):
        confidence = "HIGH"
    elif model_std < 0.2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "ticker": ticker.upper(),
        "last_date": str(last_row.name.date()),
        "last_price": float(last_row["price"]),
        "probability": float(proba_ens),
        "signal": signal,
        "action": action,
        "explanation": explanation,
        "confidence": confidence,
        "model_predictions": {
            "lstm": float(proba_lstm),
            "transformer": float(proba_transformer),
            "random_forest": float(proba_rf),
            "logistic_regression": float(proba_logreg),
            "xgboost": float(proba_xgb),
            "lightgbm": float(proba_lgb)
        }
    }


@app.post("/signal", response_model=SignalResponse)
def get_signal_post(req: SignalRequest):
    """POST endpoint (original format)"""
    window = last_lookback_window(req.ticker)
    if window is None:
        return SignalResponse(
            ticker=req.ticker.upper(),
            date="N/A",
            price=0.0,
            proba=0.0,
            signal=0,
            action="NO_DATA",
            explanation="Insufficient data.",
        )

    X_tab = scaler.transform(window[FEATURES])
    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]

    proba_lstm = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
    proba_transformer = float(transformer_model.predict(X_seq, verbose=0).ravel()[0])
    proba_rf = rf_model.predict_proba(X_tab[-1:])[:, 1][0]
    proba_logreg = logreg_model.predict_proba(X_tab[-1:])[:, 1][0]
    proba_xgb = xgb_model.predict_proba(X_tab[-1:])[:, 1][0]
    proba_lgb = lgb_model.predict_proba(X_tab[-1:])[:, 1][0]

    proba_ens = (
        ensemble_config["lstm_weight"] * proba_lstm +
        ensemble_config["transformer_weight"] * proba_transformer +
        ensemble_config["rf_weight"] * proba_rf +
        ensemble_config["logreg_weight"] * proba_logreg +
        ensemble_config["xgb_weight"] * proba_xgb +
        ensemble_config["lgb_weight"] * proba_lgb
    )

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


@app.get("/history")
def get_history(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Historical data with predictions"""
    window = last_lookback_window(ticker)
    if window is None:
        return {"ticker": ticker.upper(), "history": [], "error": "Insufficient data"}

    X_tab = scaler.transform(window[FEATURES])
    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]

    # Get predictions for all historical points
    proba_rf_all = rf_model.predict_proba(X_tab.astype(np.float32))[:, 1]
    proba_lstm_all = lstm_model.predict(X_seq, verbose=0).ravel()
    proba_transformer_all = transformer_model.predict(X_seq, verbose=0).ravel()

    # For simplicity, use last prediction for all (or you can batch predict)
    if len(proba_lstm_all) == 1:
        proba_lstm_all = np.repeat(proba_lstm_all[0], len(window))
    if len(proba_transformer_all) == 1:
        proba_transformer_all = np.repeat(proba_transformer_all[0], len(window))

    proba_logreg_all = logreg_model.predict_proba(X_tab)[:, 1]
    proba_xgb_all = xgb_model.predict_proba(X_tab)[:, 1]
    proba_lgb_all = lgb_model.predict_proba(X_tab)[:, 1]

    proba_ens_all = (
        ensemble_config["lstm_weight"] * proba_lstm_all[:len(window)] +
        ensemble_config["transformer_weight"] * proba_transformer_all[:len(window)] +
        ensemble_config["rf_weight"] * proba_rf_all +
        ensemble_config["logreg_weight"] * proba_logreg_all +
        ensemble_config["xgb_weight"] * proba_xgb_all +
        ensemble_config["lgb_weight"] * proba_lgb_all
    )

    future_price = window["price"].shift(-5)

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
        "prices": window["price"].round(2).tolist(),
        "probabilities": proba_ens_all.round(4).tolist(),
    }


@app.get("/metrics")
def get_metrics(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    """Performance metrics"""
    window = last_lookback_window(ticker)
    if window is None:
        return {"ticker": ticker.upper(), "error": "Insufficient data"}

    X_tab = scaler.transform(window[FEATURES])
    X_seq = window[FEATURES].values.astype(np.float32)[np.newaxis, ...]

    proba_rf_all = rf_model.predict_proba(X_tab)[:, 1]
    proba_lstm = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
    proba_transformer = float(transformer_model.predict(X_seq, verbose=0).ravel()[0])
    proba_logreg_all = logreg_model.predict_proba(X_tab)[:, 1]
    proba_xgb_all = xgb_model.predict_proba(X_tab)[:, 1]
    proba_lgb_all = lgb_model.predict_proba(X_tab)[:, 1]

    proba_lstm_all = np.full(len(window), proba_lstm)
    proba_transformer_all = np.full(len(window), proba_transformer)

    proba_ens_all = (
        ensemble_config["lstm_weight"] * proba_lstm_all +
        ensemble_config["transformer_weight"] * proba_transformer_all +
        ensemble_config["rf_weight"] * proba_rf_all +
        ensemble_config["logreg_weight"] * proba_logreg_all +
        ensemble_config["xgb_weight"] * proba_xgb_all +
        ensemble_config["lgb_weight"] * proba_lgb_all
    )

    future_price = window["price"].shift(-5)
    actual_ret_5d = (future_price / window["price"] - 1.0) * 100.0
    actual_label = (actual_ret_5d > 0).astype(float)

    mask = ~actual_label.isna()
    if mask.sum() == 0:
        return {"ticker": ticker.upper(), "error": "No future data"}

    proba_valid = proba_ens_all[mask.values]
    label_valid = actual_label[mask]

    pred_buy = proba_valid >= 0.5
    n_buy = int(pred_buy.sum())

    if n_buy > 0:
        hits = (label_valid[pred_buy] == 1.0).sum()
        hit_rate = float(hits) / n_buy
        avg_ret_buy = float(actual_ret_5d[mask][pred_buy].mean())
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
        "ensemble_weights": ensemble_config
    }
