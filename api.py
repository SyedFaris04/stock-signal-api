# api.py – Honest 6‑model ensemble API for your website

from typing import Any, Dict, List
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================
# 1. CUSTOM LAYERS (NEEDED TO LOAD SAVED MODELS)
# =========================================================

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        import tensorflow.keras.backend as K

        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def get_config(self):
        base_config = super().get_config()
        return base_config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return base_config

# =========================================================
# 2. LOAD MODELS, SCALERS, METADATA
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(name: str):
    return joblib.load(os.path.join(BASE_DIR, name))

scaler = load_pickle("scaler.pkl")
scaler_lstm = load_pickle("scaler_lstm.pkl")

rf_model = load_pickle("rf_model.pkl")
logreg_model = load_pickle("logreg_model.pkl")
xgb_model = load_pickle("xgb_model.pkl")
lgb_model = load_pickle("lgb_model.pkl")

lstm_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "lstm_attention_model.keras"),
    compile=False,
    custom_objects={"AttentionLayer": AttentionLayer},
)

transformer_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "transformer_model.keras"),
    compile=False,
    custom_objects={"PositionalEncoding": PositionalEncoding},
)

FEATURES: List[str] = load_pickle("features.pkl")
ensemble_weights: Dict[str, float] = load_pickle("ensemble_weights.pkl")
metadata: Dict[str, Any] = load_pickle("model_metadata.pkl")

LOOKBACK_DAYS: int = metadata.get("lookback_days", 60)
HORIZON: int = metadata.get("horizon", 5)

W_LSTM = ensemble_weights["lstm_weight"]
W_TRANS = ensemble_weights["transformer_weight"]
W_RF = ensemble_weights["rf_weight"]
W_LOG = ensemble_weights["logreg_weight"]
W_XGB = ensemble_weights["xgb_weight"]
W_LGB = ensemble_weights["lgb_weight"]

ACTIVE_FEATURES = FEATURES  # keep old name for compatibility

# =========================================================
# 3. FEATURE ENGINEERING (MATCHES TRAINING)
# =========================================================

def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-9)
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
    low  = df["Low"]
    close= df["price"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df['momentum_3d'] = df['price'].pct_change(3)
    df['momentum_10d'] = df['price'].pct_change(10)
    df['momentum_30d'] = df['price'].pct_change(30)
    df['accel_5d'] = df['price'].pct_change(5) - df['price'].pct_change(5).shift(5)

    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_ratio'] = df['vol_5d'] / (df['vol_20d'] + 1e-9)

    df['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['hl_ma'] = df['hl_spread'].rolling(10).mean()

    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_20'] + 1e-9)
    df['volume_price_trend'] = df['Volume'] * df['ret_1d']
    df['volume_surge'] = (df['Volume'] > df['volume_ma_20'] * 2).astype(int)

    df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    df['gap_ma'] = df['gap'].rolling(5).mean()

    df['true_range'] = df[['High', 'Low']].apply(lambda x: x['High'] - x['Low'], axis=1)
    df['range_ma'] = df['true_range'].rolling(10).mean()

    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14 + 1e-9)

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-9)

    raw_money_flow = typical_price * df['Volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(14).sum()
    negative_mf = negative_flow.rolling(14).sum()
    mfi_ratio = positive_mf / (negative_mf + 1e-9)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))

    df['ma_50'] = df['price'].rolling(50).mean()
    df['price_distance_ma20'] = (df['price'] - df['ma_20']) / (df['ma_20'] + 1e-9)
    df['price_distance_ma50'] = (df['price'] - df['ma_50']) / (df['ma_50'] + 1e-9)
    df['price_trend_strength'] = df['ret_1d'].rolling(10).mean() / (df['vol_5d'] + 1e-9)

    df['volatility_rank'] = df['vol_20d'].rolling(252).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5
    )

    df['returns_skew'] = df['ret_1d'].rolling(20).skew()
    df['returns_kurt'] = df['ret_1d'].rolling(20).kurt()

    return df

def last_lookback_window(ticker: str) -> pd.DataFrame | None:
    end = datetime.utcnow()
    start = end - timedelta(days=365 * 3)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["price"] = df["Close"]

    df["ret_1d"]  = df["price"].pct_change()
    df["ret_5d"]  = df["price"].pct_change(5)
    df["ret_20d"] = df["price"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    df["ma_10"]   = df["price"].rolling(10).mean()
    df["ma_20"]   = df["price"].rolling(20).mean()
    df["ma_ratio"]= df["ma_10"] / (df["ma_20"] + 1e-9)

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

    df = compute_advanced_features(df)

    df = df.dropna()
    if len(df) < LOOKBACK_DAYS:
        return None

    return df.iloc[-LOOKBACK_DAYS:]

# =========================================================
# 4. SIGNAL / EXPLANATION
# =========================================================

def action_from_proba(p: float) -> str:
    if p >= 0.55:
        return "BUY"
    elif p <= 0.45:
        return "NO_POSITION"
    else:
        return "HOLD"

def build_explanation(row: pd.Series) -> str:
    msgs = []
    price = float(row.get("price", np.nan))

    rsi = float(row.get("rsi", np.nan))
    if not np.isnan(rsi):
        if rsi > 70:
            msgs.append(f"RSI {rsi:.1f} suggests overbought conditions.")
        elif rsi < 30:
            msgs.append(f"RSI {rsi:.1f} suggests oversold conditions.")
        else:
            msgs.append(f"RSI {rsi:.1f} shows neutral momentum.")

    ma10 = float(row.get("ma_10", np.nan))
    ma20 = float(row.get("ma_20", np.nan))
    if not np.isnan(ma10) and not np.isnan(ma20) and not np.isnan(price):
        if ma10 > ma20 and price > ma10:
            msgs.append("Price is above both 10‑day and 20‑day moving averages (bullish).")
        elif ma10 < ma20 and price < ma10:
            msgs.append("Price is below both moving averages (bearish).")

    vol20 = float(row.get("vol_20d", np.nan))
    if not np.isnan(vol20):
        if vol20 > 0.03:
            msgs.append("Volatility is high; expect larger price swings.")
        elif vol20 < 0.015:
            msgs.append("Volatility is low; price moves are more stable.")

    macd_val = float(row.get("macd", np.nan))
    macd_sig = float(row.get("macd_signal", np.nan))
    if not np.isnan(macd_val) and not np.isnan(macd_sig):
        if macd_val > macd_sig:
            msgs.append("MACD is above its signal line (bullish momentum).")
        elif macd_val < macd_sig:
            msgs.append("MACD is below its signal line (bearish momentum).")

    bb_up = float(row.get("bb_up", np.nan))
    bb_low = float(row.get("bb_low", np.nan))
    if not np.isnan(bb_up) and not np.isnan(bb_low) and not np.isnan(price):
        if price > bb_up:
            msgs.append("Price is above the upper Bollinger Band (overbought risk).")
        elif price < bb_low:
            msgs.append("Price is below the lower Bollinger Band (oversold opportunity).")

    if not msgs:
        return "Indicators are mixed; consider additional factors before trading."
    return " ".join(msgs)

# =========================================================
# 5. ENSEMBLE PREDICTION
# =========================================================

def ensemble_proba_for_window(window: pd.DataFrame) -> Dict[str, Any]:
    X_tab_all = scaler.transform(window[ACTIVE_FEATURES])
    x_last = X_tab_all[-1:].astype(np.float32)

    p_rf  = float(rf_model.predict_proba(x_last)[:, 1][0])
    p_log = float(logreg_model.predict_proba(x_last)[:, 1][0])
    p_xgb = float(xgb_model.predict_proba(x_last)[:, 1][0])
    p_lgb = float(lgb_model.predict_proba(x_last)[:, 1][0])

    seq_raw = window[ACTIVE_FEATURES].values.astype(np.float32)
    seq_scaled = scaler_lstm.transform(seq_raw)
    X_seq = seq_scaled[np.newaxis, ...]

    p_lstm = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
    p_trans = float(transformer_model.predict(X_seq, verbose=0).ravel()[0])

    proba_ens = (
        W_LSTM * p_lstm +
        W_TRANS * p_trans +
        W_RF * p_rf +
        W_LOG * p_log +
        W_XGB * p_xgb +
        W_LGB * p_lgb
    )

    return {
        "ensemble": proba_ens,
        "lstm": p_lstm,
        "transformer": p_trans,
        "rf": p_rf,
        "logreg": p_log,
        "xgb": p_xgb,
        "lgb": p_lgb,
    }

# =========================================================
# 6. FASTAPI APP + ENDPOINTS
# =========================================================

class SignalRequest(BaseModel):
    ticker: str

class SignalResponse(BaseModel):
    ticker: str
    date: str
    price: float
    proba: float
    action: str
    explanation: str

app = FastAPI(title="Honest Stock Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/signal", response_model=SignalResponse)
def post_signal(req: SignalRequest):
    window = last_lookback_window(req.ticker)
    if window is None:
        return SignalResponse(
            ticker=req.ticker.upper(),
            date="N/A",
            price=0.0,
            proba=0.0,
            action="NO_DATA",
            explanation="Not enough historical data to compute indicators.",
        )

    probs = ensemble_proba_for_window(window)
    proba = float(probs["ensemble"])
    last_row = window.iloc[-1]

    return SignalResponse(
        ticker=req.ticker.upper(),
        date=str(last_row.name.date()),
        price=float(last_row["price"]),
        proba=proba,
        action=action_from_proba(proba),
        explanation=build_explanation(last_row),
    )

@app.get("/predict")
def get_predict(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    window = last_lookback_window(ticker)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "last_date": "N/A",
            "last_price": 0.0,
            "probability": 0.0,
            "action": "NO_DATA",
            "explanation": "Not enough historical data.",
        }

    probs = ensemble_proba_for_window(window)
    proba = float(probs["ensemble"])
    last_row = window.iloc[-1]

    model_agreement = 1 - (
        abs(probs["lstm"] - probs["rf"]) +
        abs(probs["transformer"] - probs["rf"])
    ) / 2
    if model_agreement > 0.8 and (proba > 0.65 or proba < 0.35):
        confidence = "HIGH"
    elif model_agreement > 0.6:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "ticker": ticker.upper(),
        "last_date": str(last_row.name.date()),
        "last_price": float(last_row["price"]),
        "probability": proba,
        "action": action_from_proba(proba),
        "explanation": build_explanation(last_row),
        "components": {k: float(v) for k, v in probs.items()},
        "confidence": confidence,
        "model_agreement": float(model_agreement),
    }

@app.get("/history")
def get_history(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    window = last_lookback_window(ticker)
    if window is None:
        return {
            "ticker": ticker.upper(),
            "history": [],
            "dates": [],
            "price": [],
            "proba": [],
        }

    probs = ensemble_proba_for_window(window)
    proba = float(probs["ensemble"])

    history = []
    for idx, row in window.iterrows():
        history.append({
            "date": str(idx.date()),
            "price": float(row["price"]),
            "probability": proba,
        })

    return {
        "ticker": ticker.upper(),
        "history": history,
        "dates": [str(i.date()) for i in window.index],
        "price": window["price"].round(2).tolist(),
        "proba": [proba] * len(window),
    }

@app.get("/metrics")
def get_metrics(ticker: str = Query(..., min_length=1)) -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "test_auc": metadata.get("test_auc_ensemble"),
        "val_auc": metadata.get("val_auc"),
        "cagr": metadata.get("cagr"),
        "sharpe": metadata.get("sharpe"),
        "model_weights": ensemble_weights,
    }

@app.get("/")
def root():
    return {
        "status": "online",
        "version": "honest-ensemble-v1",
        "lookback_days": LOOKBACK_DAYS,
        "horizon_days": HORIZON,
        "models": [
            "LSTM-Attention", "Transformer",
            "RandomForest", "LogisticRegression",
            "XGBoost", "LightGBM",
        ],
        "endpoints": ["/predict", "/signal", "/history", "/metrics"],
    }
