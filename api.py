from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # NEW
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

# ---- feature helpers (same logic as notebook, simplified) ----
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


app = FastAPI(title="Stock Signal API")

# ---- CORS configuration (IMPORTANT) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # for demo/FYP; later restrict to your frontend origin
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
    return SignalResponse(
        ticker=req.ticker.upper(),
        date=str(last_row.name.date()),
        price=float(last_row["price"]),
        proba=float(proba_ens),
        signal=signal,
        action=action,
    )
