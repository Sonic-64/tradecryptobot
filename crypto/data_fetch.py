import argparse
import json
from datetime import datetime,timezone,date,timedelta
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from binance.client import Client
import requests
from sklearn.preprocessing import StandardScaler
import joblib

INTRADAY_TF   = "1h"       # resolution to check intraday movement
LOOKBACK_DAYS = 180        # how far back to analyze
DIP_THRESHOLD = 0.0        # price must drop this % below open to count as a dip

# Try to import config for data directories (optional)

client = Client()
def get_tradable_futures_symbols():
    info = client.futures_exchange_info()
    symbols = []

    for s in info["symbols"]:
        if (
            s["status"] == "TRADING"
            and s["contractType"] == "PERPETUAL"
            and s["quoteAsset"] == "USDT"
        ):
            try:
                klines = client.futures_klines(
                    symbol=s["symbol"],
                    interval=Client.KLINE_INTERVAL_1HOUR,
                    limit=1
                )
                if klines:
                    symbols.append(s["symbol"])
            except:
                pass

    return symbols
def get_price_at(symbol, dt):
    """Get price at a specific datetime, minute accurate"""
    ts = int(dt.timestamp() * 1000)
    klines = client.get_historical_klines(
        symbol,
        Client.KLINE_INTERVAL_1MINUTE,
        start_str=ts,
        limit=1
    )
    if not klines:
        return None
    return float(klines[0][4])
def fetch_all_funding_rates(symbol="BTCUSDT", start_time=None, end_time=None):
    all_data = []
    current_start = start_time

    while True:
        data = client.futures_funding_rate(
            symbol=symbol,
            startTime=current_start,
            endTime=end_time,
            limit=1000
        )

        if not data:
            break

        all_data.extend(data)

        last_time = data[-1]["fundingTime"]
        current_start = last_time + 1  # move forward

        if len(data) < 1000:
            break

    df = pd.DataFrame(all_data)

    if df.empty:
        return df

    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)

    return df[["fundingTime", "fundingRate"]]
def align_funding(funding_df, price_df):
    funding_df = funding_df.set_index("fundingTime").sort_index()

    # Resample funding to 3H to match your candles
    funding_aligned = funding_df.resample("1h").ffill()

    # Align exactly to your price dataframe index
    funding_aligned = funding_aligned.reindex(price_df.index, method="ffill")

    return funding_aligned

def download_data(symbol: str, months: int, interval: str = "1h",cutoff:int=0):




    # Be explicit to avoid surprises from future defaults
    if months == -1:
        # Full history: Start from Binance's earliest BTCUSDT data
        start_str = "1 Aug 2017"  # ~1502928000000 ms
    else:
        end_time = datetime.now() - timedelta(days=(int(cutoff)))
        start_time = end_time - timedelta(days=int(round(months * 30.44)))  # ~30.44 days/month
        start_str = start_time.strftime("%d %b %Y")
        end_str = end_time.strftime("%d %b %Y")

        # Fetch with auto-pagination (handles limits)
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_str=start_str,
        end_str=end_str
    )
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades',
                          'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']

    df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                         'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume',
                                         'Taker Buy Quote Asset Volume', 'Ignore'])

    for col in columns_to_convert:
        df[col] = df[col].astype(float)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms', utc=True)

    df = df.set_index('Open Time').sort_index()
    # Ensure DatetimeIndex (sometimes it’s plain Index)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    # Strip timezone if present (resample expects naive or consistent tz)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    funding_df = fetch_all_funding_rates(
        symbol=symbol,
        start_time=int(df.index[0].timestamp() * 1000),
        end_time=int(df.index[-1].timestamp() * 1000)
    )

    # Align to your candles
    funding_aligned = align_funding(funding_df, df)

    # Merge
    df["funding_rate"] = funding_aligned["fundingRate"]
    df = df.sort_index().dropna(how="any")
    df = df.drop(columns=['Ignore','Close Time'])

    return df
def rolling_slope(series, window=20):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(0.0)
        else:
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            coef = np.polyfit(x, y, 1)
            slopes.append(coef[0] / series.iloc[i])  # normalize by price
    return pd.Series(slopes, index=series.index)

def compute_features(df: pd.DataFrame, resample_hours: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resample data and compute technical indicators once on the full dataset.
    
    Args:
        df: Raw OHLCV DataFrame with hourly data
        resample_hours: Resampling interval in hours
    
    Returns:
        Tuple of (feature DataFrame, feature column names)
    """
    # Resample to specified interval
    df_resampled = df.resample(f'{resample_hours}h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Quote Asset Volume':'sum',
        'Number of Trades':'sum',
        'Taker Buy Base Asset Volume':'sum',
        'Taker Buy Quote Asset Volume':'sum',
        'funding_rate':'last'


    }).dropna()
    # Local ATH/ATL features
    volume_mean = df_resampled['Volume'].rolling(window=112, min_periods=1).mean()
    volume_std = df_resampled['Volume'].rolling(window=112, min_periods=1).std()

    # Z-score = (x - mean) / std
    # Dodaj małą stałą aby uniknąć dzielenia przez 0
    df_resampled['volume_zscore'] = (df_resampled['Volume'] - volume_mean) / (volume_std + 1e-8)
    df_resampled['volume_zscore'] = df_resampled['volume_zscore'].clip(-5, 5)
    funding_mean = df_resampled['funding_rate'].rolling(window=224,min_periods=1).mean()
    funding_std = df_resampled['funding_rate'].rolling(window=224,min_periods=1).std()
    df_resampled['funding_z'] = (df_resampled['funding_rate']-funding_mean)/(funding_std+1e-10)
    df_resampled['funding_z'] = df_resampled['volume_zscore'].clip(-5,5)
    df_resampled['local_ATH'] = df_resampled['Close'].rolling(window=224, min_periods=1).max()
    df_resampled['local_ATL'] = df_resampled['Close'].rolling(window=224, min_periods=1).min()
    df_resampled['pct_change'] = df_resampled['Close'].pct_change(periods=3,fill_method=None)
    df_resampled['pct_change'] = df_resampled['pct_change'].fillna(0.0)
    is_ath = df_resampled['Close'] == df_resampled['local_ATH']
    is_atl = df_resampled['Close'] == df_resampled['local_ATL']

    # Time since local high
    not_ath = ~is_ath
    cumsum_not_ath = not_ath.cumsum()
    last_ath_cumsum = cumsum_not_ath.where(is_ath).ffill().fillna(0)
    df_resampled['time_local_High'] = cumsum_not_ath - last_ath_cumsum
    df_resampled['distance_to_high'] = (df_resampled['Close']/df_resampled['local_ATH'])-1
    df_resampled['distance_to_low'] =  (df_resampled['Close']/df_resampled['local_ATL'])-1
     # Time since local low
    not_atl = ~is_atl
    cumsum_not_atl = not_atl.cumsum()
    last_atl_cumsum = cumsum_not_atl.where(is_atl).ffill().fillna(0)
    df_resampled['time_local_Low'] = cumsum_not_atl - last_atl_cumsum

    # 2. VOLATILITY RATIO (short-term vs long-term volatility)
    # Short-term volatility (6 periods = ~1.5 days for 6h candles)

    df_resampled['trend_slope_long'] = rolling_slope(df_resampled['Close'], window=224)
    # 3. VOLUME-PRICE CORRELATION (smart money detection)
    # 20-period rolling correlation between volume and price
    delta = df_resampled['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=56, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=56, min_periods=1).mean()
    # Avoid divide by zero
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    df_resampled['RSI'] = 100 - (100 / (1 + rs))
    df_resampled['RSI'] = df_resampled['RSI'].fillna(50.0)  # Default to neutral RSI

    # 4. TAKER BUY RATIO (buy pressure)
    df_resampled['taker_buy_ratio'] = (
            df_resampled['Taker Buy Base Asset Volume'] /
            (df_resampled['Volume'] + 1e-8)
    )
    df_resampled['Volume'] = df_resampled['Volume'] / df_resampled['Close']    # Technical indicators with safe NaN handling
    # EMA

    
    # RSI with safe division
    # Volatility
    df_resampled['hour'] =  df_resampled.index.hour
    bb_ma = df_resampled['Close'].rolling(20).mean()
    bb_std = df_resampled['Close'].rolling(20).std()
    df_resampled['bb_upper'] = bb_ma + (bb_std * 2)
    df_resampled['bb_lower'] = bb_ma - (bb_std * 2)
    df_resampled['bb_position'] = (df_resampled['Close'] - df_resampled['bb_lower']) / ( df_resampled['bb_upper'] - df_resampled['bb_lower'] + 1e-8)
    df_resampled['bb_width'] = (df_resampled['bb_upper'] - df_resampled['bb_lower']) / (df_resampled['Close'] + 1e-8)
    # Drop intermediate columns

    df_resampled = df_resampled.drop(columns=['local_ATH','funding_rate','bb_position','bb_width','time_local_Low','time_local_High','pct_change', 'local_ATL','Quote Asset Volume','Taker Buy Quote Asset Volume','Taker Buy Base Asset Volume','bb_upper','bb_lower','Open','Number of Trades','Volume'])
    
    # Drop any remaining NaN rows
    df_resampled = df_resampled.dropna()
    
    # Get feature column names
    feature_columns = df_resampled.columns.tolist()
    
    return df_resampled, feature_columns


def fetch_klines_range(symbol: str, interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch klines over a longer range by paginating."""
    BINANCE_API = "https://api.binance.com"
    all_frames = []
    current    = start_ts

    while current < end_ts:
        resp = requests.get(
            f"{BINANCE_API}/api/v3/klines",
            params={
                "symbol":    symbol,
                "interval":  interval,
                "startTime": current,
                "endTime":   end_ts,
                "limit":     1000,
            },
            timeout=10
        )
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break

        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
        df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)

        all_frames.append(df)
        current = int(raw[-1][6]) + 1  # next start = last close_time + 1ms

        if len(raw) < 1000:
            break

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames).drop_duplicates("open_time")
    return combined.set_index("open_time").sort_index()
def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    BINANCE_API = "https://api.binance.com"
    resp = requests.get(
        f"{BINANCE_API}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10
    )
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    return df.set_index("open_time")





# ── 2. Core analysis ──────────────────────────────────────────────────────────
def analyze_days(symbol: str, lookback_days: int):
    """
    For every 3h candle, look at the next HORIZON candles (24h).
    If price closes higher after HORIZON candles  → LONG case: did it dip below open first?
    If price closes lower after HORIZON candles   → SHORT case: did it spike above open first?

    Results are grouped by candle hour (0,3,6,9,12,15,18,21) so you can see
    which signal hour gives the most reliable dip/spike pattern.
    """
    end_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ts = end_ts - lookback_days * 86400 * 1000


    hourly = fetch_klines_range(symbol, INTRADAY_TF, start_ts, end_ts)



    candles = hourly.resample("3h").agg({
        "open":      "first",
        "high":      "max",
        "low":       "min",
        "close":     "last",
        "volume":    "sum",
        "close_time":"last",
    }).dropna()
    HORIZON = 8
    candle_list = list(candles.iterrows())
    long_results  = []
    short_results = []

    for idx, (candle_ts, candle_row) in enumerate(candle_list):
        # Need enough candles ahead for the full horizon
        if idx + HORIZON >= len(candle_list):
            continue

        candle_open  = candle_row["open"]
        horizon_close_ts, horizon_row = candle_list[idx + HORIZON]
        horizon_close = horizon_row["close"]
        candle_hour   = candle_ts.hour
        is_up         = horizon_close > candle_open

        pct_move = (horizon_close - candle_open) / candle_open * 100

        # Get all 1h candles between signal and horizon close
        horizon_end = horizon_row["close_time"]
        mask        = (hourly.index >= candle_ts) & (hourly.index < horizon_end)
        window      = hourly[mask]
        if window.empty:
            continue

        if is_up:
            # LONG: did price dip below candle open within the horizon?
            dip_level    = candle_open
            min_low      = window["low"].min()
            had_dip      = min_low < dip_level
            move_pct     = (candle_open - min_low) / candle_open * 100
            extreme_hour = window["low"].idxmin().hour

            long_results.append({
                "candle_ts":     candle_ts,
                "candle_hour":   candle_hour,       # 0,3,6,9,12,15,18,21
                "candle_open":   candle_open,
                "horizon_close": horizon_close,
                "pct_move":      round(pct_move, 2),
                "extreme_price": round(min_low, 2),
                "adverse_pct":   round(move_pct, 2),
                "had_adverse":   had_dip,
                "extreme_hour":  extreme_hour,
            })
        else:
            # SHORT: did price spike above candle open within the horizon?
            spike_level  = candle_open
            max_high     = window["high"].max()
            had_spike    = max_high > spike_level
            move_pct     = (max_high - candle_open) / candle_open * 100
            extreme_hour = window["high"].idxmax().hour

            short_results.append({
                "candle_ts":     candle_ts,
                "candle_hour":   candle_hour,
                "candle_open":   candle_open,
                "horizon_close": horizon_close,
                "pct_move":      round(abs(pct_move), 2),
                "extreme_price": round(max_high, 2),
                "adverse_pct":   round(move_pct, 2),
                "had_adverse":   had_spike,
                "extreme_hour":  extreme_hour,

            })
    long_df = pd.DataFrame(long_results)
    short_df = pd.DataFrame(short_results)
    long_adv = long_df[long_df["had_adverse"]]["adverse_pct"]
    short_adv = short_df[short_df["had_adverse"]]["adverse_pct"]
    all_adv = pd.concat([long_adv, short_adv])
    all_change = pd.concat([long_df["pct_move"], short_df["pct_move"]])
    return (round(all_change.mean(),4)/100), (round(all_change.median(),4)/100), (round(all_adv.mean(),4)/100), (round(all_adv.median(),4)/100)
def build_windows(
    df_features: pd.DataFrame,
    window_days: int,
    resample_hours: int,
    horizon: int,
    step: int
) -> Tuple[np.ndarray, np.ndarray]:

    steps_per_day = 24 / resample_hours
    window_size = int(window_days * steps_per_day)
    
    if window_size < 1:
        raise ValueError(f"Window size too small: {window_size} steps. Increase window_days or decrease resample_hours.")
    
    if len(df_features) < window_size + horizon:
        raise ValueError(f"Not enough data: {len(df_features)} samples, need at least {window_size + horizon}")
    
    # Convert to numpy array
    feature_array = df_features.values.astype(np.float32)
    num_features = feature_array.shape[1]
    
    # Extract Close prices for targets (assuming 'Close' is in the features)
    close_idx = df_features.columns.get_loc('Close')
    close_prices = feature_array[:, close_idx]
    
    # Calculate number of windows
    max_start = len(feature_array) - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError(f"Not enough data: {len(feature_array)} samples, need at least {window_size + horizon}")
    
    # Generate all valid start indices with step
    start_indices = np.arange(0, max_start, step)
    
    if len(start_indices) == 0:
        raise ValueError(f"No valid windows can be created with the given parameters.")
    
    # Vectorized window creation using advanced indexing
    # Create index array for all windows: shape (num_windows, window_size)
    window_indices = start_indices[:, None] + np.arange(window_size)
    
    # Extract windows: shape (num_windows, window_size, num_features)
    X = feature_array[window_indices].astype(np.float32)
    
    # Targets: Close price horizon steps ahead
    target_indices = start_indices + window_size + horizon - 1
    # Ensure we don't go out of bounds
    valid_mask = target_indices < len(close_prices)
    X = X[valid_mask]
    target_indices = target_indices[valid_mask]
    
    y = close_prices[target_indices].astype(np.float32)
    
    return X, y

def get_evaluate_window(symbol:str,window_days:int,resample_hours:int):
    steps_per_day = 24 // resample_hours
    window_steps = window_days * steps_per_day
    df = download_data(symbol,(window_days*60))
    df_c,_ = compute_features(df,resample_hours=resample_hours)
    window_df = df_c.iloc[-window_steps:]

    X = window_df.values.astype(np.float32)
    return X
def scale_live_window(X, scaler):
    """
    X: np.ndarray (T, F)
    returns: torch.Tensor (1, T, F)
    """
    T, F = X.shape

    # flatten time
    X_2d = X.reshape(-1, F)

    # scale
    X_scaled = scaler.transform(X_2d)

    # reshape back
    X_scaled = X_scaled.reshape(1, T, F)

    return torch.tensor(X_scaled, dtype=torch.float32)
def make_dataset(
    symbol: str,
    months: int,
    window_days: int,
    resample_hours: int,
    horizon: int,
    step: int,
    cutoff:int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Orchestrate dataset creation: download, compute features, build windows, and save.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    df_raw = download_data(symbol, months,cutoff=cutoff)

    df_raw = df_raw[df_raw.index >= df_raw.index[0].ceil('D')]

    df_features, feature_names = compute_features(df_raw, resample_hours)

    X, y = build_windows(df_features, window_days, resample_hours, horizon, step)

    
    # Create output directory
    outdir_path = Path(f"{symbol}_{window_days}_{resample_hours}_{horizon}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Save arrays
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    np.save(x_path, X)
    np.save(y_path, y)
    
    # Save feature names as JSON
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)

    
    return X, y, feature_names





# Keep these functions for backward compatibility with other modules
def safe_divide(numerator, denominator, default=1.0):
    """Helper function to safely divide two values."""
    try:
        num = float(numerator)
        den = float(denominator)
        if den <= 0:
            return default
        return num / den
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def safe_float(value):
    """Helper function to safely convert values to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        # For nested dictionaries, try to get the first numeric value
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0





# Backward compatibility: keep get_training_data for existing code
def get_training_data(symbol, days, months=-1, interval=12):
    """
    Legacy function for backward compatibility.
    Use make_dataset() or main() for new code.
    """
    df = download_data(symbol=symbol, months=months)
    window_hours = days * 24
    resample_hours = interval
    
    df_features, feature_names = compute_features(df, resample_hours)
    
    # Use default horizon=1, step=1
    X, y = build_windows(df_features, days, resample_hours, horizon=1, step=1)
    
    return X, y, feature_names


# Token/Wallet data collection functions (stubs for backward compatibility)
# NOTE: These functions were removed during refactoring as they are part of a different
# API-based token data collection system. They need to be re-implemented based on
# your original API integration code.







