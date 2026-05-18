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
from scipy.signal import periodogram
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
def align_to_hour(ts_ms):
    # floor to full hour
    return int((ts_ms // (3600 * 1000)) * (3600 * 1000))
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
def lyapunov_exponent(series, lag=1):
    """
    Positive → chaotic, unpredictable
    Negative → stable, predictable
    Near 0   → edge of chaos, regime transition
    """
    n = len(series)
    divergences = []
    for i in range(1, n - lag):
        d0 = abs(series[i] - series[i-1]) + 1e-10
        d1 = abs(series[i+lag] - series[i]) + 1e-10
        divergences.append(np.log(d1 / d0))
    return np.mean(divergences)


def phase_space_density(series, lag=6, threshold=0.02):
    """
    Reconstructs the attractor of the price dynamics.
    Measures how often price returns to current region.
    High density = strong attractor = price likely to stay/return
    Low density  = price in unexplored territory = higher uncertainty
    """
    n = len(series)
    if n < lag * 2:
        return 0.5

    # embed in 2D phase space
    x1 = series[:-lag]
    x2 = series[lag:]

    # normalise
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-8)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-8)

    # current point
    cx, cy = x1[-1], x2[-1]

    # count nearby points (recurrence)
    distances = np.sqrt((x1 - cx) ** 2 + (x2 - cy) ** 2)
    return (distances < threshold).mean()


def variance_ratio(series, k=6):
    """
    VR > 1 → positive autocorrelation → momentum
    VR < 1 → negative autocorrelation → mean reversion
    VR = 1 → random walk → no edge
    """
    n = len(series)
    rets = np.diff(np.log(series + 1e-8))
    mu = rets.mean()

    # variance of 1-period returns
    var1 = np.sum((rets - mu) ** 2) / (n - 2)

    # variance of k-period returns
    rets_k = np.log(series[k:] / (series[:-k] + 1e-8))
    mu_k = k * mu
    var_k = np.sum((rets_k - mu_k) ** 2) / (k * (n - k - 1))

    return var_k / (var1 + 1e-8)


def dfa(series, min_scale=4, max_scale=None):
    """
    Measures long-range correlations.
    alpha > 0.5 → persistent (trending)
    alpha < 0.5 → anti-persistent (mean reverting)
    alpha = 0.5 → uncorrelated (random walk)
    More robust than Hurst for short series.
    """
    n = len(series)
    if max_scale is None:
        max_scale = n // 4

    scales = np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        num=10, dtype=int
    )
    scales = np.unique(scales)

    # cumulative sum (integrate)
    y = np.cumsum(series - series.mean())

    fluctuations = []
    for scale in scales:
        n_segments = n // scale
        if n_segments < 2:
            continue
        rms = []
        for seg in range(n_segments):
            segment = y[seg * scale:(seg + 1) * scale]
            x_seg = np.arange(scale)
            # detrend segment with linear fit
            coeffs = np.polyfit(x_seg, segment, 1)
            trend = np.polyval(coeffs, x_seg)
            rms.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuations.append(np.mean(rms))

    if len(fluctuations) < 2:
        return 0.5

    alpha = np.polyfit(
        np.log(scales[:len(fluctuations)]),
        np.log(fluctuations), 1
    )[0]
    return alpha


def ou_parameters(series):
    """
    Fits Ornstein-Uhlenbeck process to price.
    Returns mean reversion speed theta.
    High theta = fast mean reversion = fade signals
    Low theta  = slow reversion = trend following works
    """
    x = series[:-1]
    dx = np.diff(series)

    # OLS regression: dx = -theta * x * dt + noise
    if len(x) < 10 or x.std() < 1e-8:
        return 0.0

    theta = -np.polyfit(x, dx, 1)[0]
    return max(theta, 0.0)

def spectral_entropy(series):
    """
    Measures how spread out the frequency content is.
    Low  = energy concentrated in few frequencies = trending
    High = energy spread across all frequencies   = random/noisy
    """
    _, psd = periodogram(series)
    psd_norm = psd / (psd.sum() + 1e-8)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))


def liq_proximity_features(df, window=168):
    """
    Approximates liquidation clusters from price history.

    Logic:
    - High volume at a price level = many positions opened there
    - Their liquidations sit at fixed distances below/above
    - Common retail leverage = 10x → liq at ±10% from entry
    - Common pro leverage = 20x → liq at ±5% from entry
    """
    close = df['Close'].values
    volume = df['Volume'].values
    n = len(close)

    # features to compute
    liq_dense_above = np.zeros(n)
    liq_dense_below = np.zeros(n)
    liq_imbalance = np.zeros(n)
    nearest_liq_above = np.zeros(n)
    nearest_liq_below = np.zeros(n)

    for i in range(window, n):
        current = close[i]

        # past window of prices and volumes
        past_prices = close[i - window:i]
        past_volumes = volume[i - window:i]

        # normalise volumes to weights
        weights = past_volumes / (past_volumes.sum() + 1e-8)

        liq_above_density = 0.0
        liq_below_density = 0.0
        nearest_above = current * 1.20  # default: 20% away
        nearest_below = current * 0.80

        for lev in [5, 10, 20]:
            liq_long = past_prices * (1 - 1 / lev)  # long liquidations
            liq_short = past_prices * (1 + 1 / lev)  # short liquidations

            # density of long liquidations below current price
            long_liq_below = liq_long[liq_long < current]
            long_liq_weights = weights[liq_long < current]
            liq_below_density += long_liq_weights.sum()

            # density of short liquidations above current price
            short_liq_above = liq_short[liq_short > current]
            short_liq_weights = weights[liq_short > current]
            liq_above_density += short_liq_weights.sum()

            # nearest cluster
            if len(long_liq_below) > 0:
                nearest_below = max(
                    nearest_below, long_liq_below.max()
                )
            if len(short_liq_above) > 0:
                nearest_above = min(
                    nearest_above, short_liq_above.min()
                )

        liq_dense_above[i] = liq_above_density
        liq_dense_below[i] = liq_below_density
        liq_imbalance[i] = (
                                   liq_above_density - liq_below_density
                           ) / (liq_above_density + liq_below_density + 1e-8)
        nearest_liq_above[i] = (nearest_above - current) / current
        nearest_liq_below[i] = (current - nearest_below) / current

    df['liq_dense_above'] = liq_dense_above
    df['liq_dense_below'] = liq_dense_below
    df['liq_imbalance'] = liq_imbalance
    df['nearest_liq_above'] = nearest_liq_above  # % distance to nearest cluster above
    df['nearest_liq_below'] = nearest_liq_below  # % distance to nearest cluster below

    return df


def accumulation_distribution(df, forecast_horizon=6):
    mfm = (
        (df['Close'] - df['Low']) -
        (df['High'] - df['Close'])
    ) / (df['High'] - df['Low'] + 1e-8)

    mfv = mfm * df['Volume']
    adl = mfv.cumsum()
    adl_std = adl.rolling(42).std()

    df['adl_slope'] = adl.diff(forecast_horizon) / (adl_std + 1e-8)
    df['adl_div']   = (
        np.sign(df['Close'].diff(forecast_horizon)) *
        np.sign(adl.diff(forecast_horizon))
    )
    # mfm_mean dropped — redundant with candle_pos

    return df
def compute_features(df: pd.DataFrame, resample_hours: int,offset_hours=0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Resample data and compute technical indicators once on the full dataset.
    
    Args:
        df: Raw OHLCV DataFrame with hourly data
        resample_hours: Resampling interval in hours
    
    Returns:
        Tuple of (feature DataFrame, feature column names)
    """
    # Resample to specified interval
    resample_kwargs = {}
    if offset_hours:
        resample_kwargs["offset"] = f"{offset_hours}h"
    df_resampled = df.resample(f'{resample_hours}h', **resample_kwargs).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Quote Asset Volume':'sum',
        'Number of Trades':'sum',
        'Taker Buy Base Asset Volume':'sum',
        'Taker Buy Quote Asset Volume':'sum',
        'funding_rate':'last',


    }).dropna()
    # Local ATH/ATL features
    volume_mean = df_resampled['Volume'].rolling(window=168, min_periods=1).mean()
    volume_std = df_resampled['Volume'].rolling(window=168, min_periods=1).std()
    df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled.index.hour / 24)
    df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled.index.hour / 24)
    # Z-score = (x - mean) / std
    # Dodaj małą stałą aby uniknąć dzielenia przez 0
    df_resampled['volume_zscore'] = (df_resampled['Volume'] - volume_mean) / (volume_std + 1e-8)
    df_resampled['volume_zscore'] = df_resampled['volume_zscore'].clip(-5, 5)
    funding_mean = df_resampled['funding_rate'].rolling(window=84,min_periods=1).mean()
    funding_std = df_resampled['funding_rate'].rolling(window=84,min_periods=1).std()
    df_resampled['funding_z'] = (df_resampled['funding_rate']-funding_mean)/(funding_std+1e-10)
    df_resampled['funding_z'] = df_resampled['funding_z'].clip(-5,5)
    df_resampled['vwap'] = (
            df_resampled['Quote Asset Volume'] /
            (df_resampled['Volume'] + 1e-8)
    )
    df_resampled['local_ATH'] = df_resampled['High'].rolling(window=168, min_periods=1).max()
    df_resampled['local_ATL'] = df_resampled['Low'].rolling(window=168, min_periods=1).min()
    df_resampled['pct_change'] = df_resampled['Close'].pct_change(periods=8,fill_method=None)
    df_resampled['funding_change'] = df_resampled['funding_rate'].pct_change(periods=8,fill_method=None)
    df_resampled['pct_change'] = df_resampled['pct_change'].fillna(0.0)
    is_ath = df_resampled['Close'] == df_resampled['local_ATH']
    is_atl = df_resampled['Close'] == df_resampled['local_ATL']
    df_resampled['r6'] = df_resampled['vwap'].pct_change(periods=6, fill_method=None)
    df_resampled['candle_pos'] = (df_resampled['Close'] - df_resampled['Low'])/(df_resampled['High']-df_resampled['Low']+ 1e-8)
    df_resampled['candle_pos'] = df_resampled['candle_pos'].clip(0.0, 1.0)
    trades_mean = df_resampled['Number of Trades'].rolling(window=168,min_periods=1).mean()
    trades_std = df_resampled['Number of Trades'].rolling(window=168,min_periods=1).std()
    df_resampled['trades_z'] = (df_resampled['Number of Trades'] - trades_mean)/(trades_std+1e-10)
    df_resampled['trades_z'] = df_resampled['trades_z'].clip(-5,5)
    df_resampled['trades_change'] = df_resampled['Number of Trades'].pct_change(periods=1,fill_method=None)
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
    df_resampled['lyapunov'] = df_resampled['Close'].rolling(90).apply(
        lambda x: lyapunov_exponent(x), raw=True)
    df_resampled['phase_density'] = df_resampled['Close'].rolling(60).apply(
        lambda x: phase_space_density(x), raw=True
    )
    df_resampled['variance_ratio'] = df_resampled['Close'].rolling(90).apply(
        lambda x: variance_ratio(x, k=6), raw=True)
    df_resampled['spectral_entropy'] = df_resampled['Close'].rolling(64).apply(
        lambda x: spectral_entropy(x), raw=True
    )
    df_resampled['trend_slope_short'] = rolling_slope(df_resampled['vwap'], window=168)

    delta = df_resampled['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=30, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=30, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    df_resampled['RSI'] = 100 - (100 / (1 + rs))
    df_resampled['RSI'] = df_resampled['RSI'].fillna(50.0)  # Default to neutral RSI

    # 4. TAKER BUY RATIO (buy pressure)
    df_resampled['taker_buy_ratio'] = (
            df_resampled['Taker Buy Base Asset Volume'] /
            (df_resampled['Volume'] + 1e-8)
    )
    df_resampled['taker_buy_ratio'] = df_resampled['taker_buy_ratio'].rolling(6).mean()
    df_resampled['vol_change'] = df_resampled['Volume'].pct_change(periods=1)

    df_resampled['hour'] =  df_resampled.index.hour
    bb_ma = df_resampled['Close'].rolling(30).mean()
    bb_std = df_resampled['Close'].rolling(30).std()
    df_resampled['bb_upper'] = bb_ma + (bb_std * 2)
    df_resampled['bb_lower'] = bb_ma - (bb_std * 2)
    df_resampled['distance_hl_position'] = (df_resampled['vwap'] - df_resampled['local_ATL'])/(df_resampled['local_ATH']-df_resampled['local_ATL'])
    df_resampled['distance_hl_position'] = df_resampled['distance_hl_position'].clip(0.0,1.0)
    df_resampled['bb_width'] = (df_resampled['bb_upper'] - df_resampled['bb_lower']) / (df_resampled['Close'])
    df_resampled['bb_pos'] = (df_resampled['vwap']-df_resampled['bb_lower'])/(df_resampled['bb_upper']-df_resampled['bb_lower'])
    df_resampled['bb_squeeze'] = (
            df_resampled['bb_width'] < df_resampled['bb_width'].rolling(60).quantile(0.2)
    ).astype(float)
    df_resampled['ou_speed'] = df_resampled['Close'].rolling(60).apply(
        lambda x: ou_parameters(x), raw=True
    )
    daily_vwap = df_resampled['vwap'].rolling(6).mean()
    df_resampled['close_vwap_dev'] = (df_resampled['Close']-daily_vwap)/daily_vwap
    # Drop intermediate columns
    df_resampled = accumulation_distribution(df_resampled)


    df_resampled = df_resampled.drop(columns=['local_ATH','funding_change','volume_zscore','phase_density','vol_change','funding_z','bb_squeeze','time_local_Low','time_local_High','candle_pos','trades_change','hour_sin','hour_cos','trades_z','vwap','pct_change','distance_to_high','distance_to_low','Number of Trades','hour','funding_rate', 'local_ATL','Quote Asset Volume','Taker Buy Quote Asset Volume','Taker Buy Base Asset Volume','bb_upper','bb_lower','Open','Volume'])
    
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, timestamps) where timestamps are the candle-index datetimes
    of each window's first candle, used by make_dataset to sort across phases."""

    steps_per_day = 24 / resample_hours
    window_size = int(window_days * steps_per_day)

    if window_size < 1:
        raise ValueError(
            f"Window size too small: {window_size} steps. Increase window_days or decrease resample_hours.")

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

    # Timestamps of each window's first candle — used for cross-phase chronological sorting
    timestamps = np.array(df_features.index[start_indices[valid_mask]], dtype="datetime64[ns]")

    return X, y, timestamps

def get_evaluate_window(symbol:str,window_days:int,resample_hours:int):
    steps_per_day = 24 // resample_hours
    window_steps = window_days * steps_per_day
    df = download_data(symbol,(window_days*2))
    df_c,_ = compute_features(df,resample_hours=resample_hours)
    window_df = df_c.iloc[-window_steps:]

    X = window_df.values.astype(np.float32)
    return X
def scale_live_window(X, scaler,feature_names):
    """
    X: np.ndarray (T, F)
    returns: torch.Tensor (1, T, F)
    """
    MODEL_EXCLUDE = {'Close', 'High', 'Low'}
    model_idx = [i for i, n in enumerate(feature_names) if n not in MODEL_EXCLUDE]
    X = X[:, model_idx]
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
    step_hours: int,
    cutoff: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Orchestrate dataset creation: download, compute features, build windows, and save.

    step_hours controls how many hours separate consecutive window starts, independently
    of resample_hours.

    - step_hours >= resample_hours : single-phase, windows advance by
      step_hours // resample_hours candles (original behaviour when step_hours == resample_hours).
    - step_hours < resample_hours  : multi-phase; resample_hours must be divisible by
      step_hours.  The raw 1 h data is resampled n_phases = resample_hours // step_hours
      times with increasing hour offsets, giving n_phases times more windows while keeping
      the same candle resolution.  All windows are merged and sorted chronologically so
      that the downstream train/val/test time-split remains valid.

    Returns:
        Tuple of (X, y, feature_names)
    """
    if step_hours <= 0:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours}")

    outdir_path = Path(f"{symbol}_{window_days}_{resample_hours}_{horizon}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    df_raw = download_data(symbol, months, cutoff=cutoff)
    df_raw = df_raw[df_raw.index >= df_raw.index[0].ceil('D')]

    if step_hours >= resample_hours:
        # ── Simple case: single resample phase ───────────────────────────────
        step_candles = step_hours // resample_hours
        df_features, feature_names = compute_features(df_raw, resample_hours)
        X, y, _ = build_windows(df_features, window_days, resample_hours, horizon, step_candles)

    else:
        # ── Multi-phase case ─────────────────────────────────────────────────
        if resample_hours % step_hours != 0:
            raise ValueError(
                f"step_hours ({step_hours}) must divide resample_hours ({resample_hours}) "
                f"evenly when step_hours < resample_hours.  "
                f"Valid step_hours values: {[resample_hours // k for k in range(2, resample_hours + 1) if resample_hours % k == 0]}"
            )
        n_phases = resample_hours // step_hours
        print(f"[make_dataset] step_hours={step_hours} < resample_hours={resample_hours}: "
              f"building {n_phases} phase-shifted datasets (offsets: "
              f"{[p * step_hours for p in range(n_phases)]}h)")

        all_X:  list[np.ndarray] = []
        all_y:  list[np.ndarray] = []
        all_ts: list[np.ndarray] = []

        for phase in range(n_phases):
            offset = phase * step_hours
            df_features, feature_names = compute_features(df_raw, resample_hours, offset_hours=offset)
            X_p, y_p, ts_p = build_windows(df_features, window_days, resample_hours, horizon, step=1)
            all_X.append(X_p)
            all_y.append(y_p)
            all_ts.append(ts_p)
            print(f"  phase {phase} (offset={offset}h): {len(X_p)} windows")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        timestamps = np.concatenate(all_ts, axis=0)

        # Sort chronologically so that the train/val/test index-based split in
        # Train_val() still corresponds to oldest→newest.
        sort_idx = np.argsort(timestamps)
        X = X[sort_idx]
        y = y[sort_idx]
        print(f"  total windows after merge + sort: {len(X)}")

    # ── Persist ───────────────────────────────────────────────────────────────
    df_features.to_csv(f"{outdir_path}/df.csv")

    x_path       = outdir_path / "X.npy"
    y_path       = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    np.save(x_path, X)
    np.save(y_path, y)

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







