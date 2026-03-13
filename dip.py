"""
Intraday Dip Analysis — Binance API
-------------------------------------
For every day BTC (or any symbol) closed ABOVE the daily open,
check if price dipped BELOW the open at any point during the day.

This tells you: "How often does a 24h winner dip first before going up?"

Requirements: pip install requests pandas matplotlib
"""

import requests
import pandas as pd
from datetime import datetime, timezone

BINANCE_API = "https://api.binance.com"

# ── Config ────────────────────────────────────────────────────────────────────
         # candles in your trading horizon (8 × 3h = 24h)

# ── 1. Fetch klines ───────────────────────────────────────────────────────────
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
def analyze_days(symbol: str, lookback_days: int, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    print(f"  Got {len(candles)} 3h candles")

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
            dip_level    = candle_open * (1 - threshold)
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
            spike_level  = candle_open * (1 + threshold)
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

    return pd.DataFrame(long_results), pd.DataFrame(short_results)


# ── 3. Stats ──────────────────────────────────────────────────────────────────
def print_stats_side(df: pd.DataFrame, side: str, threshold: float):
    label_adverse = "dip below open" if side == "LONG" else "spike above open"
    label_dir     = "UP" if side == "LONG" else "DOWN"

    total   = len(df)
    adverse = df["had_adverse"].sum()
    rate    = adverse / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"  {side} — 24h {label_dir} candles  ({label_adverse})")
    print(f"{'='*60}")
    print(f"  Total candles  : {total}")
    print(f"  Had adverse    : {adverse} ({rate:.1f}%)")
    print(f"  Threshold      : {threshold*100:.1f}%")

    if adverse == 0:
        return

    adv_df = df[df["had_adverse"]]
    print(f"  Avg depth      : {adv_df['adverse_pct'].mean():.2f}%")
    print(f"  Median depth   : {adv_df['adverse_pct'].median():.2f}%")
    print(f"  Max depth      : {adv_df['adverse_pct'].max():.2f}%")
    print()

    # ── Per signal-hour breakdown ─────────────────────────────────
    print(f"  Adverse rate & avg depth & median depth by signal candle hour (UTC):")
    hours_stats = df.groupby("candle_hour").agg(
        n             = ("had_adverse", "count"),
        adverse_rate  = ("had_adverse", "mean"),
        avg_depth     = ("adverse_pct", "mean"),
        median_depth=("adverse_pct", "median"),
    ).reset_index()
    hours_stats["adverse_rate"] = (hours_stats["adverse_rate"] * 100).round(1)
    hours_stats["avg_depth"]    = hours_stats["avg_depth"].round(3)
    hours_stats["median_depth"] = hours_stats["median_depth"].round(3)
    print(hours_stats.to_string(index=False))


def print_stats(symbol,long_df: pd.DataFrame, short_df: pd.DataFrame, threshold: float):
    print(f"\n{'='*60}")
    print(f"3H CANDLE DIP ANALYSIS — {symbol}")
    print(f"Lookback: {LOOKBACK_DAYS} days | Horizon: {HORIZON} candles | Threshold: {threshold*100:.1f}%")
    print(f"{'='*60}")
    total = len(long_df) + len(short_df)
    print(f"Total 3h candles analyzed : {total}")
    print(f"  UP   (close > open) : {len(long_df)}")
    print(f"  DOWN (close < open) : {len(short_df)}")
    print_stats_side(long_df,  "LONG",  threshold)
    print_stats_side(short_df, "SHORT", threshold)


# ── 4. Plot ───────────────────────────────────────────────────────────────────






# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]
    for symbol in symbols:
        long_df, short_df = analyze_days(symbol, LOOKBACK_DAYS, DIP_THRESHOLD)
        print(f"calculating {symbol} dips data")
        if long_df.empty and short_df.empty:
            print("No data found.")
        else:
            long_df.to_csv(f"marketdata/{symbol}_long_dip_data.csv", index=False)
            short_df.to_csv(f"marketdata/{symbol}_short_spike_data.csv", index=False)
            print(f"Raw data saved to {symbol}_long_dip_data.csv and {symbol}_short_spike_data.csv")

            print_stats(symbol,long_df, short_df, DIP_THRESHOLD)