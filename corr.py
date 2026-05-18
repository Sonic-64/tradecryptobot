"""
Feature Correlation Table — from saved X.npy / y.npy
------------------------------------------------------
Loads your existing make_dataset() output and correlates every feature
(plus streak/momentum features derived from the window) against the
forward return implied by y.

Usage:
    python corr_from_npy.py --dir BTCUSDT_7_3_8
    python corr_from_npy.py --dir BTCUSDT_7_3_8 --method spearman
    python corr_from_npy.py --dir . --dirs BTCUSDT_7_3_8 ETHUSDT_7_3_8 SOLUSDT_7_3_8

Directory must contain:
    X.npy          shape (N, window, F)
    y.npy          shape (N,)  — future Close price
    features.json  list of F feature names
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# ── CONFIG ────────────────────────────────────────────────────────────────────

MIN_OBS   = 50     # minimum non-NaN pairs for a valid correlation
ALPHA     = 0.05   # significance threshold

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_dir(path: str):
    X = np.load(os.path.join(path, "X.npy"))           # (N, W, F)
    y = np.load(os.path.join(path, "y.npy"))           # (N,)
    with open(os.path.join(path, "features.json")) as f:
        feature_names = json.load(f)
    print(f"  Loaded: X{X.shape}  y{y.shape}  features={len(feature_names)}")
    return X, y, feature_names


# ── FORWARD RETURN ────────────────────────────────────────────────────────────

def get_fwd_return(X, y, feature_names):
    """log(y / close_at_last_window_candle)"""
    close_idx = feature_names.index("Close")
    close_now = X[:, -1, close_idx]           # last candle's Close in each window
    fwd = np.log(y / (close_now + 1e-9))
    return fwd


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_features(X, feature_names):
    """
    Build a DataFrame of signals to correlate.
    For point-in-time features (e.g. RSI, funding) we take the last candle.
    For window-level features (streak, momentum) we compute from the Close series.
    """
    close_idx = feature_names.index("Close")
    close_series = X[:, :, close_idx]          # (N, W)

    rows = {}

    # ── A) Last-candle values for every raw feature ──────────────────────────
    for i, name in enumerate(feature_names):
        rows[f"last_{name}"] = X[:, -1, i]

    # ── B) Streak features (from Close series in window) ─────────────────────
    log_ret = np.diff(np.log(close_series + 1e-9), axis=1)   # (N, W-1)

    # Signed streak at end of window
    streak = np.zeros(len(X))
    for n in range(len(X)):
        s = 0
        for r in reversed(log_ret[n]):
            if (r > 0 and s >= 0) or (r < 0 and s <= 0):
                s += (1 if r > 0 else -1)
            else:
                break
        streak[n] = s




    # ── D) Volatility of the window ───────────────────────────────────────────


    return pd.DataFrame(rows)


# ── CORRELATION ───────────────────────────────────────────────────────────────
def compute_feature_corr(features_df: pd.DataFrame,
                         method="pearson"):

    if method == "pearson":
        corr_matrix = features_df.corr(method="pearson")
    else:
        corr_matrix = features_df.corr(method="spearman")

    return corr_matrix


def plot_feature_corr(corr_matrix: pd.DataFrame,
                      symbol: str,
                      method: str,
                      out_path: str):

    matrix = corr_matrix.values

    vmax = min(np.nanmax(np.abs(matrix)), 1.0)

    fig, ax = plt.subplots(
        figsize=(16, 14),
        facecolor="#0d0d12"
    )

    ax.set_facecolor("#0d0d12")

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest"
    )

    labels = corr_matrix.columns.tolist()

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=90,
        fontsize=7,
        color="#ddd"
    )

    ax.set_yticklabels(
        labels,
        fontsize=7,
        color="#ddd"
    )

    cb = plt.colorbar(
        im,
        ax=ax,
        fraction=0.02,
        pad=0.01
    )

    cb.ax.tick_params(colors="#888", labelsize=8)

    cb.set_label(
        f"{method.capitalize()} correlation",
        color="#888",
        fontsize=8
    )

    ax.set_title(
        f"{symbol} — Feature Correlation Matrix",
        fontsize=12,
        color="#e0e0e0",
        pad=12
    )

    plt.tight_layout()

    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="#0d0d12"
    )

    print(f"  Feature corr plot → {out_path}")
def compute_corr(features_df: pd.DataFrame, fwd: np.ndarray, method="pearson"):
    target = pd.Series(fwd, name="fwd_return")
    records = []
    for col in features_df.columns:
        x = features_df[col].values.astype(float)
        y = target.values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if mask.sum() < MIN_OBS:
            records.append({"feature": col, "corr": np.nan, "p_value": np.nan, "n": mask.sum()})
            continue
        if method == "pearson":
            r, p = stats.pearsonr(x[mask], y[mask])
        else:
            r, p = stats.spearmanr(x[mask], y[mask])
        records.append({"feature": col, "corr": round(r, 4), "p_value": round(p, 4), "n": int(mask.sum())})

    df = pd.DataFrame(records).set_index("feature")
    df["abs_corr"] = df["corr"].abs()
    df["significant"] = df["p_value"] < ALPHA
    return df.sort_values("abs_corr", ascending=False)


# ── PLOT ──────────────────────────────────────────────────────────────────────

def plot_single(corr_df: pd.DataFrame, symbol: str, method: str, out_path: str):
    top = corr_df.dropna(subset=["corr"]).head(40)
    features = top.index.tolist()[::-1]
    values   = top["corr"].values[::-1]
    sig      = top["significant"].values[::-1]

    fig, ax = plt.subplots(figsize=(11, max(6, len(features) * 0.32)),
                           facecolor="#0d0d12")
    ax.set_facecolor("#0d0d12")

    colors = ["#4caf50" if v > 0 else "#f44336" for v in values]
    bars   = ax.barh(features, values, color=colors, alpha=0.8, height=0.7)

    # Mark non-significant bars
    for bar, s in zip(bars, sig):
        if not s:
            bar.set_alpha(0.3)
            bar.set_hatch("//")

    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel(f"{method.capitalize()} correlation with {' '.join(os.path.basename(out_path).split('_')[:1])} forward return",
                  fontsize=9, color="#aaa")
    ax.set_title(f"{symbol} — Top 40 Features by |corr|  (hatched = p≥{ALPHA})",
                 fontsize=11, color="#e0e0e0", pad=10)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.spines[["top","right","bottom","left"]].set_color("#333")

    # Value labels
    for bar, val, s in zip(bars, values, sig):
        ax.text(val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}{'*' if s else ''}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=7, color="#ccc")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d12")
    print(f"  Plot → {out_path}")


def plot_multi(all_corr: dict, method: str, out_path: str):
    """Heatmap comparing correlations across multiple symbols."""
    symbols  = list(all_corr.keys())
    all_feats = sorted(set(f for df in all_corr.values() for f in df.index),
                       key=lambda f: np.nanmean([all_corr[s].loc[f, "abs_corr"]
                                                  if f in all_corr[s].index else np.nan
                                                  for s in symbols]),
                       reverse=True)[:50]

    matrix = np.full((len(all_feats), len(symbols)), np.nan)
    sig_mat = np.zeros((len(all_feats), len(symbols)), dtype=bool)
    for j, sym in enumerate(symbols):
        df = all_corr[sym]
        for i, feat in enumerate(all_feats):
            if feat in df.index:
                matrix[i, j] = df.loc[feat, "corr"]
                sig_mat[i, j] = df.loc[feat, "significant"]

    vmax = min(np.nanmax(np.abs(matrix)), 0.4)
    fig, ax = plt.subplots(figsize=(max(9, len(symbols) * 2.5),
                                     max(8, len(all_feats) * 0.33)),
                           facecolor="#0d0d12")
    ax.set_facecolor("#0d0d12")

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    for i in range(len(all_feats)):
        for j in range(len(symbols)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            s   = sig_mat[i, j]
            txt = f"{val:+.2f}{'*' if s else ''}"
            col = "white" if abs(val) > vmax * 0.5 else "#bbb"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=col, fontweight="bold" if s else "normal")

    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, fontsize=10, color="#e0e0e0")
    ax.set_yticks(range(len(all_feats)))
    ax.set_yticklabels(all_feats, fontsize=7.5, color="#ddd")
    ax.tick_params(left=False, bottom=False)

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, shrink=0.6)
    cb.ax.tick_params(colors="#888", labelsize=8)
    cb.set_label(f"{method.capitalize()} r", color="#888", fontsize=8)

    ax.set_title(f"Feature → Forward Return Correlation  (* p<{ALPHA})\n"
                 f"Top 50 by mean |corr|, {method}",
                 fontsize=11, color="#e0e0e0", pad=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d12")
    print(f"  Multi-symbol plot → {out_path}")
# ── PRINT FEATURE ↔ FEATURE CORRELATION ─────────────────────────────────────

def print_top_feature_pairs(corr_matrix: pd.DataFrame,
                            top_n=40,
                            min_corr=0.5):

    print("\n  Top feature ↔ feature correlations:\n")

    pairs = []

    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            f1 = cols[i]
            f2 = cols[j]

            corr = corr_matrix.iloc[i, j]

            if np.isnan(corr):
                continue

            pairs.append((f1, f2, corr, abs(corr)))

    pairs.sort(key=lambda x: x[3], reverse=True)

    print(f"  {'Feature A':<35} {'Feature B':<35} {'corr':>8}")
    print("  " + "-" * 85)

    shown = 0

    for f1, f2, corr, abs_corr in pairs:

        if abs_corr < min_corr:
            continue

        print(
            f"  {f1:<35} "
            f"{f2:<35} "
            f"{corr:>+8.4f}"
        )

        shown += 1

        if shown >= top_n:
            break

    if shown == 0:
        print("  No feature pairs above threshold.")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_one(dir_path: str, method: str):
    symbol = os.path.basename(dir_path.rstrip("/")).split("_")[0]

    print(f"\n── {symbol}  ({dir_path}) ──")

    X, y, feature_names = load_dir(dir_path)

    fwd = get_fwd_return(X, y, feature_names)

    feats = extract_features(X, feature_names)

    # ── FEATURE ↔ TARGET CORRELATION ───────────────────────────────
    corr = compute_corr(feats, fwd, method=method)

    print(f"\n  Top 20 features ({method}):\n")
    print(f"  {'Feature':<35} {'corr':>8}  {'p':>8}  {'sig':>5}  {'n':>6}")
    print("  " + "-" * 70)

    for feat, row in corr.head(20).iterrows():
        sig = "✱" if row["significant"] else ""

        print(
            f"  {feat:<35} "
            f"{row['corr']:>+8.4f}  "
            f"{row['p_value']:>8.4f}  "
            f"{sig:>5}  "
            f"{int(row['n']):>6}"
        )

    # ── SAVE FEATURE ↔ TARGET CSV ──────────────────────────────────
    csv_path = f"{symbol}_correlations.csv"

    corr[
        ["corr", "p_value", "significant", "n"]
    ].to_csv(
        csv_path,
        float_format="%.4f"
    )

    print(f"  CSV → {csv_path}")

    # ── FEATURE ↔ TARGET PLOT ──────────────────────────────────────
    png_path = f"{symbol}_correlations.png"

    plot_single(
        corr,
        symbol,
        method,
        png_path
    )

    # ── FEATURE ↔ FEATURE CORRELATION ──────────────────────────────
    feature_corr = compute_feature_corr(
        feats,
        method=method
    )

    # Save matrix CSV
    feature_corr_csv = f"{symbol}_feature_feature_corr.csv"

    feature_corr.to_csv(
        feature_corr_csv,
        float_format="%.4f"
    )

    print(f"  Feature-feature CSV → {feature_corr_csv}")

    # Save matrix heatmap
    feature_corr_png = f"{symbol}_feature_feature_corr.png"

    plot_feature_corr(
        feature_corr,
        symbol,
        method,
        feature_corr_png
    )
    print_top_feature_pairs(
        feature_corr,
        top_n=40,
        min_corr=0.5
    )
    return corr, symbol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  default=None,
                        help="Single symbol dir, e.g. BTCUSDT_7_3_8")
    parser.add_argument("--dirs", nargs="+", default=None,
                        help="Multiple symbol dirs for cross-symbol heatmap")
    parser.add_argument("--method", default="pearson",
                        choices=["pearson", "spearman"])
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else ([args.dir] if args.dir else None)
    if not dirs:
        parser.error("Provide --dir or --dirs")

    all_corr = {}
    for d in dirs:
        corr, symbol = run_one(d, args.method)
        all_corr[symbol] = corr

        # Save CSV
        csv_path = f"{symbol}_correlations.csv"
        corr[["corr", "p_value", "significant", "n"]].to_csv(csv_path, float_format="%.4f")
        print(f"  CSV → {csv_path}")

        # Single-symbol bar chart
        png_path = f"{symbol}_correlations.png"
        plot_single(corr, symbol, args.method, png_path)

    # Multi-symbol heatmap if more than one dir
    if len(all_corr) > 1:
        plot_multi(all_corr, args.method, "all_symbols_correlations.png")


if __name__ == "__main__":
    main()