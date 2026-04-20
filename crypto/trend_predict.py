import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from .model import HMMRegime
import matplotlib.pyplot as plt
def get_hmm_signal_from_batch(xb, scaler, hmm, feature_names, hmm_weight=0.2):
    xb_np = xb.cpu().numpy()   # (B, T, F)
    B, T, F = xb_np.shape

    signals = np.zeros(B)

    for i in range(B):
        # ✅ use FULL WINDOW (not last step)
        window_scaled = xb_np[i]  # (T, F)

        # unscale
        window_raw = scaler.inverse_transform(window_scaled)

        # build dataframe
        df_win = pd.DataFrame(window_raw, columns=feature_names)

        # 🚨 SAFETY FIX: ensure enough data
        if len(df_win) < 10:
            signals[i] = 0.0
            continue

        try:
            regimes = hmm.transform(df_win)

            bull = regimes["bull_prob"][-1]
            bear = regimes["bear_prob"][-1]
            side = regimes["side_prob"][-1]

            signals[i] = (
                hmm_weight *
                (1 - side) *
                (bull - bear)
            )

        except Exception:
            # 🚨 fallback if HMM breaks
            signals[i] = 0.0

    return signals
def plot_hmm_states(df, hmm, title="HMM Regimes"):
    regimes = hmm.transform(df)
    states = regimes["states"]
    close = df["Close"].values

    plt.figure(figsize=(14, 6))

    for s in np.unique(states):
        mask = states == s
        label = hmm.state_map.get(s, f"state {s}")
        plt.scatter(
            np.arange(len(close))[mask],
            close[mask],
            s=10,
            label=label
        )

    plt.plot(close, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.show()
def evaluate_full(hmm, df_train, df_val):
    X_train = hmm._clean(hmm._features(df_train))

    X_val = hmm._clean(hmm._features(df_val))

    loglik_train = hmm.model.score(X_train)
    loglik_val   = hmm.model.score(X_val)

    persistence = np.mean(np.diag(hmm.model.transmat_))

    print("\n=== HMM DIAGNOSTICS ===")
    print(f"loglik train: {loglik_train:.2f}")
    print(f"loglik val  : {loglik_val:.2f}")
    print(f"gap         : {(loglik_train - loglik_val):.2f}")
    print(f"persistence : {persistence:.3f}")

    return loglik_train, loglik_val, persistence
def evaluate_hmm(hmm, df, name):
    # build features
    X_raw = hmm._clean(hmm._features(df))

    loglik = hmm.model.score(X_raw)
    persistence = np.mean(np.diag(hmm.model.transmat_))
    means = hmm.model.means_   # (states, features)

    print(f"\n=== {name} ===")
    print(f"loglik: {loglik:.2f}")
    print(f"persistence: {persistence:.3f}")

    print("\n📊 State means (per feature):")
    for i, m in enumerate(means):
        print(f"state {i} ({hmm.state_map[i]}): {np.round(m, 6)}")

    # =============================
    # 🔬 Diagnostics (VERY IMPORTANT)
    # =============================

    regimes = hmm.transform(df)
    signal = regimes["bull_prob"] - regimes["bear_prob"]

    close = df["Close"].values

    returns = np.log(close[1:] / (close[:-1] + 1e-10))
    returns = np.pad(returns, (1, 0))

    vol = pd.Series(returns).rolling(10).std().fillna(0).values

    corr_ret = np.corrcoef(signal, returns)[0, 1]
    corr_vol = np.corrcoef(signal, vol)[0, 1]

    print("\n🔬 Regime diagnostics:")
    print(f"corr with returns: {corr_ret:.3f}")
    print(f"corr with vol    : {corr_vol:.3f}")

    return loglik, persistence
def test_hmm(dataset_dir):
    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    X_raw = np.load(x_path)

    df = pd.read_csv(f"{dataset_dir}/df.csv")
    df = df.iloc[-len(X_raw):]

    N = len(df)
    train_end = int(0.8 * N)
    val_end = int(0.94 * N)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]





    # Plot on validation (IMPORTANT)

    # =============================
    # FEATURE SEARCH
    # =============================

    print("\n🔍 Searching best HMM config...")



    hmm = HMMRegime()
    hmm.fit(df_train)





    # =============================
    # FINAL MODEL
    # =============================
    symbol = dataset_dir.split("_")[0]
    hmm = HMMRegime()
    hmm.fit(df_train)
    plot_hmm_states(df_val, hmm, title=f"{symbol} HMM (VAL)")
    print("\n📊 FINAL MODEL:")
    loglik, persistence = evaluate_hmm(hmm, df_val, "FINAL")
    evaluate_full(hmm,df_train, df_val)
    # =============================
    # SAVE
    # =============================



    joblib.dump(hmm, Path(dataset_dir) / f"hmm.pkl")

    meta = {
        "persistence": persistence,
        "loglik": loglik
    }

    with open(Path(dataset_dir) / f"hmm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ HMM saved for {symbol}")

    return hmm
def train_hmm(dataset_dir,live=False):
    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    X_raw = np.load(x_path)

    df = pd.read_csv(f"{dataset_dir}/df.csv")
    df = df.iloc[-len(X_raw):]

    N = len(df)
    if live == False:
        train_end = int(0.75 * N)
        val_end = int(0.9 * N)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
    else:
        train_start = int(N*0.1)
        train_end = int((0.85+0.1) * N)
        val_end = int(N)

        df_train = df.iloc[train_start:train_end]
        df_val = df.iloc[train_end:val_end]

    meta_path = Path(dataset_dir) / "hmm_meta.json"

    if not meta_path.exists():
        raise ValueError("❌ No hmm_meta.json found. Run test_hmm first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)



    # =============================
    # TRAIN FINAL MODEL
    # =============================
    hmm = HMMRegime()
    hmm.fit(df_train)

    loglik, persistence = evaluate_hmm(hmm, df_val, "FINAL")

    # =============================
    # SAVE (overwrite model, keep features)
    # =============================
    joblib.dump(hmm, Path(dataset_dir) / "hmm.pkl")

    # update metrics but KEEP features
    meta["persistence"] = persistence
    meta["loglik"] = loglik

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


    return hmm