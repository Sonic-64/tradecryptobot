import numpy as np
import torch

def calibration_check(model, loader, device, n_bins=10):
    all_probs, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for xb, y_class, y_change in loader:
            prob = torch.sigmoid(model(xb.to(device)))
            all_probs.extend(prob.cpu().numpy().flatten())
            all_labels.extend(y_class.numpy().flatten())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    # --- Brier score ---
    brier = np.mean((probs - labels)**2)
    print(f"\nBrier score: {brier:.5f}\n")

    print(f"{'Bin range':>15} {'Count':>8} {'Mean prob':>10} {'True rate':>10} {'Gap':>8}")

    ece = 0.0
    N = len(probs)

    bin_data = []  # useful for plotting later

    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins

        # include 1.0 in last bin
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi) & (1-probs>=lo)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = mask.sum()
        if count < 10:
            continue

        mean_p = probs[mask].mean()
        true_r = labels[mask].mean()
        gap = mean_p - true_r

        ece += (count / N) * abs(gap)

        print(f"{lo:.2f}–{hi:.2f} {count:>10} {mean_p:>10.3f} {true_r:>10.3f} {gap:>+8.3f}")

        bin_data.append({
            "range": (lo, hi),
            "count": int(count),
            "mean_prob": float(mean_p),
            "true_rate": float(true_r),
            "gap": float(gap),
        })

    print(f"Brier:{brier}")
    return {
        "brier": float(brier),
        "bins": bin_data
    }
def get_attention_weights(model, xb):
    """
    Extract attention weights from LSTMModel.
    Shows which timesteps the model focused on.
    xb: (1, T, F) scaled tensor
    """
    model.eval()
    with torch.no_grad():
        lstm_out, _ = model.lstm(xb)  # (1, T, hidden*2)

        attn_scores = model.attn(lstm_out)  # (1, T, 1)
        attn_weights = torch.softmax(
            attn_scores / (lstm_out.size(-1) ** 0.5), dim=1
        )  # (1, T, 1)

    return attn_weights.squeeze().cpu().numpy()  # (T,)





def plot_attention_over_time(model, xb, feature_names,
                             resample_hours=3):
    """
    Visualize which candles the LSTM focused on.
    """
    weights = get_attention_weights(model, xb)
    T = len(weights)
    hours_ago = [(T - i - 1) * resample_hours for i in range(T)]

    print("\nAttention over time (which candles mattered most):")
    print(f"{'Hours ago':>10} {'Attention':>12} {'Bar'}")
    print('─' * 50)

    # show top 15 most attended candles
    top_idx = np.argsort(weights)[::-1][:15]
    for idx in sorted(top_idx):
        h = hours_ago[idx]
        w = weights[idx]
        bar = '█' * int(w * 500)
        print(f"  {h:>8}h   {w:>10.4f}   {bar}")