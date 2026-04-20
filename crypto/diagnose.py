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


def analyze_feature_importance(model, loader, scaler,
                               feature_names, n_batches=20):
    """
    Permutation importance — shuffle each feature and measure
    accuracy drop. Bigger drop = more important feature.
    Works for any model, not just attention models.
    """
    model.eval()

    # baseline accuracy
    baseline_correct = 0
    total = 0
    all_xb = []
    all_y = []

    with torch.no_grad():
        for i, (xb, y_class, _) in enumerate(loader):
            if i >= n_batches:
                break
            all_xb.append(xb)
            all_y.append(y_class)

    all_xb = torch.cat(all_xb)  # (N, T, F)
    all_y = torch.cat(all_y)  # (N, 1)

    with torch.no_grad():
        baseline_preds = (torch.sigmoid(model(all_xb)) > 0.5)
        baseline_acc = (baseline_preds == all_y).float().mean().item()

    print(f"\nBaseline accuracy: {baseline_acc:.3f}")
    print(f"\n{'Feature':<20} {'Acc when shuffled':>18} {'Drop':>8} {'Importance':>12}")
    print('─' * 62)

    importances = {}

    F = all_xb.shape[2]
    for f_idx, f_name in enumerate(feature_names):
        xb_permuted = all_xb.clone()

        # shuffle this feature across the batch dimension
        # keeps time structure intact — only shuffles which
        # sample gets which feature values
        perm = torch.randperm(all_xb.shape[0])
        xb_permuted[:, :, f_idx] = all_xb[perm, :, f_idx]

        with torch.no_grad():
            perm_preds = (torch.sigmoid(model(xb_permuted)) > 0.5)
            perm_acc = (perm_preds == all_y).float().mean().item()

        drop = baseline_acc - perm_acc
        importances[f_name] = drop

        bar = '█' * max(0, int(drop * 200))
        print(f"  {f_name:<18} {perm_acc:>18.3f} {drop:>+8.3f} {bar}")

    # sort by importance
    print(f"\n{'─' * 62}")
    print("Ranked by importance:")
    for name, imp in sorted(importances.items(),
                            key=lambda x: x[1], reverse=True):
        print(f"  {name:<20} {imp:+.4f}")

    return importances


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