import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
from pathlib import Path
import json
import torch
import torch.nn as nn
from hmmlearn.hmm import GaussianHMM
import torch.nn.functional as F


class HMMRegime:
    # fixed feature order — column 0 is ALWAYS r1
    # so means_[:, 0] is safe for state labeling

    def __init__(self, features_config=None):
        # features_config kept for backward compat but ignored
        # FEATURES is now fixed for stability
        self.model = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=300,
            min_covar=0.02,
            random_state=42,
        )
        self.state_map = None

    def _features(self, df):
        """
        FIX 1: use NaN rows instead of zero-padding.
        Zero-padding made HMM think "0 return" was a real regime observation.
        """
        close = df['Close'].values.astype(np.float64)
        distance = df["distance_hl_position"].values.astype(np.float64)
        funding_z = df["funding_z"].values.astype(np.float64)
        volume = df["volume_zscore"].values.astype(np.float64)
        taker = df["taker_buy_ratio"].values.astype(np.float64)
        funding_delta = np.diff(funding_z, prepend=funding_z[0])
        funding_delta = pd.Series(funding_delta).rolling(window=8,min_periods=1).mean().values
        taker_delta = np.diff(taker, prepend=taker[0])
        taker_delta = pd.Series(taker_delta).rolling(8, min_periods=1).mean().values

        N = len(close)
        out = np.full((N,4), np.nan)

        # --- 1. STRONG DIRECTION (core signal)
        r1 = np.log(close[1:] / (close[:-1] + 1e-10))


        out[1:, 0] = r1
        # --- 2. MEDIUM-TERM TREND
        r8 = np.log(close[8:] / (close[:-8] + 1e-10))
        out[8:, 1] = r8

        # --- 3. TREND STRENGTH (not volatility!)

        sign = (np.sign(r8))
        sign_strength = pd.Series(sign).rolling(7, min_periods=1).mean().abs().values
        ext_dist = pd.Series(distance * 2 - 1).rolling(16, min_periods=1).mean().abs().values
        out[8:, 2] = sign_strength
        out[:, 3] = ext_dist




        return out

    def _clean(self, X):
        """Drop NaN rows."""
        mask = ~np.isnan(X).any(axis=1)
        return X[mask]

    def fit(self, df):
        X_raw = self._clean(self._features(df))



        self.model.fit(X_raw)

        # state labeling — identical to original
        # safe because column 0 is always r1 (fixed FEATURES order)
        score = self.model.means_[:, 2] + 0.5 * self.model.means_[:, 3]
        order = np.argsort(score)
        self.state_map = {
            int(order[0]): "sideways",
            int(order[1]): "trending",
        }

    def transform(self, df):
        """Label full sequence — identical output format to original."""
        X_raw = self._clean(self._features(df))


        states = self.model.predict(X_raw)
        probs = self.model.predict_proba(X_raw)

        # pad back to original length with neutral values
        N = len(df)
        pad = N - len(states)
        states = np.pad(states, (pad, 0), constant_values=1)
        probs = np.pad(probs, ((pad, 0), (0, 0)),
                       constant_values=1 / 2)

        n = len(states)
        trend = np.zeros(n)
        side = np.zeros(n)

        for i in range(2):
            label = self.state_map[i]
            if label == "trending":
                trend += probs[:, i]
            elif label == "sideways":
                side += probs[:, i]

        return {
            "trend_prob": trend,
            "side_prob": side,
            "states": states,
        }

    def predict_window(self, df_window):
        """
        Classify regime for a SINGLE window (full sequence).
        Returns dict with scalar values (last timestep only).
        """
        X_raw = self._clean(self._features(df_window))

        if len(X_raw) < 10:
            return {"trend_prob": 1 / 2, "side_prob": 1 / 2}


        probs = self.model.predict_proba(X_raw)  # (T, 3)

        last = probs[-1]  # last timestep probabilities
        trending = sum(last[i] for i, n in self.state_map.items() if n == "trending")
        side = sum(last[i] for i, n in self.state_map.items() if n == "sideways")

        return {"trend_prob": trending, "side_prob": side}

    def save(self, path):
        joblib.dump(self, path)

    def load(cls, path):
        return joblib.load(path)
class WeightedBCELoss(nn.Module):
    """
    BCEWithLogitsLoss weighted by absolute price change magnitude.

    Intuition:
      a 3% move predicted correctly matters more than a 0.5% move
      model should focus capacity on the large moves that drive PnL

    Weight formula:
      weight = clip(abs(price_change) * scale, min_w, max_w)

      scale=20:  0.5% move → weight 0.10 (min)
                 1.2% move → weight 0.24
                 3.0% move → weight 0.60
                 5.0% move → weight 1.00
                 8.0% move → weight 1.60
                 capped at max_w=2.0
    """

    def __init__(
        self,
        scale:  float = 20.0,   # multiplier on raw price change
        min_w:  float = 0.25,   # floor — never fully ignore small moves
        max_w:  float = 2.0,
        pos_weight: float = 1.0,# ceiling — don't let outliers dominate
    ):
        super().__init__()
        self.scale = scale
        self.min_w = min_w
        self.max_w = max_w
        self.pos_weight = pos_weight

    def forward(
        self,
        logits:       torch.Tensor,   # (B, 1) raw logits
        labels:       torch.Tensor,   # (B, 1) binary 0/1
        price_changes: torch.Tensor,  # (B, 1) raw price change ratio
    ) -> torch.Tensor:

        # per-sample BCE (unreduced)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )   # (B, 1)
        class_weights = labels * self.pos_weight + (1 - labels) * 1.0
        # weight = f(|price_change|)
        weights = (price_changes.abs() * self.scale).clamp(
            self.min_w, self.max_w
        )   # (B, 1)

        return (bce * weights*class_weights).mean()
# Version with both improvements but simpler combination

class WeightedBrierLoss(nn.Module):
    """
    Weighted Brier score — squared error weighted by move size.

    Brier:          (p - y)²
    Weighted Brier: w × (p - y)²   where w = f(|price_change|)

    advantages over WeightedBCE:
      - naturally bounded loss (max per sample = 1.0)
      - gentler gradients → more stable training
      - better calibration in practice
      - probabilities stay in useful range (0.4-0.6)

    disadvantages:
      - weaker signal discrimination near extremes
      - BCE better at pushing confident predictions
        further toward 0/1
    """

    def __init__(self, scale=60.0, min_w=0.20, max_w=3.0):
        super().__init__()
        self.scale = scale
        self.min_w = min_w
        self.max_w = max_w

    def forward(self, logits:torch.Tensor, labels:torch.Tensor, price_changes:torch.Tensor):
        # apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)  # (B, 1)

        # squared error per sample
        brier = (probs - labels) ** 2  # (B, 1)

        # weight by move magnitude
        weights = (price_changes.abs() * self.scale).clamp(
            self.min_w, self.max_w
        )  # (B, 1)

        return (brier * weights).mean()
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.norm = nn.LayerNorm(hidden_size * 4)

        self.fc = nn.Linear(hidden_size * 4, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(128, 1)

    def forward(self, x,training=False):
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attn(lstm_out)
        attn_weights = torch.softmax(
            attn_scores / (lstm_out.size(-1) ** 0.5), dim=1
        )

        attn_pool = (lstm_out * attn_weights).sum(dim=1)
        last = lstm_out[:, -1, :]

        combined = torch.cat([attn_pool, last], dim=1)
        combined = self.norm(combined)

        out = self.fc(combined)
        out = self.relu(out)
        out = self.dropout(out)

        return self.out(out)
class CNNModel(nn.Module):
    """
    1-D Temporal CNN for binary direction classification.

    Input:  (B, T, F)  — same window as LSTM, optionally sliced to recent N steps
    Output: logit (B,) — single value per sample; sigmoid → probability of UP

    Architecture:
      input_proj  : Linear F → num_filters   (applied per timestep)
      conv blocks : 3× dilated causal Conv1d with residual skip
                    dilation 1 → 2 → 4, receptive field grows exponentially
      global pool : mean over time → (B, C)
      FC head     : Linear → logit
    """

    def __init__(
        self,
        input_size: int,
        num_filters: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 3,
        dropout: float = 0.5,

    ):
        super().__init__()


        self.input_proj = nn.Linear(input_size, num_filters)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            padding  = (kernel_size - 1) * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        num_filters, num_filters,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        self.trunk = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(num_filters // 2, 1)

    def forward(self, x,training=False):
        # x: (B, T, F)

        # Optionally slice to most recent N steps
        # e.g. cnn_window_steps=28 → last 3.5 days of 3h candles

        x = x[:, -24:, :]   # (B, N, F)

        # Project features: (B, T, F) → (B, T, C)
        x = self.input_proj(x)

        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)

        # Dilated causal conv blocks with residual skip
        for block in self.blocks:
            residual = x
            out      = block(x)
            out      = out[..., :x.size(-1)]   # trim causal padding
            x        = out + residual

        # Global average pool over time → (B, C)
        x = x.mean(dim=-1)

        # FC head → single logit per sample
        x = self.trunk(x)
        return self.classifier(x)  # (B,)


class MoEEnsemble:
    """2 LSTM + 2 CNN experts gated by HMM regime probabilities (trending/sideways)."""

    REGIME_NAMES = ['trending', 'sideways']
    N_EXPERTS    = 2

    def __init__(self, input_size, hidden_size=24,
                 num_filters=24, kernel_size=4,
                 dropout=0.25, device='cpu'):
        self.device      = device
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters

        self.lstms = nn.ModuleList([
            LSTMModel(input_size, hidden_size, dropout=dropout)
            for _ in range(self.N_EXPERTS)
        ]).to(device)

        self.cnns = nn.ModuleList([
            CNNModel(input_size, num_filters, kernel_size,
                     dropout=dropout)
            for _ in range(self.N_EXPERTS)
        ]).to(device)

    def forward_moe(self, xb, regime_probs):
        """
        xb:           (B, T, F)
        regime_probs: (B, 2) — [p_trending, p_sideways]
        returns:      (B, 1) — final P(up)
        """
        xb           = xb.to(self.device)
        regime_probs = regime_probs.to(self.device)

        # (B, 2) — one probability per expert
        lstm_p = torch.stack([
            torch.sigmoid(self.lstms[r](xb)).squeeze(-1)
            for r in range(self.N_EXPERTS)
        ], dim=1)

        cnn_p = torch.stack([
            torch.sigmoid(self.cnns[r](xb)).squeeze(-1)
            for r in range(self.N_EXPERTS)
        ], dim=1)

        expert_p = (lstm_p + cnn_p) / 2.0                               # (B, 2)
        final_p  = (expert_p * regime_probs).sum(dim=1, keepdim=True)   # (B, 1)
        return final_p

    def save(self, dataset_dir, seed=42):
        d = Path(dataset_dir)
        for r, name in enumerate(self.REGIME_NAMES):
            torch.save(self.lstms[r].state_dict(),
                       d / f"{seed}_lstm_{name}.pt")
            torch.save(self.cnns[r].state_dict(),
                       d / f"{seed}_cnn_{name}.pt")
        json.dump({
            "input_size":  self.input_size,
            "hidden_size": self.hidden_size,
            "num_filters": self.num_filters,
        }, open(d / "moe_config.json", "w"))

    @classmethod
    def load(cls, dataset_dir, seed=42, device='cpu'):
        cfg = json.load(open(Path(dataset_dir) / "moe_config.json"))
        moe = cls(cfg["input_size"], cfg["hidden_size"],
                  cfg["num_filters"], device=device)
        for r, name in enumerate(cls.REGIME_NAMES):
            moe.lstms[r].load_state_dict(torch.load(
                Path(dataset_dir) / f"{seed}_lstm_{name}.pt",
                map_location=device))
            moe.cnns[r].load_state_dict(torch.load(
                Path(dataset_dir) / f"{seed}_cnn_{name}.pt",
                map_location=device))
        for m in list(moe.lstms) + list(moe.cnns):
            m.eval()
        return moe

# Test if code runs
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs and targets should be in the format (B, 1)
        # Calculate BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate modulating factor
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Combine all terms
        focal_loss = alpha_t * modulating_factor * bce_loss

        return focal_loss.mean()