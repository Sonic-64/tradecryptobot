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
import torch.nn.functional as F

# Version with both improvements but simpler combination
class ImprovedLSTMModel(nn.Module):
    """
    Simple enhanced LSTM with both improvements but cleaner architecture
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        # 1. LSTM (keep as is)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

        self.temp_conv = nn.Conv1d(
            in_channels=hidden_size * 2,  # From bidirectional LSTM
            out_channels=hidden_size,
            kernel_size=3,  # Better for crypto patterns
            padding=1
        )

        # 4. FIXED: hidden_size * 4 (not *3)
        # lstm_last: hidden*2, conv_last: hidden, avg_pool: hidden = hidden*4
        self.combine_layer = nn.Linear(hidden_size * 4, 128)

        # 5. Processing in correct order
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 6. Output heads
        self.class_head = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # ----- Feature Gating (FIXED) -----
        # Give EACH feature its own importance weight
        # x shape: [batch, seq_len, features]
        feature_importance = self.feature_gate(x)  # [batch, seq_len, features]
        weighted_input = x * feature_importance  # [batch, seq_len, features]

        # ----- LSTM Processing -----
        lstm_out, _ = self.lstm(weighted_input)  # [batch, seq_len, hidden*2]

        # ----- Temporal Convolution -----
        conv_input = lstm_out.permute(0, 2, 1)  # [batch, hidden*2, seq_len]
        conv_out = self.temp_conv(conv_input)  # [batch, hidden, seq_len]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, seq_len, hidden]

        # ----- Combine Features (FIXED dimensions) -----
        lstm_last = lstm_out[:, -1, :]  # [batch, hidden*2]
        conv_last = conv_out[:, -1, :]  # [batch, hidden]
        avg_pool = conv_out.mean(dim=1)  # [batch, hidden]

        # CORRECT: hidden*2 + hidden + hidden = hidden*4
        combined = torch.cat([lstm_last, conv_last, avg_pool], dim=1)

        # ----- Final Processing -----
        out = self.combine_layer(combined)  # [batch, 128]
        out = self.relu(out)
        out = self.dropout(out)

        return self.class_head(out)

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

    def forward(self, x):
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

    def forward(self, x):
        # x: (B, T, F)

        # Optionally slice to most recent N steps
        # e.g. cnn_window_steps=28 → last 7 days of 3h candles

        x = x[:, -28:, :]   # (B, N, F)

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



class CryptoEnsemble:
    """
    Load multiple trained models and make ensemble predictions
    """

    def __init__(self, dataset_dir,model_type="binary_model", device='cpu'):
        self.device = device
        self.models = []
        self.price_models = []
        self.scalers = []
        self.weights = []

        # Load ensemble info
        ensemble_path = Path(dataset_dir) / "ensemble_info.json"
        if not ensemble_path.exists():
            raise FileNotFoundError(f"No ensemble info found at {ensemble_path}")

        with open(ensemble_path, 'r') as f:
            self.info = json.load(f)

        # Load model config
        config_path = Path(dataset_dir) / "model_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load each model
        for i, seed in enumerate(self.info['seeds']):
            # Load scaler
            scaler_path = Path(dataset_dir) / "scaler.pkl"
            scaler = joblib.load(scaler_path)
            self.scalers.append(scaler)

            # Load model
            model = ImprovedLSTMModel(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(device)

            model_path = Path(dataset_dir) / f"{seed}_{model_type}.pt"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            self.models.append(model)

        # Get weights (use validation accuracy)

    def predict(self, X_raw):
        """
        X_raw: numpy array of shape (T, F) - raw unscaled window
        Returns: ensemble probability and predicted change
        """
        all_probs = []
        all_changes = []

        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Scale using this model's scaler
            T, F = X_raw.shape
            X_scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(1, T, F)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

            # Predict
            with torch.no_grad():
                logits = model(X_tensor)
                prob = torch.sigmoid(logits).item()

            all_probs.append(prob)

        # Weighted average
        ensemble_prob = np.average(all_probs, weights=self.weights)

        # Calculate agreement (standard deviation of predictions) # Normalized 0-1

        return ensemble_prob,

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