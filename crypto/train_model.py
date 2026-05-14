import math
import time
from asyncio.windows_events import INFINITE

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from itertools import product, combinations
import numpy as np
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .dataset import NumpyDataset, ArrayDataset
from . import analyze_days

from .tune import  save_eval_results, insert_eval_results, load_eval_results,get_best_config
from .model import LSTMModel, CNNModel,MLPModel, WeightedBCELoss, WeightedBrierLoss
from .data_fetch import (
    make_dataset,
    get_evaluate_window,
    scale_live_window, get_price_at,download_data, compute_features
)
from .diagnose import calibration_check




def train(model, loader, criterion, optimizer,
          device, clip: float = 1.0,
          weighted: bool = False):
    model.train()
    total_loss = 0.0

    for xb, y_class, y_change in loader:   # ← unpack y_change
        xb, y_class, y_change = (
            xb.to(device),
            y_class.to(device),
            y_change.to(device),
        )
        optimizer.zero_grad()
        logits = model(xb,training=True)

        if weighted:
            loss = criterion(logits, y_class,y_change)
        else:
            loss = criterion(logits, y_class)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)






def evaluate(model, loader, class_criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, y_class, _ in loader:
            xb, y_class = xb.to(device),y_class.to(device)

            logits  = model(xb)


            loss = class_criterion(logits, y_class)
            total_loss += loss.item() * xb.size(0)

            # For accuracy calculation, apply sigmoid to logits
            pred_class = torch.sigmoid(logits)
            correct += ((pred_class > 0.5) == y_class).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return avg_loss, accuracy


def evaluate_weighted(model, loader, device,
                      scale=60.0, min_w=0.20, max_w=3.0):
    """
    Two metrics:
    1. weighted accuracy  — accuracy weighted by move size
                           matches what training optimized
                           use for early stopping
    """
    model.eval()

    weighted_correct = 0.0
    weighted_total = 0.0

    plain_correct = 0
    plain_total = 0

    with torch.no_grad():
        for xb, y_class, y_change in loader:
            xb = xb.to(device)
            y_class = y_class.to(device)
            y_change = y_change.to(device)

            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct = (preds == y_class).float()  # (B, 1)
            brier = (probs-preds)**2
            # weights — same formula as WeightedBCELoss
            weights = (y_change.abs() * scale).clamp(min_w, max_w)
            brier_weighted = (brier * weights).mean()
            # weighted accuracy
            weighted_correct += (correct * weights).sum().item()
            weighted_total += weights.sum().item()

            plain_correct += correct.sum().item()
            plain_total += len(y_class)

    weighted_acc = weighted_correct / (weighted_total + 1e-8)

    return weighted_acc, brier_weighted
def randomize_windows(X: torch.Tensor) -> torch.Tensor:
    X_np = X.detach().cpu().numpy()  # detach from graph, move to CPU first
    X_random = np.random.randn(*X_np.shape).astype(np.float32)

    for f in range(X_np.shape[2]):
        mean = float(X_np[:, :, f].mean())  # explicit float — never a Tensor
        std  = float(X_np[:, :, f].std())
        X_random[:, :, f] = X_random[:, :, f] * std + mean

    return torch.from_numpy(X_random)





def grid_search(model_type,dataset_dir):
    print(f"performing grid search on model: {model_type} and dataset_dir : {dataset_dir}")
    drop = [0.0,0.2,0.5]
    hidden = [64]
    layers = [2,3]
    best_acc = 0
    best_config = None
    for hidden_size, num_layers,dropout in product(hidden,layers,drop):
        acc,gap = train_with_params(dataset_dir, hidden_size, num_layers, dropout, model_type)
        if acc > best_acc:
            best_acc = acc
            best_config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            }
    print("final_best_config")
    print(best_config)

def create_sample_weights(
    future_returns,
    scale=60.0,
    min_w=0.15,
    max_w=3.0,
):
    """
    Larger future moves matter more.
    """

    weights = np.clip(
        np.abs(future_returns) * scale,
        min_w,
        max_w,
    )


    return weights

def train_with_params(dataset_dir,hidden_size,num_layers,dropout,model_type="CNN",SEED = 42, EPOCHS = 120, BATCH = 1024, LR = 1e-4):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"TRAININ ON {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    if not x_path.exists() or not y_path.exists():
        print(f"Dataset not found in {dataset_dir}")
        return

    dataset = NumpyDataset(x_path, y_path, features_path, filter_noise=False)

    if len(dataset) == 0:
        print("No valid training data\n")
        exit(1)
    pos_ratio = (dataset.y_class.sum() / len(dataset)).item()

    N = len(dataset)
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    N_raw = len(X_raw)

    # Define boundaries on the FULL unfiltered data
    train_start = 0
    train_end = int(0.80 * N_raw)
    val_end = int(N_raw)
    save_split(X_raw[train_start:train_end], y_raw[train_start:train_end], dataset_dir, "train")
    save_split(X_raw[train_end:val_end], y_raw[train_end:val_end], dataset_dir, "val")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy", f"{outdir_path}/train_y.npy", features_path,
                            filter_noise=False)
    val_ds = NumpyDataset(f"{outdir_path}/val_X.npy", f"{outdir_path}/val_y.npy", features_path, filter_noise=False)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    # ===== RESHAPE AND SCALE FEATURES =====
    Ntr, T, F = X_train.shape
    Nval = X_val.shape[0]
    # Flatten time steps for scaling
    Xtr_2d = X_train.view(-1, F).numpy()
    Xval_2d = X_val.view(-1, F).numpy()
    # Fit scaler on TRAIN only and transform both train and test
    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(Xtr_2d)  # ✅ Fit on train only
    Xval_2d = scaler.transform(Xval_2d)
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)

    # ===== ALWAYS-UP BASELINE =====

    # Reshape back to original LSTM shape
    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_val = torch.from_numpy(Xval_2d).float().view(Nval, T, F)
    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.X = X_train
    val_ds.X = X_val

    class_criterion = WeightedBCELoss(
        scale=60.0, min_w=0.15, max_w=3.0
    )
    # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features

    # Slightly larger model for better capacity
    if model_type == "LSTM":
        model = LSTMModel(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout

        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if model_type == "CNN":
        model = CNNModel(
            input_size=input_size,
            num_filters=hidden_size,
            kernel_size=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if model_type == "MLP":
        model = MLPModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
     # Add weight decay

    # Learning rate scheduler



    # Early stopping variables
    best_acc = 0.0
    best_brier = 1
    patience = 40
    patience_counter = 0
    best_score = -float("inf")
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device, weighted=True
        )


        train_acc , _ = evaluate_weighted(model,train_loader,device)
        # Step the scheduler
        scheduler.step()
        score = train_acc
        # Early stopping based on accuracy

        if score>best_score:


            best_score = score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}/grid_search_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"patience triggered")
            break
    model.load_state_dict(torch.load(f"{dataset_dir}/grid_search_model.pt"))
    model.eval()
    weighted_test, _ = evaluate_weighted(model, val_loader, device)
    weighted_train, train_brier = evaluate_weighted(model, train_loader, device)
    print(
        f"model type : {model_type} using HIDDEN SIZE:{hidden_size} NUM FILTERS:{hidden_size} NUM LAYERS:{num_layers} KERNEL_SIZE:{num_layers} DROPOUT:{dropout}")
    print(
        f"  weighted  train: {weighted_train:.3f}  test: {weighted_test:.3f}  gap: {weighted_train - weighted_test:.3f}")
    # Load best model for final save
    return best_score,(abs(weighted_train))
# # --- Main function to run the training and evaluation to call from main.py ---



# ── Evaluate MoE ──────────────────────────────────────────────────────────────


def save_split(X, y, directory, name):
    np.save(f"{directory}/{name}_X.npy", X)
    np.save(f"{directory}/{name}_y.npy", y)
    # Podział na train/test

def Train_val(dataset_dir,SEED = 42, EPOCHS=120, BATCH=256, LR=1e-4):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"TRAININ ON {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    if not x_path.exists() or not y_path.exists():
        print(f"Dataset not found in {dataset_dir}")
        return

    dataset = NumpyDataset(x_path, y_path, features_path,filter_noise=False)

    if len(dataset) == 0:
        print("No valid training data\n")
        exit(1)
    pos_ratio = (dataset.y_class.sum() / len(dataset)).item()
    print(f"Label Distribution → Up: {pos_ratio:.2%}, Down: {1 - pos_ratio:.2%} across {len(dataset)} samples ")
    # Calculate class weights for better balancing

    # Split into train/test

    N = len(dataset)
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    N_raw = len(X_raw)

    # Define boundaries on the FULL unfiltered data

    train_start=int(0.10 * N_raw)
    train_end = int(0.80 * N_raw)
    test_start = 0
    test_end = int(0.900 * N_raw)
    val_end = int(N_raw)
    save_split(X_raw[train_start:train_end], y_raw[train_start:train_end], dataset_dir, "train")
    save_split(np.concatenate([X_raw[test_start:train_start], X_raw[train_end:test_end]]), np.concatenate([y_raw[test_start:train_start], y_raw[train_end:test_end]]), dataset_dir, "test")
    save_split(X_raw[test_end:val_end], y_raw[test_end:val_end], dataset_dir, "val")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy",f"{outdir_path}/train_y.npy",features_path,filter_noise=False)
    val_ds = NumpyDataset(f"{outdir_path}/val_X.npy",f"{outdir_path}/val_y.npy",features_path,filter_noise=False)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    # ===== RESHAPE AND SCALE FEATURES =====
    Ntr, T, F = X_train.shape
    Nval = X_val.shape[0]
    # Flatten time steps for scaling
    Xtr_2d = X_train.view(-1, F).numpy()
    Xval_2d = X_val.view(-1,F).numpy()
    # Fit scaler on TRAIN only and transform both train and test
    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(Xtr_2d)  # ✅ Fit on train only
    Xval_2d = scaler.transform(Xval_2d)
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)



    # ===== ALWAYS-UP BASELINE =====

    # Reshape back to original LSTM shape
    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_val = torch.from_numpy(Xval_2d).float().view(Nval,T,F)
    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.X = X_train
    val_ds.X = X_val

    class_criterion = WeightedBCELoss(
        scale=60.0, min_w=0.15, max_w=3.0
    )
    # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features
    HIDDEN_SIZE = 16
    NUM_FILTERS = 16
    NUM_LAYERS = 2
    KERNEL_SIZE = 4
    DROPOUT = 0.3
    DROPOUT_LSTM = 0.5
    print(f"using HIDDEN SIZE:{HIDDEN_SIZE} NUM LAYERS:{NUM_LAYERS} KERNEL_SIZE:{KERNEL_SIZE} DROPOUT:{DROPOUT}")
    # Slightly larger model for better capacity
    model = LSTMModel(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_LSTM
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-2
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,  # Increased
        "num_layers": NUM_LAYERS,  # Increased
        "dropout": DROPOUT_LSTM,  # Increased
    }
    with open(f"{dataset_dir}/model_config.json", "w") as f:
        json.dump(model_config, f)

    # Early stopping variables
    best_acc = 0.0
    best_brier = 1
    patience = 40
    patience_counter = 0
    best_score = 0
    print("Starting training with  class accuracy focus...\n")
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device, weighted=True
        )

        train_acc, _ = evaluate_weighted(model,train_loader,device)
        # Step the scheduler
        scheduler.step()

        score = train_acc
        # Early stopping based on accuracy

        if score>best_score:
            best_score = score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}/{SEED}_binary_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            break

    # Load best model for final save


    cnn_model = CNNModel(
        input_size=input_size,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": NUM_FILTERS,  # Increased
        "kernel_size": KERNEL_SIZE,  # Increased
        "dropout": DROPOUT,  # Increased
    }
    with open(f"{dataset_dir}/cnn_model_config.json", "w") as f:
        json.dump(cnn_model_config, f)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-2)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=EPOCHS)
    patience = 40
    patience_counter = 0
    best_acc = 0
    best_brier = 1
    best_score = 0
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, class_criterion, cnn_optimizer, device, weighted=True)

        train_acc, _ = evaluate_weighted(cnn_model, train_loader, device)
        cnn_scheduler.step()
        score = train_acc
        if score>best_score :

            best_score = score
            patience_counter = 0
            torch.save(cnn_model.state_dict(), f"{dataset_dir}/{SEED}_cnn_model.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break


    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    cnn_model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_cnn_model.pt"))
    cnn_model.eval()
    model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_binary_model.pt"))
    model.eval()
    weighted_train, train_brier = evaluate_weighted(model, train_loader, device)
    weighted_test, test_brier = evaluate_weighted(model, val_loader, device)
    print(
        f" LSTM train: {train_brier:.3f}  test: {test_brier:.3f}  brier gap : {train_brier - test_brier:.3f}  weighted LSTM train: {weighted_train:.3f}  test: {weighted_test:.3f}  gap: {weighted_train - weighted_test:.3f}")
    weighted_train, train_brier = evaluate_weighted(cnn_model, train_loader, device)
    weighted_test, test_brier = evaluate_weighted(cnn_model, val_loader, device)
    print(
        f" CNN train: {train_brier:.3f}  test: {test_brier:.3f}  brier gap: {train_brier - test_brier:.3f}  weighted CNN train: {weighted_train:.3f}  test: {weighted_test:.3f}  gap: {weighted_train - weighted_test:.3f}")
    evalpath = Path(dataset_dir) / "eval_results.json"
    if evalpath.exists() and evalpath.stat().st_size > 0:
        eval_results = load_eval_results(dataset_dir)
    else:
        eval_results = {
            "symbol": dataset_dir,
            "test": [],
            "val": [],
            "metamodel": []

        }

    print(f"Evaluating on VALIDATION DATASET")
    weighted_val, val_brier = evaluate_weighted(cnn_model, val_loader, device)
    print(f"CNN VAL performance weighted: {weighted_val:3f} brier: {val_brier}")
    weighted_val, val_brier = evaluate_weighted(model, val_loader, device)
    print(f"LSTM VAL performance weighted: {weighted_val:3f} brier: {val_brier}")
    symbol = dataset_dir.split("_")[0]
    flags = [True, False]
    thresholds = [0.50, 0.55, 0.60, 0.65]
    config = {}
    config["lstm"] = model
    config["cnn"] = cnn_model
    config["scaler"] = scaler
    config["features"] = feature_names
    for use_lstm, use_cnn in product(flags, flags):
        if not any([use_lstm, use_cnn]):
            continue

        for threshold, cnn_threshold in product(
                thresholds if use_lstm else [0.50],
                thresholds if use_cnn else [0.50],

        ):
            accuracy_confidence, r = conf_eval(
                config, val_loader,

                use_lstm=use_lstm, use_cnn=use_cnn,
                prop_threshold=threshold, cnn_threshold=cnn_threshold
            )
            insert_eval_results(eval_results["val"], r)
    save_eval_results(eval_results, dataset_dir)
    get_best_config(dataset_dir)
    print(f"STATS FOR {symbol}")




    print(f"Classical models saved to {dataset_dir}/")

    return _,_,_,_
def get_current_utc_time():
    """Get current hour and minute in UTC"""
    now_utc = datetime.now(timezone.utc)
    return now_utc.hour, now_utc.minute
def check_if_good_for_prediction(resample_hours=6, max_minutes_after=15):
    """
    Check if current time is good for prediction
    - Must be at resample hour (0, 6, 12, 18 for 6h)
    - Must be within X minutes after candle start
    """
    hour_utc, minute_utc = get_current_utc_time()

    # Check if it's a resample hour
    if hour_utc % resample_hours != 0:
        return False

    # Check if we're too late into the candle
    if minute_utc > max_minutes_after:
        return False

    return True
def train_for_live(dataset_dir,SEED = 42, EPOCHS=120, BATCH=128, LR=1e-3):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"TRAININ ON {dataset_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    if not x_path.exists() or not y_path.exists():
        print(f"Dataset not found in {dataset_dir}")
        return

    dataset = NumpyDataset(x_path, y_path, features_path, filter_noise=False)

    if len(dataset) == 0:
        print("No valid training data\n")
        exit(1)

    N = len(dataset)
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    N_raw = len(X_raw)

    # Define boundaries on the FULL unfiltered data
    train_start = int(0.2*N_raw)
    train_end = int((0.70+0.2) * N_raw)
    test_end = int(N_raw)
    save_split(X_raw[train_start:train_end], y_raw[train_start:train_end], dataset_dir, "train")
    save_split(X_raw[train_end:test_end], y_raw[train_end:test_end], dataset_dir, "test")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy", f"{outdir_path}/train_y.npy", features_path,
                            filter_noise=False)
    test_ds = NumpyDataset(f"{outdir_path}/test_X.npy", f"{outdir_path}/test_y.npy", features_path, filter_noise=False)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    # ===== RESHAPE AND SCALE FEATURES =====
    Ntr, T, F = X_train.shape
    Nte = X_test.shape[0]
    # Flatten time steps for scaling
    Xtr_2d = X_train.view(-1, F).numpy()
    Xte_2d = X_test.view(-1, F).numpy()
    # Fit scaler on TRAIN only and transform both train and test
    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(Xtr_2d)
    Xte_2d = scaler.transform(Xte_2d)
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)
    y_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))])


    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_test = torch.from_numpy(Xte_2d).float().view(Nte, T, F)
    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.X = X_train
    test_ds.X = X_test
    train_weights = np.linspace(1.0, 1.0, len(train_ds))
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(train_weights),
        num_samples=len(train_ds),
        replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]
    HIDDEN_SIZE = 16
    NUM_FILTERS = 16
    NUM_LAYERS = 2
    KERNEL_SIZE = 3
    DROPOUT = 0.5
    DROPOUT_LSTM = 0.5
    # Slightly larger model for better capacity
    model = LSTMModel(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_LSTM
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-3
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,  # Increased
        "num_layers": NUM_LAYERS,  # Increased
        "dropout": DROPOUT_LSTM,  # Increased
    }
    with open(f"{dataset_dir}/model_config.json", "w") as f:
        json.dump(model_config, f)

    class_criterion = WeightedBCELoss(
        scale=60.0, min_w=0.15, max_w=3.0
    )

    # Early stopping variables
    best_acc = 0
    best_score = 0
    best_brier = 1
    patience = 40
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device,weighted=True
        )
        test_acc,brier = evaluate_weighted(
            model, test_loader,  device
        )
        train_acc,_ = evaluate_weighted(model,train_loader,device)
        # Step the scheduler
        scheduler.step()
        score = test_acc - 1.0 * abs(train_acc - test_acc)
        # Early stopping based on accuracy
        if score>best_score:
            best_brier = brier
            best_acc=test_acc
            best_score=score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}/{SEED}_binary_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            break

    # Load best model for final save

    cnn_model = CNNModel(
        input_size=input_size,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": NUM_FILTERS,  # Increased
        "kernel_size": KERNEL_SIZE,  # Increased
        "dropout": DROPOUT,  # Increased
    }
    with open(f"{dataset_dir}/cnn_model_config.json", "w") as f:
        json.dump(cnn_model_config, f)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-3)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=EPOCHS)
    patience = 40
    patience_counter = 0
    best_acc = 0
    best_score = 0
    best_brier = 1
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, class_criterion, cnn_optimizer, device,weighted=True)
        test_acc,brier = evaluate_weighted(cnn_model, test_loader,  device)
        train_acc,_ = evaluate_weighted(cnn_model,train_loader,device)
        cnn_scheduler.step()
        score = test_acc - 1.0 * abs(train_acc - test_acc)
        if score>best_score:
            best_acc = test_acc
            best_score=score
            best_brier = brier
            patience_counter = 0
            torch.save(cnn_model.state_dict(), f"{dataset_dir}/{SEED}_cnn_model.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break






    # Use train+test portion (first 87.5% of data, same split as before)
    split_end = test_end  # already calculated above
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    try:
        close_idx = feature_names.index('Close')
    except ValueError:
        print("Warning: 'Close' not found in features, using index 3 as fallback")
        close_idx = 3

    cnn_model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_cnn_model.pt"))
    cnn_model.eval()
    model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_binary_model.pt"))
    model.eval()


    return best_acc, 0.0, 0.0, 0.0
def eval_live(months,window_days,resample_hours,horizon):
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]
    for symbol in symbols:
        make_dataset(
            symbol=symbol,
            # Defaulting to BTC-USD as per original intent or make it an arg? Adding symbol arg would be good too but sticking to requested ones first.
            months=months,
            window_days=window_days,
            resample_hours=resample_hours,
            horizon=horizon,
            step=1,
            cutoff=0
        )
        time.sleep(1)
    for symbol in symbols:
        train_for_live(dataset_dir=f"{symbol}_{window_days}_{resample_hours}_{horizon}")
    config = {}

    for symbol in symbols:
        dataset_dir = f"{symbol}_{window_days}_{resample_hours}_{horizon}"

        config[symbol] = {
            "eval": get_best_config(dataset_dir, get_acc=False),
        }

    training_time = datetime.now()
    test_live(training_time,config, resample_hours, window_days,horizon)




def paper_trade_historical(months, window_days, resample_hours, horizon,step_hours=1, cutoff=0, paper_trade_days=30, seed=42,
                           cooldown_candles=2):
    """
    Train on data ending `paper_trade_days` ago, then simulate trading on the last `paper_trade_days`.

    Timeline:
    |-------- months of training data --------|-- paper_trade_days --|-- now
                                           cutoff=paper_trade_days  cutoff=0
    """



    symbols = ["BTCUSDT","ETHUSDT", "SOLUSDT", "XRPUSDT", "LTCUSDT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Train on data ending paper_trade_days ago ──────────────────────────
    print(f"Training on data ending {paper_trade_days} days ago...")
    for symbol in symbols:
        make_dataset(
            symbol=symbol,
            months=months,
            window_days=window_days,
            resample_hours=resample_hours,
            horizon=horizon,
            step_hours=step_hours,
            cutoff=(paper_trade_days+cutoff),  # <-- key: training ends paper_trade_days ago
        )
        time.sleep(1)

    for symbol in symbols:
        train_for_live(dataset_dir=f"{symbol}_{window_days}_{resample_hours}_{horizon}")

    # ── 2. Load best configs ───────────────────────────────────────────────────
    config = {}
    symbol_models = {}
    for symbol in symbols:
        dataset_dir = f"{symbol}_{window_days}_{resample_hours}_{horizon}"
        config[symbol] = get_best_config(dataset_dir, get_acc=False)

        # Load models
        with open(f"{dataset_dir}/model_config.json") as f:
            cfg = json.load(f)
        with open(f"{dataset_dir}/cnn_model_config.json") as f:
            cnn_cfg = json.load(f)

        input_size = cfg["input_size"]
        lstm = LSTMModel(input_size=input_size, hidden_size=cfg["hidden_size"],
                                 num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
        lstm.load_state_dict(torch.load(f"{dataset_dir}/{seed}_binary_model.pt", map_location=device))
        lstm.eval()

        cnn = CNNModel(input_size=input_size, num_filters=cnn_cfg["num_filters"],
                       kernel_size=cnn_cfg["kernel_size"], dropout=cnn_cfg["dropout"]).to(device)
        cnn.load_state_dict(torch.load(f"{dataset_dir}/{seed}_cnn_model.pt", map_location=device))
        cnn.eval()
        dataset_dir = Path(dataset_dir)

        features_path = dataset_dir / "features.json"
        with open(features_path, 'r') as f:
            features = json.load(f)

        symbol_models[symbol] = {
            "lstm": lstm, "cnn": cnn,
            "scaler": joblib.load(f"{dataset_dir}/scaler.pkl"),
            "hmm":joblib.load(f"{dataset_dir}/hmm.pkl"),
            "features":features

        }

    # ── 3. Download last (paper_trade_days + window_days) of data for simulation ──

    print(f"Simulating last {paper_trade_days} days...")
    steps_per_day = 24 // resample_hours
    window_size = window_days * steps_per_day
    total_steps_needed = window_size + paper_trade_days * steps_per_day

    symbol_data = {}
    close_idx = 0
    low_idx = 0
    high_idx = 0

    for symbol in symbols:
        df_raw = download_data(symbol, months=6 + window_days // 30, cutoff=cutoff)
        df_feat, feature_names = compute_features(df_raw, resample_hours)
        df_sim = df_feat.iloc[-total_steps_needed:]
        symbol_data[symbol] = (df_sim.values.astype(np.float32), df_sim.index)
        cols = list(df_sim.columns)
        close_idx = cols.index('Close')
        low_idx = cols.index('Low')
        high_idx = cols.index('High')

    balance = 1000.0
    open_trades = {symbol: [] for symbol in symbols}
    last_trade_candle = {symbol: -999 for symbol in symbols}
    fill_stats = {symbol: {"attempts": 0, "filled": 0} for symbol in symbols}
    all_trade_logs = {symbol: {"direction": [], "entry": [], "exit": [], "pnl": [],"raw_pnl": [],
                               "opened": [], "closed": [], "trade_size": []} for symbol in symbols}

    # ── 4. Unified candle walk ─────────────────────────────────────────────────
    min_len = min(len(symbol_data[s][0]) for s in symbols)
    for i in range(window_size, min_len - horizon):
        for symbol in symbols:
            feature_array, index = symbol_data[symbol]
            candle_time = index[i]
            c = config[symbol]
            m = symbol_models[symbol]
            still_open = []
            # Close trade if horizon has passed
            for trade in open_trades[symbol]:
                if i >= trade["close_at"]:
                    exit_time = index[trade["close_at"]] + timedelta(minutes=1)
                    exit_price = get_price_at(symbol, exit_time)
                    if exit_price is None:
                        exit_price = float(feature_array[trade["close_at"], close_idx])

                    if trade["direction"] == "LONG":
                        pnl = (exit_price - trade["entry"]) / trade["entry"] * 100
                    else:
                        pnl = (trade["entry"] - exit_price) / trade["entry"] * 100
                    pnl -= 0.07
                    pnl *= 20
                    pnl = max(pnl, -100)
                    balance += trade["trade_size"] * (pnl / 100)
                    all_trade_logs[symbol]["trade_size"].append(round(trade["trade_size"], 4))
                    all_trade_logs[symbol]["direction"].append(trade["direction"])
                    all_trade_logs[symbol]["entry"].append(round(trade["entry"], 4))
                    all_trade_logs[symbol]["exit"].append(round(exit_price, 4))
                    all_trade_logs[symbol]["pnl"].append(round(pnl, 4))
                    all_trade_logs[symbol]["opened"].append(str(trade["opened"]))
                    all_trade_logs[symbol]["closed"].append(str(index[trade["close_at"]]))

                    print(
                        f"  [{candle_time}] {symbol} CLOSED {trade['direction']} @ {exit_price:.4f} | PnL: {pnl:+.2f}% | Balance: ${balance:.2f} ")
                else:
                    still_open.append(trade)
            open_trades[symbol] = still_open


            # Open trade if none open
            if i - last_trade_candle[symbol]> cooldown_candles:

                acc = c.get("accuracy")
                if not acc:
                    continue

                acc = acc[0] if isinstance(acc, list) else acc
                if acc < 0.6:
                    acc = 0.0
                window = feature_array[i - window_size:i]
                X_scaled = scale_live_window(window, m["scaler"],m["features"])
                prediction,_ = conf_eval_live(
                    m, X_scaled,
                    c["use_cnn"], c["use_lstm"],
                    c["prop_threshold"], c["cnn_threshold"]
                )
                if prediction != -1:
                    fill_stats[symbol]["attempts"] += 1
                    direction = "LONG" if prediction == 1 else "SHORT"
                    candle_open = float(feature_array[i, close_idx])

                    # look at next 4 candles (12h) for fill
                    window_end = min(i + 4, min_len - 1)
                    next_candles = feature_array[i:window_end, :]

                    dip_pct = c["DIP_PCT"] * 0.5

                    if direction == "LONG":
                        entry_price = candle_open * (1 - dip_pct)
                        filled = any(next_candles[j, low_idx] <= entry_price
                                     for j in range(len(next_candles)))
                    else:
                        entry_price = candle_open * (1 + dip_pct)
                        filled = any(next_candles[j, high_idx] >= entry_price
                                     for j in range(len(next_candles)))
                    if not filled:
                        print(f"  [{candle_time}] {symbol} {direction} DIP NOT FILLED @ {entry_price:.4f} — skipped")
                        continue

                    fill_stats[symbol]["filled"] += 1
                    print(f"  [{candle_time}] {symbol} OPENED {direction} @ {entry_price:.4f}")
                    last_trade_candle[symbol] = i
                    open_trades[symbol].append({
                        "direction": direction,
                        "entry": entry_price,
                        "close_at": i + horizon,
                        "opened": candle_time,
                        "trade_size": max(0,balance) * 0.20 * (acc/100),
                    })


    # ── 5. Summaries ──────────────────────────────────────────────────────────
    for symbol in symbols:
        trade_log = all_trade_logs[symbol]
        pnls = trade_log["pnl"]
        sizes = trade_log["trade_size"]
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            trade_log["num_trades"] = len(pnls)
            trade_log["win_rate"] = round(wins / len(pnls) * 100, 2)
            trade_log["avg_pnl"] = round(sum(pnls) / len(pnls), 4)
            trade_log["total_profit"] = round(balance - 1000.0, 2)
            trade_log["roi"] = round((balance - 1000.0) / 1000.0, 2)
            trade_log["final_balance"] = round(balance, 2)
            print(f"\n  {symbol} Summary:")
            print(f"  Trades: {len(pnls)} | Win rate: {trade_log['win_rate']}% | Avg PnL: {trade_log['avg_pnl']}%")
            print(f"  ROI: {trade_log['roi']*100}% | Final balance: ${balance:.2f}")
            print("\n─── DIP FILL STATISTICS ───")

            attempts = fill_stats[symbol]["attempts"]
            filled = fill_stats[symbol]["filled"]
            rate = filled / attempts * 100 if attempts > 0 else 0
            print(f"  {symbol}: {filled}/{attempts} filled ({rate:.1f}%)")
        else:
            print(f"  No trades taken for {symbol}")

    with open(f"{cutoff}_paper_trade_historical.json", "w") as f:
        json.dump(all_trade_logs, f, indent=2)
    print("\nSaved to paper_trade_historical.json")
    max_drawdown = min(1000,balance)/1000
    roi = round((balance - 1000.0) / 1000.0, 2)
    return (1-max_drawdown),roi


def test_live(training_time,config,resample_hours,window_days,horizon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]

    expiry = training_time + timedelta(days=30)
    symbol_models = {}
    # Convert to numpy arrays
    for s in symbols:
        dataset_dir = f"{s}_{window_days}_{resample_hours}_{horizon}"
        with open(f"{dataset_dir}/model_config.json") as f:
            cfg = json.load(f)

        input_size = cfg["input_size"]

        lstm = LSTMModel(input_size=input_size, hidden_size=cfg["hidden_size"],
                                 num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
        lstm.load_state_dict(torch.load(f"{dataset_dir}/42_binary_model.pt", map_location=device))
        lstm.eval()
        with open(f"{dataset_dir}/cnn_model_config.json") as f:
            cnn_cfg = json.load(f)
        cnn = CNNModel(input_size=input_size,num_filters=cnn_cfg["num_filters"],kernel_size=cnn_cfg["kernel_size"],dropout=cnn_cfg["dropout"]).to(device)
        cnn.load_state_dict(torch.load(f"{dataset_dir}/42_cnn_model.pt", map_location=device))
        cnn.eval()
        dataset_dir = Path(dataset_dir)

        features_path = dataset_dir / "features.json"
        with open(features_path, 'r') as f:
            features = json.load(f)
        symbol_models[s] = {
            "lstm": lstm,
            "cnn": cnn,
            "scaler": joblib.load(f"{dataset_dir}/scaler.pkl"),
            "hmm":joblib.load(f"{dataset_dir}/hmm.pkl"),
            "features": features
        }


    while True:
        if datetime.now() > expiry:
            break
        if check_if_good_for_prediction(resample_hours=resample_hours,max_minutes_after=1):

                for s in symbols:
                    c = config[s]["eval"]
                    m = symbol_models[s]
                    X = get_evaluate_window(s,window_days,resample_hours)
                    X = scale_live_window(X,m["scaler"])
                    X = torch.tensor(X).to(device)

                    prediction,_ = conf_eval_live(
                        m, X,
                        c["use_cnn"], c["use_lstm"],
                        c["prop_threshold"], c["cnn_threshold"]
                    )
                    if prediction != -1:
                        if prediction == 0:
                            trade = "SHORT"
                        else:
                            trade = "LONG"
                        accuracy = c["accuracy"]
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"[{current_time}] Go {trade} on {s} estimated model accuracy {accuracy:.2%}")



                time.sleep(int(3500*resample_hours))
    return




