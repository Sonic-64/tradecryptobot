import math
import time

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import  DataLoader
from itertools import product, combinations
import numpy as np
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .dataset import NumpyDataset
from . import analyze_days
from .trend_predict import get_hmm_signal_from_batch,evaluate_hmm,evaluate_full,train_hmm,test_hmm,plot_hmm_states
from .tune import  save_eval_results, insert_eval_results, load_eval_results,get_best_config
from .model import LSTMModel, CNNModel, WeightedBCELoss, HMMRegime,WeightedBrierLoss
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
        logits = model(xb)

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







# # --- Main function to run the training and evaluation to call from main.py ---



def conf_eval(config,loader,use_cnn=True,use_lstm=True,prop_threshold=0.50,cnn_threshold=0.50,label=""):
    accuracy_confidence = 0.5
    weighted_correct = 0.0
    weighted_total = 0.0
    all_predictions = []
    model = config["lstm"]
    cnn_model = config["cnn"]
    scaler = config["scaler"]
    features = config["features"]
    with torch.no_grad():
        for xb, y_class, y_change in loader:
            xb = xb.to("cpu")

            # Predykcja z obu modeli
            logits = model(xb)  # ignorujemy price z class_model
            prob_up = torch.sigmoid(logits)

            logits_cnn = cnn_model(xb)
            prop_cnn = torch.sigmoid(logits_cnn)

            # Przenieś na CPU
            prob_up = prob_up.cpu().numpy().flatten()
            prop_cnn = prop_cnn.cpu().numpy().flatten()
            # Zapisz wszystko
            for i in range(len(prob_up)):
                all_predictions.append({
                    'prob_up': prob_up[i],
                    'prob_up_cnn': prop_cnn[i],
                    'actual_class': y_class[i],
                    'actual_change':y_change[i],
                })


    trades = []
    correct_trades = []


    for p in all_predictions:
        cnn_long = p['prob_up_cnn']  > cnn_threshold if use_cnn else True
        cnn_short = p['prob_up_cnn']  < (1-cnn_threshold) if use_cnn else True
        lstm_long = p['prob_up']  > prop_threshold if use_lstm else True
        lstm_short = p['prob_up']  < (1-prop_threshold) if use_lstm else True
        predict_long = (lstm_long and cnn_long)

                # Short: oba modele przewidują spadek
        predict_short = (lstm_short and cnn_short )

        if predict_long or predict_short:


            trades.append(p)
            predicted_direction = 1 if predict_long else 0
            actual_direction = 1 if p['actual_class'] > 0 else 0
            is_correct = predicted_direction == actual_direction
            correct_trades.append(predicted_direction == actual_direction)
            w = float(np.clip(abs(p["actual_change"]) * 60.0, 0.20, 3.0))
            weighted_correct += w * int(is_correct)
            weighted_total += w

    result = {
        "label": label,
        "use_lstm": use_lstm,
        "use_cnn": use_cnn,
        "threshold":prop_threshold,
        "cnn_threshold":cnn_threshold,
        "num_trades": len(correct_trades),
        "total_samples": len(all_predictions),
        "coverage_pct": round(len(correct_trades) / len(all_predictions) * 100, 2) if all_predictions else 0,
        "accuracy": None,
    }
    if trades:

        accuracy_confidence = weighted_correct/weighted_total
        result["accuracy"] = round(accuracy_confidence * 100, 2)


    return accuracy_confidence,result
def conf_eval_live(config,xb,use_cnn=True,use_lstm=True,prop_threshold=0.50,cnn_threshold=0.50):
    result = []
    model = config["lstm"]
    cnn_model = config["cnn"]
    scaler = config["scaler"]
    features = config["features"]
    predicted_direction = -1
    max_signal_strenght = 1
    signal_strenght = 0
    with torch.no_grad():

        xb = xb.to("cpu")

            # Predykcja z obu modeli
        logits = model(xb)  # ignorujemy price z class_model
        prob_up = torch.sigmoid(logits)

        logits_cnn = cnn_model(xb)
        prop_cnn = torch.sigmoid(logits_cnn)
            # Przenieś na CPU
        prob_up = prob_up.cpu().numpy().flatten()
        prop_cnn = prop_cnn.cpu().numpy().flatten()
        last_candle = xb[:, -1, :].cpu().numpy()
            # Zapisz wszystko
        for i in range(len(prob_up)):
            result.append({
                'prob_up': prob_up[i],
                'prob_up_cnn': prop_cnn[i],
                'last_candle':last_candle[i],
            })


    for p in result:

        cnn_long = p['prob_up_cnn']  > cnn_threshold if use_cnn else True
        cnn_short = p['prob_up_cnn']   < (1-cnn_threshold) if use_cnn else True
        lstm_long = p['prob_up']   > prop_threshold if use_lstm else True
        lstm_short = p['prob_up']   < (1-prop_threshold) if use_lstm else True
        predict_long = (lstm_long and cnn_long )

        predict_short = (lstm_short and cnn_short)

        if predict_long or predict_short:

            predicted_direction = 1 if predict_long else 0





    return predicted_direction,math.exp(signal_strenght)
def save_split(X, y, directory, name):
    np.save(f"{directory}/{name}_X.npy", X)
    np.save(f"{directory}/{name}_y.npy", y)
    # Podział na train/test

def Train_val(dataset_dir,SEED = 42, EPOCHS=100, BATCH=32, LR=1e-3):
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
    train_end = int(0.750 * N_raw)
    test_end = int(0.900 * N_raw)
    val_end = int(N_raw)
    save_split(X_raw[0:train_end], y_raw[0:train_end], dataset_dir, "train")
    save_split(X_raw[train_end:test_end], y_raw[train_end:test_end], dataset_dir, "test")
    save_split(X_raw[test_end:val_end], y_raw[test_end:val_end], dataset_dir, "val")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy",f"{outdir_path}/train_y.npy",features_path,filter_noise=False)
    test_ds = NumpyDataset(f"{outdir_path}/test_X.npy",f"{outdir_path}/test_y.npy",features_path,filter_noise=False)
    val_ds = NumpyDataset(f"{outdir_path}/val_X.npy",f"{outdir_path}/val_y.npy",features_path,filter_noise=False)
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    # ===== RESHAPE AND SCALE FEATURES =====
    Ntr, T, F = X_train.shape
    Nte = X_test.shape[0]
    Nval = X_val.shape[0]
    # Flatten time steps for scaling
    Xtr_2d = X_train.view(-1, F).numpy()
    Xte_2d = X_test.view(-1, F).numpy()
    Xval_2d = X_val.view(-1,F).numpy()
    # Fit scaler on TRAIN only and transform both train and test
    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(Xtr_2d)  # ✅ Fit on train only
    Xte_2d = scaler.transform(Xte_2d)  # ✅ Transform test
    Xval_2d = scaler.transform(Xval_2d)
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)
    y_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))])



    # ===== ALWAYS-UP BASELINE =====
    always_up_acc = (y_test == 1).float().mean().item()

    print("\n===== BASELINE CHECK =====")
    print(f"Always-UP Accuracy: {always_up_acc:.2%}")
    # Reshape back to original LSTM shape
    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_test = torch.from_numpy(Xte_2d).float().view(Nte, T, F)
    X_val = torch.from_numpy(Xval_2d).float().view(Nval,T,F)
    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.X = X_train
    test_ds.X = X_test
    val_ds.X = X_val

    class_criterion = WeightedBCELoss(
        scale=60.0, min_w=0.20, max_w=3.0
    )
    # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    val_loader = DataLoader(val_ds,batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features
    HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    KERNEL_SIZE = 4
    DROPOUT = 0.3
    DROPOUT_LSTM = 0.3
    print(f"using HIDDEN SIZE:{HIDDEN_SIZE} NUM LAYERS:{NUM_LAYERS} KERNEL_SIZE:{KERNEL_SIZE} DROPOUT:{DROPOUT}")
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

    # Early stopping variables
    best_acc = 0.0
    best_brier = 1
    patience = 40
    patience_counter = 0
    print("Starting training with  class accuracy focus...\n")
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device,weighted=True
        )
        test_acc,brier  = evaluate_weighted(
            model, test_loader, device
        )

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc > best_acc:
            best_brier = brier
            best_acc = test_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}/{SEED}_binary_model.pt")
        else:
            patience_counter += 1


        # Early stopping
        if patience_counter >= patience:

            break

    # Load best model for final save

    print(f"Model saved as {dataset_dir}/{SEED}_binary_model.pt with brier: {best_brier}\n")


    cnn_model = CNNModel(
        input_size=input_size,
        num_filters=HIDDEN_SIZE,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": HIDDEN_SIZE,  # Increased
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
    best_brier = 1
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, class_criterion, cnn_optimizer, device,weighted=True)
        test_acc,brier = evaluate_weighted(cnn_model, test_loader, device)
        cnn_scheduler.step()

        if test_acc>best_acc:
            best_brier = brier
            best_acc = test_acc
            patience_counter = 0
            torch.save(cnn_model.state_dict(), f"{dataset_dir}/{SEED}_cnn_model.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    print(f"Model saved as {dataset_dir}/{SEED}_cnn_model.pt with brier: {best_brier}\n")




    with open(features_path, 'r') as f:
        feature_names = json.load(f)



    cnn_model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_cnn_model.pt"))
    cnn_model.eval()
    model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_binary_model.pt"))
    model.eval()
    weighted_train, train_brier = evaluate_weighted(model, train_loader, device)
    weighted_test, test_brier = evaluate_weighted(model, test_loader, device)
    print(f" LSTM train: {train_brier:.3f}  test: {test_brier:.3f}  brier gap : {train_brier - test_brier:.3f}  weighted LSTM train: {weighted_train:.3f}  test: {weighted_test:.3f}  gap: {weighted_train - weighted_test:.3f}")
    weighted_train, train_brier = evaluate_weighted(cnn_model, train_loader, device)
    weighted_test, test_brier = evaluate_weighted(cnn_model, test_loader,device)
    print(f" CNN train: {train_brier:.3f}  test: {test_brier:.3f}  brier gap: {train_brier - test_brier:.3f}  weighted CNN train: {weighted_train:.3f}  test: {weighted_test:.3f}  gap: {weighted_train - weighted_test:.3f}")
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
    weighted_val,val_brier = evaluate_weighted(cnn_model,val_loader,device)
    print(f"CNN VAL performance weighted: {weighted_val:3f} brier: {val_brier}")
    weighted_val, val_brier = evaluate_weighted(model, val_loader, device)
    print(f"LSTM VAL performance weighted: {weighted_val:3f} brier: {val_brier}")
    symbol = dataset_dir.split("_")[0]
    flags = [True, False]
    thresholds = [0.50,  0.55,  0.60, 0.65]
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


    _,median_change,_,median_dip = analyze_days(symbol=symbol,lookback_days=360)
    print(f"median dip: {median_dip:.2%}")
    print(f"estimated needed accuracy for 100% of dip entry {((median_change - median_dip * 1.0) / (2 * median_change)):.2%}")
    print(f"estimated needed accuracy for 70% of dip entry {((median_change - median_dip*0.7)/(2*median_change)):.2%}")
    print(f"estimated needed accuracy for 50% of dip entry {((median_change - median_dip * 0.5) /(2 * median_change)):.2%}")

    return best_acc,0.0,0.0,0.0
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
def train_for_live(dataset_dir,SEED = 42, EPOCHS=100, BATCH=32, LR=1e-3):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"TRAININ ON {dataset_dir}")
    train_hmm(dataset_dir,live=True)
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
    train_start = int(0.1*N_raw)
    train_end = int((0.75+0.1) * N_raw)
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
    class_criterion = BCEWithLogitsLoss()  # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]
    HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    KERNEL_SIZE = 4
    DROPOUT = 0.3
    DROPOUT_LSTM = 0.3
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
        scale=60.0, min_w=0.20, max_w=3.0
    )

    # Early stopping variables
    best_acc = 0.0
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

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc>best_acc:
            best_brier = brier
            best_acc=test_acc
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
        num_filters=HIDDEN_SIZE,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": HIDDEN_SIZE,  # Increased
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
    best_brier = 1
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, class_criterion, cnn_optimizer, device,weighted=True)
        test_acc,brier = evaluate_weighted(cnn_model, test_loader,  device)
        cnn_scheduler.step()

        if test_acc>best_acc:
            best_acc = test_acc
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




def paper_trade_historical(months, window_days, resample_hours, horizon, cutoff=0, paper_trade_days=30, seed=42,
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
            step=1,
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
        df_feat, _ = compute_features(df_raw, resample_hours)
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
            if i - last_trade_candle[symbol]>= cooldown_candles:

                acc = c.get("accuracy")
                if not acc:
                    continue
                acc = acc[0] if isinstance(acc, list) else acc
                if acc < 0.6:
                    acc = 0.0
                window = feature_array[i - window_size:i]
                X_scaled = scale_live_window(window, m["scaler"])
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




