import time

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import joblib

from torch.utils.data import Dataset, DataLoader
from itertools import product
from torch.utils.data.dataset import random_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss,roc_auc_score
import numpy as np
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .tune import  save_eval_results, insert_eval_results, load_eval_results,get_best_results
from .model import ImprovedLSTMModel,CNNModel,FocalLoss
from .data_fetch import (
    make_dataset,
    get_evaluate_window,
    scale_live_window, get_price_at
)

class NumpyDataset(Dataset):
    def __init__(self, X_path, y_path, features_path,filter_noise=True,min_move = 0.004):
        self.X = torch.FloatTensor(np.load(X_path))
        self.y_prices = torch.FloatTensor(np.load(y_path))

        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

        try:
            self.close_idx = self.feature_names.index('Close')
        except ValueError:
            print("Warning: 'Close' not found in features, using index 3 as fallback")
            self.close_idx = 3

        # Generate binary labels (1 if future price > current price)
        # X shape: (N, T, F)
        # current_price is the Close price at the last time step of the window

        current_prices = self.X[:, -1, self.close_idx]

        price_diff = (self.y_prices - current_prices) / (current_prices + 1e-8)
        if filter_noise:
            valid_mask = price_diff.abs() > min_move

            self.X = self.X[valid_mask]
            self.y_prices = self.y_prices[valid_mask]
            price_diff = price_diff[valid_mask]
        self.y_class = (price_diff > 0).float()
        # Calculate price change ratio for regression target
        # Avoid division by zero
        safe_current_prices = torch.where(current_prices == 0, torch.ones_like(current_prices), current_prices)
        self.y_change =  price_diff



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx].unsqueeze(0), self.y_change[idx].unsqueeze(0)
def train(
        model,
        loader,
        class_criterion,
        optimizer,
        device,
):
    model.train()
    total_loss = 0.0
    for xb, y_class, _ in loader:
        xb, y_class = xb.to(device), y_class.to(device)
        optimizer.zero_grad()
        logits = model(xb)

        # Use raw logits for Focal Loss (it applies sigmoid internally)
        loss_class = class_criterion(logits, y_class)
        loss = loss_class

        loss.backward()
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









# # --- Main function to run the training and evaluation to call from main.py ---



def conf_eval(model,cnn_model,lr_model,loader,use_cnn=True,use_lstm=True,use_lr=True,prop_threshold=0.50,cnn_threshold=0.50,lr_threshold=0.50,label=""):
    accuracy_confidence = 0.5
    all_predictions = []
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
            X_np = xb[:, -16:, :].numpy().reshape(len(xb), -1)
            prop_lr = lr_model.predict_proba(X_np)[:, 1]
            y_class = y_class.cpu().numpy().flatten()
            last_candle = xb[:, -1, :].cpu().numpy()
            # Zapisz wszystko
            for i in range(len(prob_up)):
                all_predictions.append({
                    'prob_up': prob_up[i],
                    'prob_up_cnn': prop_cnn[i],
                    'prop_up_lr':prop_lr[i],
                    'actual_class': y_class[i],
                    'actual_change':y_change[i],
                    'last_candle':last_candle[i]
                })


    trades = []
    correct_trades = []

    changes = []

    for p in all_predictions:
        cnn_long = p['prob_up_cnn'] > cnn_threshold if use_cnn else True
        cnn_short = p['prob_up_cnn'] < (1-cnn_threshold) if use_cnn else True
        lstm_long = p['prob_up'] > prop_threshold if use_lstm else True
        lstm_short = p['prob_up'] < (1-prop_threshold) if use_lstm else True
        lr_long = p['prop_up_lr'] > lr_threshold if use_lr else True
        lr_short = p['prop_up_lr'] < (1-lr_threshold) if use_lr else True
        predict_long = (lstm_long and cnn_long and lr_long)

                # Short: oba modele przewidują spadek
        predict_short = (lstm_short and cnn_short and lr_short)

        if predict_long or predict_short:

            changes.append(abs(p['actual_change']))
            trades.append(p)
            predicted_direction = 1 if predict_long else 0
            actual_direction = 1 if p['actual_class'] > 0 else 0
            correct_trades.append(predicted_direction == actual_direction)

    result = {
        "label": label,
        "use_lstm": use_lstm,
        "use_cnn": use_cnn,
        "use_lr":use_lr,
        "threshold":prop_threshold,
        "cnn_threshold":cnn_threshold,
        "lr_threshold":lr_threshold,
        "num_trades": len(correct_trades),
        "total_samples": len(all_predictions),
        "coverage_pct": round(len(correct_trades) / len(all_predictions) * 100, 2) if all_predictions else 0,
        "accuracy": None,
        "mean_change": None,
        "median_change": None,
    }
    if trades:

        accuracy_confidence = sum(correct_trades)/len(correct_trades)
        result["accuracy"] = round(accuracy_confidence * 100, 2)
        result["mean_change"] = round(float(np.mean(changes)) * 100, 4)
        result["median_change"] = round(float(np.median(changes)) * 100, 4)

    return accuracy_confidence,result
def conf_eval_live(model,cnn_model,lr_model,xb,use_cnn=True,use_lstm=True,use_lr=True,prop_threshold=0.50,cnn_threshold=0.50,lr_threshold=0.50):
    result = []
    predicted_direction = -1
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
        X_np = xb[:, -16:, :].numpy().reshape(len(xb), -1)
        prop_lr = lr_model.predict_proba(X_np)[:, 1]
        last_candle = xb[:, -1, :].cpu().numpy()
            # Zapisz wszystko
        for i in range(len(prob_up)):
            result.append({
                'prob_up': prob_up[i],
                'prob_up_cnn': prop_cnn[i],
                'prop_up_lr':prop_lr[i],
                'last_candle':last_candle[i]
            })


    for p in result:
        cnn_long = p['prob_up_cnn'] > cnn_threshold if use_cnn else True
        cnn_short = p['prob_up_cnn'] < (1-cnn_threshold) if use_cnn else True
        lstm_long = p['prob_up'] > prop_threshold if use_lstm else True
        lstm_short = p['prob_up'] < (1-prop_threshold) if use_lstm else True
        lr_long = p['prop_up_lr'] > lr_threshold if use_lr else True
        lr_short = p['prop_up_lr'] < (1-lr_threshold) if use_lr else True
        predict_long = (lstm_long and cnn_long and lr_long)

        predict_short = (lstm_short and cnn_short and lr_short)

        if predict_long or predict_short:
            predicted_direction = 1 if predict_long else 0





    return predicted_direction
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
    train_end = int(0.70 * N_raw)
    test_end = int(0.875 * N_raw)
    val_end = int(N_raw*0.935 )
    save_split(X_raw[0:train_end], y_raw[0:train_end], dataset_dir, "train")
    save_split(X_raw[train_end:test_end], y_raw[train_end:test_end], dataset_dir, "test")
    save_split(X_raw[test_end:val_end], y_raw[test_end:val_end], dataset_dir, "val")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy",f"{outdir_path}/train_y.npy",features_path,filter_noise=True)
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

    up_ratio = y_test.mean().item()


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
    alpha=1-up_ratio
    class_criterion = FocalLoss(alpha=alpha,gamma=1.0)  # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    val_loader = DataLoader(val_ds,batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    DROPOUT = 0.5
    # Slightly larger model for better capacity
    model = ImprovedLSTMModel(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-4
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,  # Increased
        "num_layers": NUM_LAYERS,  # Increased
        "dropout": DROPOUT,  # Increased
    }
    with open(f"{dataset_dir}/model_config.json", "w") as f:
        json.dump(model_config, f)

    # Early stopping variables
    best_acc = 0.0
    patience = 50
    patience_counter = 0
    best_brier = 1
    print("Starting training with  class accuracy focus...\n")
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device,
        )
        _,test_acc  = evaluate(
            model, test_loader, class_criterion, device
        )

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc > best_acc:
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

    print(f"Model saved as {dataset_dir}/{SEED}_binary_model.pt with accuracy: {best_acc:.2%}\n")

    best_mse = 64.0
    patience = 20
    patience_counter = 0
    cnn_model = CNNModel(
        input_size=input_size,
        num_filters=32,
        kernel_size=4,
        dropout=0.5,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": 32,  # Increased
        "kernel_size": 4,  # Increased
        "dropout": 0.5,  # Increased
    }
    with open(f"{dataset_dir}/cnn_model_config.json", "w") as f:
        json.dump(cnn_model_config, f)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-3)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=EPOCHS)
    cnn_criterion = nn.BCEWithLogitsLoss()
    patience = 50
    patience_counter = 0
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, cnn_criterion, cnn_optimizer, device)
        _, test_acc = evaluate(cnn_model, test_loader, cnn_criterion, device)
        cnn_scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(cnn_model.state_dict(), f"{dataset_dir}/{SEED}_cnn_model.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    print(f"Model saved as {dataset_dir}/{SEED}_cnn_model.pt with accuracy: {best_acc:.2%}\n")

    X_raw = np.load(x_path)
    y_raw = np.load(y_path)

    # Use train+test portion (first 87.5% of data, same split as before)
    split_end = test_end  # already calculated above
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    try:
        close_idx = feature_names.index('Close')
    except ValueError:
        print("Warning: 'Close' not found in features, using index 3 as fallback")
        close_idx = 3
    X_lr = X_raw[:split_end, -16:, :].reshape(split_end, -1)
    Nx = len(X_lr)
    X_lr = scaler.transform(X_lr.reshape(-1, F)).reshape(Nx, 16, F)
    X_lr = X_lr.reshape(Nx, -1)
    y_lr = (y_raw[:split_end] > X_raw[:split_end, -1, close_idx]).astype(int)  # same binary label logic

    lr_model =LogisticRegression(C=0.1, max_iter=1000,class_weight="balanced")
    lr_model.fit(X_lr, y_lr)
    lr_val_acc = lr_model.score(X_lr, y_lr)
    print(f"LR accuracy at 0.5 threshold: {lr_val_acc:.2%}")
    joblib.dump(lr_model, f"{dataset_dir}/{SEED}_lr_model.pkl")

    cnn_model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_cnn_model.pt"))
    cnn_model.eval()
    model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_binary_model.pt"))
    model.eval()

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
    flags = [True, False]
    thresholds = [0.50, 0.55, 0.60, 0.65]

    for use_lstm, use_lr,use_cnn in product(flags, flags, flags):
        if not any([use_lstm, use_lr,use_cnn]):
            continue

        for threshold, lr_threshold ,cnn_threshold in product(
                thresholds if use_lstm else [0.50],
                thresholds if use_lr else [0.50],
                thresholds if use_cnn else [0.50],

        ):
            accuracy_confidence, r = conf_eval(
                model, cnn_model, lr_model,val_loader,

                use_lstm=use_lstm, use_cnn=use_cnn, use_lr=use_lr,
                prop_threshold=threshold,cnn_threshold=cnn_threshold,lr_threshold=lr_threshold
            )
            insert_eval_results(eval_results["val"], r)

    get_best_results(eval_results["val"])
    all_changes = dataset.y_change.numpy()
    avg_change = np.mean(np.abs(all_changes))
    print(f"average  change: {avg_change:.4%}")

    save_eval_results(eval_results, dataset_dir)
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

    # Calculate class weights for better balancing

    # Split into train/test

    N = len(dataset)
    X_raw = np.load(x_path)
    y_raw = np.load(y_path)
    N_raw = len(X_raw)

    # Define boundaries on the FULL unfiltered data
    train_start = int(0.125*N_raw)
    train_end = int((0.70+0.125) * N_raw)
    test_end = int(N_raw)
    save_split(X_raw[train_start:train_end], y_raw[train_start:train_end], dataset_dir, "train")
    save_split(X_raw[train_end:test_end], y_raw[train_end:test_end], dataset_dir, "test")

    train_ds = NumpyDataset(f"{outdir_path}/train_X.npy", f"{outdir_path}/train_y.npy", features_path,
                            filter_noise=True)
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
    Xtr_2d = scaler.fit_transform(Xtr_2d)  # ✅ Fit on train only
    Xte_2d = scaler.transform(Xte_2d)  # ✅ Transform test
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)
    y_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))])

    up_ratio = y_test.mean().item()

    # ===== ALWAYS-UP BASELINE =====
    always_up_acc = (y_test == 1).float().mean().item()

    # Reshape back to original LSTM shape
    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_test = torch.from_numpy(Xte_2d).float().view(Nte, T, F)
    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.X = X_train
    test_ds.X = X_test
    alpha = 1 - up_ratio
    class_criterion = FocalLoss(alpha=alpha, gamma=1.0)  # Increased gamma
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    DROPOUT = 0.5
    # Slightly larger model for better capacity
    model = ImprovedLSTMModel(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-4
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,  # Increased
        "num_layers": NUM_LAYERS,  # Increased
        "dropout": DROPOUT,  # Increased
    }
    with open(f"{dataset_dir}/model_config.json", "w") as f:
        json.dump(model_config, f)

    # Early stopping variables
    best_acc = 0.0
    patience = 20
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            optimizer,
            device,
        )
        _, test_acc = evaluate(
            model, test_loader, class_criterion, device
        )

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc > best_acc:
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

    cnn_model = CNNModel(
        input_size=input_size,
        num_filters=32,
        kernel_size=4,
        dropout=0.5,
    ).to(device)
    cnn_model_config = {
        "input_size": input_size,
        "num_filters": 32,  # Increased
        "kernel_size": 4,  # Increased
        "dropout": 0.5,  # Increased
    }
    with open(f"{dataset_dir}/cnn_model_config.json", "w") as f:
        json.dump(cnn_model_config, f)
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-3)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=EPOCHS)
    cnn_criterion = nn.BCEWithLogitsLoss()
    patience = 50
    patience_counter = 0
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train(cnn_model, train_loader, cnn_criterion, cnn_optimizer, device)
        _, test_acc = evaluate(cnn_model, test_loader, cnn_criterion, device)
        cnn_scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(cnn_model.state_dict(), f"{dataset_dir}/{SEED}_cnn_model.pt")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break




    X_raw = np.load(x_path)
    y_raw = np.load(y_path)

    # Use train+test portion (first 87.5% of data, same split as before)
    split_end = test_end  # already calculated above
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    try:
        close_idx = feature_names.index('Close')
    except ValueError:
        print("Warning: 'Close' not found in features, using index 3 as fallback")
        close_idx = 3
    X_lr = X_raw[:split_end, -16:, :].reshape(split_end, -1)
    Nx = len(X_lr)
    X_lr = scaler.transform(X_lr.reshape(-1, F)).reshape(Nx, 16, F)
    X_lr = X_lr.reshape(Nx, -1)
    y_lr = (y_raw[:split_end] > X_raw[:split_end, -1, close_idx]).astype(int)  # same binary label logic

    lr_model = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    lr_model.fit(X_lr, y_lr)
    joblib.dump(lr_model, f"{dataset_dir}/{SEED}_lr_model.pkl")
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
        eval_results = load_eval_results(dataset_dir)

        config[symbol] = {
            "eval": get_best_results(eval_results["val"], get_acc=False),
        }

    training_time = datetime.now()
    test_live(training_time,config, resample_hours, window_days,horizon)




def paper_trade_historical(months, window_days, resample_hours, horizon,cutoff=0, paper_trade_days=30, seed=42):
    """
    Train on data ending `paper_trade_days` ago, then simulate trading on the last `paper_trade_days`.

    Timeline:
    |-------- months of training data --------|-- paper_trade_days --|-- now
                                           cutoff=paper_trade_days  cutoff=0
    """
    from .data_fetch import download_data, compute_features, build_windows, scale_live_window
    from .tune import load_eval_results, get_best_results
    import torch, joblib, json, numpy as np
    from pathlib import Path
    from datetime import datetime

    symbols = ["BTCUSDT","ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]
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
        eval_results = load_eval_results(dataset_dir)
        config[symbol] = get_best_results(eval_results["val"], get_acc=False)

        # Load models
        with open(f"{dataset_dir}/model_config.json") as f:
            cfg = json.load(f)
        with open(f"{dataset_dir}/cnn_model_config.json") as f:
            cnn_cfg = json.load(f)

        input_size = cfg["input_size"]
        lstm = ImprovedLSTMModel(input_size=input_size, hidden_size=cfg["hidden_size"],
                                 num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
        lstm.load_state_dict(torch.load(f"{dataset_dir}/{seed}_binary_model.pt", map_location=device))
        lstm.eval()

        cnn = CNNModel(input_size=input_size, num_filters=cnn_cfg["num_filters"],
                       kernel_size=cnn_cfg["kernel_size"], dropout=cnn_cfg["dropout"]).to(device)
        cnn.load_state_dict(torch.load(f"{dataset_dir}/{seed}_cnn_model.pt", map_location=device))
        cnn.eval()



        symbol_models[symbol] = {
            "lstm": lstm, "cnn": cnn,
            "lr": joblib.load(f"{dataset_dir}/{seed}_lr_model.pkl"),
            "scaler": joblib.load(f"{dataset_dir}/scaler.pkl"),
        }

    # ── 3. Download last (paper_trade_days + window_days) of data for simulation ──

    print(f"Simulating last {paper_trade_days} days...")
    steps_per_day = 24 // resample_hours
    window_size = window_days * steps_per_day
    total_steps_needed = window_size + paper_trade_days * steps_per_day

    symbol_data = {}
    symbol_close_idx = {}
    symbol_vol_threshold = {}
    for symbol in symbols:
        df_raw = download_data(symbol, months=6 + window_days // 30, cutoff=cutoff)
        df_feat, _ = compute_features(df_raw, resample_hours)
        df_baseline = df_feat.iloc[:-total_steps_needed]
        df_sim = df_feat.iloc[-total_steps_needed:]
        symbol_data[symbol] = (df_sim.values.astype(np.float32), df_sim.index)
        symbol_close_idx[symbol] = list(df_sim.columns).index('Close')
        close_idx = symbol_close_idx[symbol]
        baseline_closes = df_baseline[df_baseline.columns[close_idx]].values
        baseline_returns = np.diff(baseline_closes) / baseline_closes[:-1]
        baseline_vol = np.std(baseline_returns) * 100
        symbol_vol_threshold[symbol] = baseline_vol * 2.0

    balance = 1000.0
    open_trades = {symbol: None for symbol in symbols}
    all_trade_logs = {symbol: {"direction": [], "entry": [], "exit": [], "pnl": [],
                               "opened": [], "closed": [], "trade_size": []} for symbol in symbols}

    # ── 4. Unified candle walk ─────────────────────────────────────────────────
    min_len = min(len(symbol_data[s][0]) for s in symbols)
    for i in range(window_size, min_len - horizon):
        for symbol in symbols:
            feature_array, index = symbol_data[symbol]
            close_idx = symbol_close_idx[symbol]
            candle_time = index[i]
            open_trade = open_trades[symbol]
            c = config[symbol]
            m = symbol_models[symbol]

            # Close trade if horizon has passed
            if open_trade is not None and i >= open_trade["close_at"]:
                exit_time = index[open_trade["close_at"]] + timedelta(minutes=1)
                exit_price = get_price_at(symbol, exit_time)
                if exit_price is None:
                    exit_price = float(feature_array[open_trade["close_at"], close_idx])

                if open_trade["direction"] == "LONG":
                    pnl = (exit_price - open_trade["entry"]) / open_trade["entry"] * 100
                else:
                    pnl = (open_trade["entry"] - exit_price) / open_trade["entry"] * 100
                pnl -= 0.09
                pnl *= 10
                pnl = max(pnl, -100)
                balance += open_trade["trade_size"] * (pnl / 100)

                all_trade_logs[symbol]["trade_size"].append(round(open_trade["trade_size"], 4))
                all_trade_logs[symbol]["direction"].append(open_trade["direction"])
                all_trade_logs[symbol]["entry"].append(round(open_trade["entry"], 4))
                all_trade_logs[symbol]["exit"].append(round(exit_price, 4))
                all_trade_logs[symbol]["pnl"].append(round(pnl, 4))
                all_trade_logs[symbol]["opened"].append(str(open_trade["opened"]))
                all_trade_logs[symbol]["closed"].append(str(index[open_trade["close_at"]]))
                open_trades[symbol] = None

                print(
                    f"  [{candle_time}] {symbol} CLOSED {open_trade['direction']} @ {exit_price:.4f} | PnL: {pnl:+.2f}% | Balance: ${balance:.2f}")

            # Open trade if none open
            if open_trades[symbol] is None:


                window = feature_array[i - window_size:i]
                X_scaled = scale_live_window(window, m["scaler"])
                prediction = conf_eval_live(
                    m["lstm"], m["cnn"],  m["lr"], X_scaled,
                    c["use_cnn"], c["use_lstm"], c["use_lr"],
                    c["prop_threshold"], c["cnn_threshold"], c["lr_threshold"]
                )
                if prediction != -1:
                    direction = "LONG" if prediction == 1 else "SHORT"
                    entry_price = get_price_at(symbol, candle_time + timedelta(minutes=1))
                    if entry_price is None:
                        entry_price = float(feature_array[i, close_idx])
                    open_trades[symbol] = {
                        "direction": direction,
                        "entry": entry_price,
                        "close_at": i + horizon,
                        "opened": candle_time,
                        "trade_size": balance * 0.07 * c["accuracy"],
                    }
                    print(f"  [{candle_time}] {symbol} OPENED {direction} @ {entry_price:.4f}")

    # ── 5. Summaries ──────────────────────────────────────────────────────────
    for symbol in symbols:
        trade_log = all_trade_logs[symbol]
        pnls = trade_log["pnl"]
        sizes = trade_log["trade_size"]
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            total_invested = sum(sizes)
            total_profit = sum(s * (p / 100) for s, p in zip(sizes, pnls))
            total_returned = total_invested + total_profit
            trade_log["num_trades"] = len(pnls)
            trade_log["win_rate"] = round(wins / len(pnls) * 100, 2)
            trade_log["avg_pnl"] = round(sum(pnls) / len(pnls), 4)
            trade_log["total_invested"] = round(total_invested, 2)
            trade_log["total_returned"] = round(total_returned, 2)
            trade_log["total_profit"] = round(total_profit, 2)
            trade_log["roi"] = round(total_profit / total_invested * 100, 2)
            trade_log["final_balance"] = round(balance, 2)
            print(f"\n  {symbol} Summary:")
            print(f"  Trades: {len(pnls)} | Win rate: {trade_log['win_rate']}% | Avg PnL: {trade_log['avg_pnl']}%")
            print(f"  ROI: {trade_log['roi']}% | Final balance: ${balance:.2f}")
        else:
            print(f"  No trades taken for {symbol}")

    with open("paper_trade_historical.json", "w") as f:
        json.dump(all_trade_logs, f, indent=2)
    print("\nSaved to paper_trade_historical.json")
    return all_trade_logs
def trend_follow_predict(window, close_idx=0, volume_zscore_idx=3, rsi_idx=10, volatility_idx=11, bb_position_idx=13):
    """
    Pure rule-based trend following predictor. No ML.
    Uses last 8 candles (24h at 3h intervals) for all signals.

    Signals:
      1. Momentum     — 24h price change direction and magnitude
      2. Volume       — recent volume above baseline confirms move
      3. RSI filter   — avoid overbought longs / oversold shorts
      4. Vol breakout — price breaking recent 24h high or low

    Returns: 1 (LONG), 0 (SHORT), -1 (no trade)
    window: raw unscaled feature array, shape (window_size, num_features)
    """
    TREND_CANDLES = 8          # 24h lookback
    MOMENTUM_THRESHOLD = 0.005 # minimum 0.5% move to signal trend
    RSI_OVERBOUGHT    = 68     # don't go long above this
    RSI_OVERSOLD      = 32     # don't go short below this
    VOL_MULTIPLIER    = 1.1    # recent volume must be 10% above baseline

    recent = window[-TREND_CANDLES:]   # last 8 candles
    closes = recent[:, close_idx]

    # ── 1. Momentum ───────────────────────────────────────────────────────────
    momentum = (closes[-1] - closes[0]) / (closes[0] + 1e-8)

    # ── 2. Volume confirmation ────────────────────────────────────────────────
    # volume_zscore > 0 means above rolling mean — use last 2 vs previous 6
    vol_zscores = recent[:, volume_zscore_idx]
    vol_recent   = vol_zscores[-2:].mean()
    vol_baseline = vol_zscores[:-2].mean()
    volume_confirming = vol_recent > vol_baseline * VOL_MULTIPLIER

    # ── 3. RSI filter ─────────────────────────────────────────────────────────
    rsi = recent[-1, rsi_idx]

    # ── 4. Volatility breakout ────────────────────────────────────────────────
    # Is current close breaking out of previous 7 candles range?
    prior_closes  = closes[:-1]
    breakout_up   = closes[-1] > prior_closes.max()
    breakout_down = closes[-1] < prior_closes.min()

    # ── Decision ─────────────────────────────────────────────────────────────
    long_signal  = (momentum >  MOMENTUM_THRESHOLD and
                    volume_confirming and
                    rsi < RSI_OVERBOUGHT and
                    breakout_up)

    short_signal = (momentum < -MOMENTUM_THRESHOLD and
                    volume_confirming and
                    rsi > RSI_OVERSOLD and
                    breakout_down)

    if long_signal:
        return 1
    if short_signal:
        return 0
    return -1
def trend_follow_paper_trade(months, window_days, resample_hours, horizon, cutoff=0, paper_trade_days=30):
    """
    Pure rule-based trend following paper trade. No ML, no training.
    Uses trend_follow_predict() on every candle.

    Signals (all must agree to open):
      - Momentum:  24h price change > 0.5%
      - Volume:    recent volume above baseline
      - RSI:       not overbought/oversold
      - Breakout:  price breaking 24h high/low
    """
    from .data_fetch import download_data, compute_features

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]

    print(f"Trend follow paper trade — last {paper_trade_days} days (cutoff={cutoff})...")
    steps_per_day = 24 // resample_hours
    window_size = window_days * steps_per_day
    total_steps_needed = window_size + paper_trade_days * steps_per_day

    # ── 1. Download data ───────────────────────────────────────────────────────
    symbol_data = {}
    symbol_feat_cols = {}
    for symbol in symbols:
        df_raw = download_data(symbol, months=2 + window_days // 30, cutoff=cutoff)
        df_feat, feat_cols = compute_features(df_raw, resample_hours)
        df_sim = df_feat.iloc[-total_steps_needed:]
        symbol_data[symbol] = (df_sim.values.astype(np.float32), df_sim.index)
        symbol_feat_cols[symbol] = feat_cols

    # ── 2. Pre-compute baseline volatility ────────────────────────────────────
    symbol_vol_threshold = {}
    for symbol in symbols:
        df_raw = download_data(symbol, months=2 + window_days // 30, cutoff=cutoff)
        df_feat, _ = compute_features(df_raw, resample_hours)
        df_baseline = df_feat.iloc[:-total_steps_needed]
        if len(df_baseline) > 1:
            baseline_closes = df_baseline['Close'].values
            baseline_returns = np.diff(baseline_closes) / baseline_closes[:-1]
            baseline_vol = np.std(baseline_returns) * 100
        else:
            baseline_vol = 1.0
        symbol_vol_threshold[symbol] = baseline_vol * 2.0
        print(f"  {symbol} vol threshold: {symbol_vol_threshold[symbol]:.4f}%")

    # ── 3. Get feature indices ─────────────────────────────────────────────────
    def get_idx(cols, name, fallback):
        return cols.index(name) if name in cols else fallback

    balance = 1000.0
    max_concurrent = 2
    COOLDOWN_LOSSES = 3
    open_trades = {symbol: None for symbol in symbols}
    consecutive_losses = {symbol: 0 for symbol in symbols}
    all_trade_logs = {symbol: {"direction": [], "entry": [], "exit": [], "pnl": [],
                               "opened": [], "closed": [], "trade_size": []} for symbol in symbols}

    # ── 4. Unified candle walk ─────────────────────────────────────────────────
    min_len = min(len(symbol_data[s][0]) for s in symbols)
    for i in range(window_size, min_len - horizon):
        for symbol in symbols:
            feature_array, index = symbol_data[symbol]
            feat_cols = symbol_feat_cols[symbol]
            close_idx        = get_idx(feat_cols, 'Close', 0)
            vol_zscore_idx   = get_idx(feat_cols, 'volume_zscore', 3)
            rsi_idx          = get_idx(feat_cols, 'RSI', 10)
            volatility_idx   = get_idx(feat_cols, 'Volatility', 11)
            bb_position_idx  = get_idx(feat_cols, 'bb_position', 13)

            candle_time = index[i]
            open_trade = open_trades[symbol]

            # ── Close trade ───────────────────────────────────────────────────
            if open_trade is not None and i >= open_trade["close_at"]:
                exit_time = index[open_trade["close_at"]] + timedelta(minutes=1)
                exit_price = get_price_at(symbol, exit_time)
                if exit_price is None:
                    exit_price = float(feature_array[open_trade["close_at"], close_idx])

                if open_trade["direction"] == "LONG":
                    pnl = (exit_price - open_trade["entry"]) / open_trade["entry"] * 100
                else:
                    pnl = (open_trade["entry"] - exit_price) / open_trade["entry"] * 100
                pnl -= 0.18
                pnl *= 20
                pnl = max(pnl, -100)
                balance += open_trade["trade_size"] * (pnl / 100)

                if pnl < 0:
                    consecutive_losses[symbol] += 1
                else:
                    consecutive_losses[symbol] = 0

                all_trade_logs[symbol]["trade_size"].append(round(open_trade["trade_size"], 4))
                all_trade_logs[symbol]["direction"].append(open_trade["direction"])
                all_trade_logs[symbol]["entry"].append(round(open_trade["entry"], 4))
                all_trade_logs[symbol]["exit"].append(round(exit_price, 4))
                all_trade_logs[symbol]["pnl"].append(round(pnl, 4))
                all_trade_logs[symbol]["opened"].append(str(open_trade["opened"]))
                all_trade_logs[symbol]["closed"].append(str(index[open_trade["close_at"]]))
                open_trades[symbol] = None



            # ── Open trade ────────────────────────────────────────────────────
            if open_trades[symbol] is None:

                # Losing streak cooldown
                if consecutive_losses[symbol] >= COOLDOWN_LOSSES:
                    continue

                # Max concurrent positions
               ## open_count = sum(1 for t in open_trades.values() if t is not None)
                ##if open_count >= max_concurrent:
                  ##  continue

                # Volatility filter
                recent_closes = feature_array[i - 24:i, close_idx]
                recent_returns = np.diff(recent_closes) / recent_closes[:-1]
                current_vol = np.std(recent_returns) * 100
                if current_vol > symbol_vol_threshold[symbol]:
                    continue

                window = feature_array[i - window_size:i]
                prediction = trend_follow_predict(
                    window,
                    close_idx=close_idx,
                    volume_zscore_idx=vol_zscore_idx,
                    rsi_idx=rsi_idx,
                    volatility_idx=volatility_idx,
                    bb_position_idx=bb_position_idx,
                )

                if prediction != -1:
                    direction = "LONG" if prediction == 1 else "SHORT"
                    entry_price = get_price_at(symbol, candle_time + timedelta(minutes=1))
                    if entry_price is None:
                        entry_price = float(feature_array[i, close_idx])
                    open_trades[symbol] = {
                        "direction": direction,
                        "entry": entry_price,
                        "close_at": i + horizon,
                        "opened": candle_time,
                        "trade_size": balance * 0.05 ,
                    }
                    print(f"  [{candle_time}] {symbol} OPENED {direction} @ {entry_price:.4f} | Vol: {current_vol:.4f}%")

    # ── 5. Summaries ──────────────────────────────────────────────────────────
    for symbol in symbols:
        trade_log = all_trade_logs[symbol]
        pnls = trade_log["pnl"]
        sizes = trade_log["trade_size"]
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            total_invested = sum(sizes)
            total_profit = sum(s * (p / 100) for s, p in zip(sizes, pnls))
            total_returned = total_invested + total_profit
            trade_log["num_trades"] = len(pnls)
            trade_log["win_rate"] = round(wins / len(pnls) * 100, 2)
            trade_log["avg_pnl"] = round(sum(pnls) / len(pnls), 4)
            trade_log["total_invested"] = round(total_invested, 2)
            trade_log["total_returned"] = round(total_returned, 2)
            trade_log["total_profit"] = round(total_profit, 2)
            trade_log["roi"] = round(total_profit / total_invested * 100, 2)
            trade_log["final_balance"] = round(balance, 2)
            print(f"\n  {symbol} Summary:")
            print(f"  Trades: {len(pnls)} | Win rate: {trade_log['win_rate']}% | Avg PnL: {trade_log['avg_pnl']}%")
            print(f"  ROI: {trade_log['roi']}% | Final balance: ${balance:.2f}")
        else:
            print(f"  No trades taken for {symbol}")

    with open("trend_follow_paper_trade.json", "w") as f:
        json.dump(all_trade_logs, f, indent=2)
    print("\nSaved to trend_follow_paper_trade.json")
    return all_trade_logs

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

        lstm = ImprovedLSTMModel(input_size=input_size, hidden_size=cfg["hidden_size"],
                                 num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
        lstm.load_state_dict(torch.load(f"{dataset_dir}/42_binary_model.pt", map_location=device))
        lstm.eval()
        with open(f"{dataset_dir}/cnn_model_config.json") as f:
            cnn_cfg = json.load(f)
        cnn = CNNModel(input_size=input_size,num_filters=cnn_cfg["num_filters"],kernel_size=cnn_cfg["kernel_size"],dropout=cnn_cfg["dropout"]).to(device)
        cnn.load_state_dict(torch.load(f"{dataset_dir}/42_cnn_model.pt", map_location=device))
        cnn.eval()



        symbol_models[s] = {
            "lstm": lstm,
            "cnn": cnn,
            "lr": joblib.load(f"{dataset_dir}/42_lr_model.pkl"),
            "scaler": joblib.load(f"{dataset_dir}/scaler.pkl"),
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

                    prediction = conf_eval_live(m["lstm"], m["cnn"], m["lr"], m["mlp"],  X,c["use_cnn"],c["use_ltsm"],c["use_mlp"],c["use_lr"],c["prop_threshold"],c["cnn_threshold"],c["mlp_threshold"],c["lr_threshold"])
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





