from itertools import combinations
import numpy as np
import json
import math
from pathlib import Path
def save_eval_results(results, dataset_dir):
    """
    results: list of dicts built during evaluation
    Saves to {dataset_dir}/eval_results.json
    """
    path = Path(dataset_dir) / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
def _compute_score(accuracy_pct, num_trades):
    """
    Compounding score: (2 * accuracy - 0.01) ** num_trades
    accuracy_pct: 0-100 scale
    """
    if accuracy_pct is None or num_trades is None or num_trades == 0:
        return None
    base = (2 * accuracy_pct - 0.04)
    if base <= 0:
        return None
    accuracy_weight = 1.2
    return round(math.log(base) * num_trades ** (1/accuracy_weight), 6)
def load_eval_results(dataset_dir):
    path = Path(dataset_dir) / "eval_results.json"
    if not path.exists():
        return {"symbol": dataset_dir, "test": [], "val": []}

    with open(path) as f:
        data = json.load(f)

    # Migrate old scalar entries to lists
    for split in ["test", "val"]:
        for entry in data.get(split, []):
            for field in ["accuracy", "num_trades", "coverage_pct","accuracy_avg","accuracy_avg_weighted"]:
                if field in entry and not isinstance(entry[field], list):
                    entry[field] = [entry[field]]

    return data
def insert_eval_results(row,new_result):
    key = (new_result['use_lstm'], new_result['use_cnn'],new_result['use_mlp'],new_result['use_lr'],new_result['threshold'],new_result['cnn_threshold'],new_result['mlp_threshold'],new_result['lr_threshold'])

    for existing in row:
        existing_key = (existing['use_lstm'], existing['use_cnn'],existing['use_mlp'],existing['use_lr'],existing['threshold'],existing['cnn_threshold'],existing['mlp_threshold'],existing['lr_threshold'])
        if existing_key == key:
            existing['accuracy'].append(new_result['accuracy'])
            existing['num_trades'].append(new_result['num_trades'])
            existing['coverage_pct'].append(new_result['coverage_pct'])
            valid = [a for a in existing['accuracy'] if a is not None]
            existing['accuracy_avg'] = round(sum(valid) / len(valid), 2) if valid else None
            valid = [(a, t) for a, t in zip(existing['accuracy'], existing['num_trades'])
                     if a is not None]
            total_trades = sum(t for _, t in valid)
            existing['accuracy_avg_weighted'] = round(
                sum(a * t for a, t in valid) / total_trades, 2
            ) if total_trades > 0 else None
            existing['score'] = _compute_score(existing['accuracy_avg_weighted'], total_trades)
            return
    score = _compute_score(new_result['accuracy'], new_result['num_trades'])
    row.append({
        'use_lstm': new_result['use_lstm'],
        'use_cnn': new_result['use_cnn'],
        'use_mlp':new_result['use_mlp'],
        'use_lr':new_result['use_lr'],
        'threshold':new_result['threshold'],
        'cnn_threshold':new_result['cnn_threshold'],
        'mlp_threshold':new_result['mlp_threshold'],
        'lr_threshold':new_result['lr_threshold'],
        'accuracy': new_result['accuracy'],
        'accuracy_avg':new_result['accuracy'],
        'accuracy_avg_weighted':new_result['accuracy'],
        'score':score,
        'num_trades': new_result['num_trades'],
        'coverage_pct': new_result['coverage_pct'],
    })

def get_best_results(row,get_acc=False):
    best_acc = 0.0
    result = {
        "use_lstm": False,
        "use_cnn": False,
        "use_mlp": False,
        "use_lr": False,
        "prop_threshold": 0.0,
        "cnn_threshold":0.0,
        "mlp_threshold":0.0,
        "lr_threshold":0.0,
        "accuracy": 0,
        "score":0,
    }
    for exists in row:
        if get_acc:

            if exists['accuracy_avg_weighted'] is None:
                continue
            if exists['accuracy_avg_weighted'] > result['accuracy']:
                result['accuracy'] = exists['accuracy_avg_weighted']
                result['use_lr'] = exists['use_lr']
                result['use_mlp'] = exists['use_mlp']
                result['use_cnn'] = exists['use_cnn']
                result['use_lstm'] = exists['use_lstm']
                result['prop_threshold'] = exists['threshold']
                result['cnn_threshold'] = exists['cnn_threshold']
                result['mlp_threshold'] = exists['mlp_threshold']
                result['lr_threshold'] = exists['lr_threshold']
                result['score'] = exists['score']
        else:
            if exists['score'] is None:
                continue
            if exists['score'] > result['score']:
                result['accuracy'] = exists['accuracy_avg_weighted']
                result['use_lr'] = exists['use_lr']
                result['use_mlp'] = exists['use_mlp']
                result['use_cnn'] = exists['use_cnn']
                result['use_lstm'] = exists['use_lstm']
                result['prop_threshold'] = exists['threshold']
                result['cnn_threshold'] = exists['cnn_threshold']
                result['mlp_threshold'] = exists['mlp_threshold']
                result['lr_threshold'] = exists['lr_threshold']
                result['score'] = exists['score']
    print(json.dumps(result, indent=2))
    return result
