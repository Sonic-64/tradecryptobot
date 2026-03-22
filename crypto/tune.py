from itertools import combinations
import numpy as np
import json
import math
from pathlib import Path

from crypto import analyze_days


def save_eval_results(results, dataset_dir):
    """
    results: list of dicts built during evaluation
    Saves to {dataset_dir}/eval_results.json
    """
    path = Path(dataset_dir) / "eval_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
def _compute_score(accuracy_pct,coverage):
    """
    Compounding score: (2 * accuracy - 0.01) ** num_trades
    accuracy_pct: 0-100 scale
    """
    if accuracy_pct is None or coverage is None or coverage == 0:
        return 0

    acc = accuracy_pct/100
    base = (2 * acc - 0.1)
    if base <= 0:
        return 0
    coverage_factor = coverage ** 1.2
    return round(math.log(2*acc - 0.06) * coverage_factor, 6)
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
    key = (new_result['use_lstm'], new_result['use_cnn'],new_result['threshold'],new_result['cnn_threshold'])

    for existing in row:
        existing_key = (existing['use_lstm'], existing['use_cnn'],existing['threshold'],existing['cnn_threshold'])
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
            valid_coverage = [c for c in existing['coverage_pct'] if c is not None]
            existing['coverage_avg'] = round(
                sum(valid_coverage) / len(valid_coverage), 2
            ) if valid_coverage else None
            existing['score'] = _compute_score(existing['accuracy_avg_weighted'],existing["coverage_avg"])
            return
    score = _compute_score(new_result['accuracy'], new_result['coverage_pct'])
    row.append({
        'use_lstm': new_result['use_lstm'],
        'use_cnn': new_result['use_cnn'],
        'threshold':new_result['threshold'],
        'cnn_threshold':new_result['cnn_threshold'],
        'accuracy': new_result['accuracy'],
        'accuracy_avg':new_result['accuracy'],
        'accuracy_avg_weighted':new_result['accuracy'],
        'score':score,
        'num_trades': new_result['num_trades'],
        'coverage_pct': new_result['coverage_pct'],
        'coverage_avg':new_result['coverage_pct'],
    })

def get_best_config(dataset_dir,get_acc=False):
    evalpath = Path(dataset_dir) / "eval_results.json"

    if evalpath.exists() and evalpath.stat().st_size > 0:
        eval_results = load_eval_results(dataset_dir)
    else:
        return None
    row = eval_results["val"]
    result = {
        "use_lstm": False,
        "use_cnn": False,
        "prop_threshold": 0.0,
        "cnn_threshold":0.0,
        "accuracy": 0,
        "coverage": 0,
        "score":0,
        "DIP_PCT":0,
        "MEDIAN_CHANGE":0,
        "BE":0,
    }
    symbol = dataset_dir.split("_")[0]

    _ , result["MEDIAN_CHANGE"], _ , result["DIP_PCT"] = analyze_days(symbol,180)
    result["BE"] =(result["MEDIAN_CHANGE"] - (0.7*result["DIP_PCT"])) / (2*result["MEDIAN_CHANGE"])
    for exists in row:
        if get_acc:

            if exists['accuracy_avg_weighted'] is None:
                continue
            if exists['accuracy_avg_weighted'] > result['accuracy']:
                result['accuracy'] = exists['accuracy_avg_weighted']
                result['coverage'] = exists['coverage_avg']
                result['use_cnn'] = exists['use_cnn']
                result['use_lstm'] = exists['use_lstm']
                result['prop_threshold'] = exists['threshold']
                result['cnn_threshold'] = exists['cnn_threshold']
                result['score'] = exists['score']
        else:
            if exists['score'] is None:
                continue
            if exists['score'] > result['score']:
                result['accuracy'] = exists['accuracy_avg_weighted']
                result['coverage'] = exists['coverage_avg']
                result['use_cnn'] = exists['use_cnn']
                result['use_lstm'] = exists['use_lstm']
                result['prop_threshold'] = exists['threshold']
                result['cnn_threshold'] = exists['cnn_threshold']
                result['score'] = exists['score']
    print(json.dumps(result, indent=2))
    return result
