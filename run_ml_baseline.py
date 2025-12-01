"""Script to run ML baseline experiments and compare to rule-based scoring.

Usage:
  python run_ml_baseline.py

This will simulate a cohort, compute signals, inject outcomes, train RandomForest and XGBoost (if installed),
and save a small feature-importance plot to `artifacts/`.
"""
import os
import json
import matplotlib.pyplot as plt

from simulate import simulate_customers
from signal_model import compute_customer_signals
from ml_models import prepare_ml_dataset, train_random_forest, try_train_xgboost
from backtest import compute_backtest_metrics
import json


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_feature_importance(fi: dict, outpath: str):
    if not fi:
        print('No feature importance to plot')
        return
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(8, max(3, len(names)*0.3)))
    plt.barh(range(len(names)), vals[::-1])
    plt.yticks(range(len(names)), names[::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    ensure_dir('artifacts')
    print('Simulating cohort...')
    df = simulate_customers(2000, months=6, seed=123)
    scores = compute_customer_signals(df)

    print('Preparing dataset and injecting outcomes...')
    X, y, full = prepare_ml_dataset(scores, delinquency_rate=0.05, signal_strength=0.7, seed=42)

    print('Training Random Forest...')
    rf_res = train_random_forest(X, y)
    print('RF Metrics:', json.dumps(rf_res['metrics'], indent=2))
    if rf_res.get('feature_importances'):
        plot_feature_importance(rf_res['feature_importances'], 'artifacts/feature_importance_rf.png')
        print('Saved RF feature importance to artifacts/feature_importance_rf.png')

    print('Attempting XGBoost (if installed)...')
    xgb_res = try_train_xgboost(X, y)
    if xgb_res is None:
        print('XGBoost not installed; skipping XGBoost baseline.')
    else:
        print('XGBoost Metrics:', json.dumps(xgb_res['metrics'], indent=2))
        if xgb_res.get('feature_importances'):
            plot_feature_importance(xgb_res['feature_importances'], 'artifacts/feature_importance_xgb.png')
            print('Saved XGB feature importance to artifacts/feature_importance_xgb.png')

    # Compute rule-based metrics using backtest harness and save a comparison JSON
    try:
        print('Computing rule-based backtest metrics...')
        rule_metrics = compute_backtest_metrics(full)
        # Extract scalar-friendly items
        def scalarize(m):
            out = {}
            for k, v in m.items():
                if isinstance(v, (float, int)):
                    out[k] = v
                elif isinstance(v, (list, tuple)):
                    out[k] = v
                else:
                    try:
                        out[k] = float(v)
                    except Exception:
                        out[k] = str(type(v))
            return out

        comparison = {
            'rule_based': scalarize(rule_metrics),
            'random_forest': rf_res['metrics'],
            'xgboost': xgb_res['metrics'] if xgb_res is not None else None,
        }

        with open('artifacts/ml_comparison.json', 'w') as fh:
            json.dump(comparison, fh, indent=2)
        print('Saved ML comparison to artifacts/ml_comparison.json')
    except Exception as e:
        print('Could not compute rule-based metrics:', e)


if __name__ == '__main__':
    main()
