"""ML baseline utilities: train RandomForest / XGBoost and report feature importance.

This module is conservative about optional dependencies: it will skip XGBoost/LightGBM
if they are not installed and will raise informative errors if scikit-learn is missing.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from backtest import inject_outcomes


def prepare_ml_dataset(scores_df: pd.DataFrame, label_col: str = 'became_delinquent',
                       delinquency_rate: float = 0.05, signal_strength: float = 0.7, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare X, y for ML training from a scored DataFrame.

    - If `label_col` not present, inject synthetic outcomes using `inject_outcomes`.
    - Select numeric feature columns and drop identifier columns.
    Returns: X, y, full_df (with labels)
    """
    if label_col not in scores_df.columns:
        full = inject_outcomes(scores_df, delinquency_rate=delinquency_rate, signal_strength=signal_strength, seed=seed)
    else:
        full = scores_df.copy()

    # Numeric features only (drop identifiers)
    num = full.select_dtypes(include=[np.number]).copy()
    drop_cols = [c for c in ['customer_id'] if c in num.columns]
    if label_col in num.columns:
        drop_cols.append(label_col)
    X = num.drop(columns=drop_cols)
    y = full[label_col]
    return X, y, full


def train_random_forest(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Train a Random Forest classifier and return model + eval metrics + feature importances.

    Imports scikit-learn locally so the module can be imported even when sklearn
    is not installed in the environment (useful for lightweight deployments).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
    except Exception as e:
        raise ImportError('scikit-learn is required to train ML baselines. Install scikit-learn in this Python environment.') from e

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state, class_weight='balanced')
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'pr_auc': float(average_precision_score(y_test, probs)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
    }

    fi = None
    if hasattr(clf, 'feature_importances_'):
        fi = dict(zip(X.columns.tolist(), clf.feature_importances_.tolist()))

    return {'model': clf, 'metrics': metrics, 'feature_importances': fi, 'X_test': X_test, 'y_test': y_test, 'y_proba': probs}


def try_train_xgboost(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Optional[Dict[str, Any]]:
    """Attempt to train an XGBoost classifier. Returns same dict as train_random_forest or None if xgboost missing."""
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=random_state, verbosity=0)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'pr_auc': float(average_precision_score(y_test, probs)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
    }

    fi = None
    if hasattr(clf, 'feature_importances_'):
        fi = dict(zip(X.columns.tolist(), clf.feature_importances_.tolist()))

    return {'model': clf, 'metrics': metrics, 'feature_importances': fi, 'X_test': X_test, 'y_test': y_test, 'y_proba': probs}
