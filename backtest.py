"""Backtest harness for early risk signal validation.

This module injects simulated delinquency outcomes (30+ DPD) into scored cohorts,
then computes validation metrics like precision, recall, lead time, and ROC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix


def inject_outcomes(scores_df, delinquency_rate=0.05, signal_strength=0.7, seed=42):
    """Inject synthetic 30+ DPD outcomes correlated with composite score.

    Args:
        scores_df: DataFrame with columns ['customer_id', 'composite_score', 'tier', ...]
        delinquency_rate: Overall % of customers who become delinquent
        signal_strength: Correlation strength between score and outcome (0-1)
        seed: Random seed

    Returns:
        DataFrame with added 'became_delinquent' (0/1) column
    """
    rng = np.random.default_rng(seed)
    n = len(scores_df)

    # Base delinquency probability
    base_prob = delinquency_rate

    # Scale scores to 0-1
    scores_norm = (scores_df['composite_score'] - scores_df['composite_score'].min()) / (
        scores_df['composite_score'].max() - scores_df['composite_score'].min() + 1e-6
    )

    # Outcome probability = base + (signal_strength * normalized_score)
    outcome_probs = base_prob + signal_strength * (scores_norm - 0.5)
    outcome_probs = np.clip(outcome_probs, 0, 1)

    outcomes = rng.binomial(1, outcome_probs)

    result = scores_df.copy()
    result['became_delinquent'] = outcomes
    return result


def compute_backtest_metrics(outcomes_df, score_column='composite_score'):
    """Compute precision, recall, lead_time proxy, and ROC metrics.

    Args:
        outcomes_df: DataFrame with 'composite_score' and 'became_delinquent' columns

    Returns:
        Dictionary of metrics
    """
    y_true = outcomes_df['became_delinquent']
    y_score = outcomes_df[score_column]

    # Precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # At threshold 50 (ENGAGE tier cutoff)
    engagement_threshold = 50
    flagged = y_score >= engagement_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged).ravel()

    precision_at_50 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_at_50 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at_50 = 2 * (precision_at_50 * recall_at_50) / (precision_at_50 + recall_at_50 + 1e-6)

    # Lead time proxy: avg score of delinquent cohort
    delinquent_scores = outcomes_df[outcomes_df['became_delinquent'] == 1][score_column]
    lead_time_signal = delinquent_scores.mean() if len(delinquent_scores) > 0 else 0.0

    metrics = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision_at_50': precision_at_50,
        'recall_at_50': recall_at_50,
        'f1_at_50': f1_at_50,
        'lead_time_signal': lead_time_signal,
        'flagged_count': int(flagged.sum()),
        'delinquent_count': int(y_true.sum()),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'fpr': fpr,
        'tpr': tpr,
        'recall': recall,
        'precision': precision,
    }

    return metrics


if __name__ == '__main__':
    from simulate import simulate_customers
    from signal_model import compute_customer_signals

    df = simulate_customers(500, months=6)
    scores = compute_customer_signals(df)
    outcomes = inject_outcomes(scores, delinquency_rate=0.05)
    metrics = compute_backtest_metrics(outcomes)
    print('Backtest Metrics:')
    for k, v in metrics.items():
        if not isinstance(v, np.ndarray):
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
