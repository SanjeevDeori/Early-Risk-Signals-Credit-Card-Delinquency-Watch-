from simulate import simulate_customers
from signal_model import compute_customer_signals
from backtest import inject_outcomes, compute_backtest_metrics


def test_backtest_metrics_run():
    df = simulate_customers(n_customers=200, months=4, seed=2)
    s = compute_customer_signals(df)
    outcomes = inject_outcomes(s, delinquency_rate=0.05, signal_strength=0.6, seed=2)
    metrics = compute_backtest_metrics(outcomes)
    # Basic metric checks
    assert 'roc_auc' in metrics
    assert 0.0 <= metrics['roc_auc'] <= 1.0
