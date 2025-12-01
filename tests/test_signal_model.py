import pandas as pd
from simulate import simulate_customers
from signal_model import compute_customer_signals


def test_compute_customer_signals_basic():
    df = simulate_customers(n_customers=50, months=3, seed=1)
    s = compute_customer_signals(df)
    assert isinstance(s, pd.DataFrame)
    assert 'composite_score' in s.columns
    # Scores should be within 0-100
    assert s['composite_score'].between(0, 100).all()
