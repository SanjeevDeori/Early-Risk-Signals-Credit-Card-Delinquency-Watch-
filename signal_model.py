import pandas as pd
import numpy as np

DEFAULT_WEIGHTS = {
    'payment': 0.30,
    'spending': 0.25,
    'credit_stress': 0.25,
    'anomaly': 0.12,
    'external': 0.08,
}

def compute_customer_signals(df, weights=None):
    """Compute per-customer signals using the last two months of data where available.

    Args:
        df: Customer-month transaction data
        weights: Optional dict overriding default weights for score components

    Returns a summary DataFrame with signals and scores scaled 0-100.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    # ensure ordering
    df = df.copy()
    df.sort_values(['customer_id', 'month'], inplace=True)

    summaries = []
    for cid, g in df.groupby('customer_id'):
        if g['month'].nunique() < 2:
            continue
        last = g[g['month'] == g['month'].max()].iloc[0]
        prev = g[g['month'] == g['month'].max() - 1].iloc[0]

        # Payment signals
        min_due_switch = int((prev['payment_amt'] >= prev['min_due'] * 0.95) and (last['payment_amt'] <= last['min_due'] * 1.05))
        payment_decay_pct = max(0.0, (prev['payment_amt'] - last['payment_amt']) / max(1.0, prev['payment_amt']))
        timing_drift = last['payment_day'] - prev['payment_day']

        # Spending signals
        velocity_decline_pct = max(0.0, (prev['spend'] - last['spend']) / max(1.0, prev['spend']))
        cash_advance_flag = int(last['cash_adv'] > prev['cash_adv'] and last['cash_adv'] > 0)
        essentials_increase = max(0.0, last['essentials_ratio'] - prev['essentials_ratio'])

        # Credit stress
        utilization = last['utilization']
        utilization_above_70 = max(0.0, (utilization - 0.7) / 0.3) if utilization > 0.7 else 0.0
        declined_count = int(last['declined_count'])
        revolving_growth = max(0.0, (last['balance'] - prev['balance']) / max(1.0, prev['balance']))

        # Anomaly
        small_txn_freq = max(0, int(last['num_txn'] / 10))
        merchant_concentration = last['essentials_ratio']

        # Scoring (simple linear weights as in the design doc)
        payment_score = (40 * min_due_switch) + (30 * payment_decay_pct * 100) + (20 * min(30, abs(timing_drift))) + (10 * 0)
        spending_score = (35 * velocity_decline_pct * 100) + (30 * cash_advance_flag) + (25 * essentials_increase * 100) + (10 * 0)
        credit_stress_score = (40 * utilization_above_70 * 100) + (30 * min(declined_count, 10)) + (20 * min(revolving_growth, 1.0) * 100) + (10 * 0)
        anomaly_score = (25 * min(1.0, abs(timing_drift)/30)) + (25 * min(1.0, small_txn_freq/10)) + (25 * merchant_concentration * 100) + (25 * 0)

        # normalize to 0-100 per component
        def clamp(x):
            return float(max(0.0, min(100.0, x)))

        payment_score = clamp(payment_score / 1.0 if payment_score <= 100 else 100.0)
        spending_score = clamp(spending_score / 1.0 if spending_score <= 100 else 100.0)
        credit_stress_score = clamp(credit_stress_score / 1.0 if credit_stress_score <= 100 else 100.0)
        anomaly_score = clamp(anomaly_score / 1.0 if anomaly_score <= 100 else 100.0)

        composite = (
            weights['payment'] * payment_score +
            weights['spending'] * spending_score +
            weights['credit_stress'] * credit_stress_score +
            weights['anomaly'] * anomaly_score +
            weights['external'] * 0
        )

        # tier assignment
        tier = 'MONITOR'
        if composite >= 76:
            tier = 'INTERVENE'
        elif composite >= 51:
            tier = 'ENGAGE'

        summaries.append({
            'customer_id': cid,
            'payment_score': payment_score,
            'spending_score': spending_score,
            'credit_stress_score': credit_stress_score,
            'anomaly_score': anomaly_score,
            'composite_score': composite,
            'tier': tier,
            'utilization': utilization,
            'velocity_decline_pct': velocity_decline_pct,
            'payment_decay_pct': payment_decay_pct,
        })

    return pd.DataFrame(summaries)

if __name__ == '__main__':
    import simulate
    df = simulate.simulate_customers(300, months=6)
    s = compute_customer_signals(df)
    print('Default weights:')
    print(s.head())
    
    # Example with custom weights
    custom_weights = {'payment': 0.40, 'spending': 0.30, 'credit_stress': 0.20, 'anomaly': 0.10, 'external': 0.0}
    s_custom = compute_customer_signals(df, weights=custom_weights)
    print('\nCustom weights (payment-heavy):')
    print(s_custom.head())
