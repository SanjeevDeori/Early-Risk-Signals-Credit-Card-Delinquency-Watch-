"""Real data loader and signal computation for credit card delinquency watch.

Maps real dataset columns to signal categories and computes early risk scores.
Supports both simulated and real data workflows.
"""
import pandas as pd
import numpy as np


def load_real_data(filepath):
    """Load real credit card data from CSV.
    
    Expected columns:
    - Customer ID
    - Credit Limit
    - Utilisation %
    - Avg Payment Ratio
    - Min Due Paid Frequency
    - Merchant Mix Index
    - Cash Withdrawal %
    - Recent Spend Change %
    - DPD Bucket Next Month (label for backtesting)
    
    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(filepath, sep='\t')
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Map to standard names for processing
    column_mapping = {
        'Customer ID': 'customer_id',
        'Credit Limit': 'credit_limit',
        'Utilisation %': 'utilization',
        'Avg Payment Ratio': 'avg_payment_ratio',
        'Min Due Paid Frequency': 'min_due_frequency',
        'Merchant Mix Index': 'merchant_mix_index',
        'Cash Withdrawal %': 'cash_withdrawal_pct',
        'Recent Spend Change %': 'spend_change_pct',
        'DPD Bucket Next Month': 'dpd_bucket_next_month',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Ensure numeric types
    numeric_cols = [col for col in df.columns if col not in ['customer_id']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def compute_real_data_signals(df):
    """Compute risk signals from real dataset.
    
    Maps real columns to signal categories:
    1. Payment Behavior: avg_payment_ratio, min_due_frequency
    2. Spending Pattern: spend_change_pct, cash_withdrawal_pct, merchant_mix_index
    3. Credit Stress: utilization, credit_limit
    4. Transaction Anomalies: merchant_mix_index concentration
    5. External: credit_limit (proxy for credit tier risk)
    
    Args:
        df: DataFrame with real customer data
        
    Returns:
        DataFrame with risk scores and tier assignments
    """
    scores = []
    
    for idx, row in df.iterrows():
        customer_id = row['customer_id']
        
        # Signal 1: Payment Behavior (40 points)
        # Low payment ratio = risk; low min_due_frequency = risk
        payment_ratio_normalized = min(row['avg_payment_ratio'] / 100.0, 1.0)
        min_due_freq_normalized = min(row['min_due_frequency'] / 100.0, 1.0)
        payment_score = (
            40 * (1 - payment_ratio_normalized) +  # Inverse: lower ratio = higher risk
            20 * (1 - min_due_freq_normalized)  # Inverse: lower frequency = higher risk
        )
        
        # Signal 2: Spending Pattern Disruption (30 points)
        # Negative spend change = risk; high cash withdrawal = risk
        spend_decline = max(0.0, -row['spend_change_pct'] / 100.0)  # Negative is bad
        cash_withdrawal_normalized = min(row['cash_withdrawal_pct'] / 30.0, 1.0)  # 30% is max
        spending_score = (
            20 * spend_decline +
            10 * cash_withdrawal_normalized
        )
        
        # Signal 3: Credit Stress (30 points)
        # High utilization = risk; low limit may indicate previous stress
        utilization_normalized = min(row['utilization'] / 100.0, 1.0)
        credit_limit_normalized = min(row['credit_limit'] / 200000.0, 1.0)
        credit_stress_score = (
            25 * utilization_normalized +  # High util = risk
            5 * (1 - credit_limit_normalized)  # Lower limit = risk (prior stress)
        )
        
        # Signal 4: Merchant Concentration Anomaly (15 points)
        # Merchant mix index close to 1.0 = concentrated spending = risk
        merchant_concentration = max(0.0, (row['merchant_mix_index'] - 0.5) / 0.5)
        anomaly_score = 15 * merchant_concentration
        
        # Signal 5: External / External (5 points)
        # Very high utilization is a strong stress signal
        external_score = 5 * max(0.0, (utilization_normalized - 0.8) / 0.2)
        
        # Composite score (0-100 scale)
        composite = payment_score + spending_score + credit_stress_score + anomaly_score + external_score
        composite = min(100.0, max(0.0, composite))
        
        # Tier assignment
        if composite >= 76:
            tier = 'INTERVENE'
        elif composite >= 51:
            tier = 'ENGAGE'
        else:
            tier = 'MONITOR'
        
        scores.append({
            'customer_id': customer_id,
            'utilization': row['utilization'],
            'avg_payment_ratio': row['avg_payment_ratio'],
            'min_due_frequency': row['min_due_frequency'],
            'cash_withdrawal_pct': row['cash_withdrawal_pct'],
            'spend_change_pct': row['spend_change_pct'],
            'merchant_mix_index': row['merchant_mix_index'],
            'credit_limit': row['credit_limit'],
            'payment_score': payment_score,
            'spending_score': spending_score,
            'credit_stress_score': credit_stress_score,
            'anomaly_score': anomaly_score,
            'composite_score': composite,
            'tier': tier,
            'dpd_bucket_next_month': row.get('dpd_bucket_next_month', -1),
        })
    
    return pd.DataFrame(scores)


if __name__ == '__main__':
    # Test on real data
    df = load_real_data('credit_card_data.csv')
    print(f'Loaded {len(df)} customers')
    print(df.head())
    
    scores = compute_real_data_signals(df)
    print('\nComputed scores:')
    print(scores[['customer_id', 'composite_score', 'tier']].head(10))
    print('\nTier distribution:')
    print(scores['tier'].value_counts())
