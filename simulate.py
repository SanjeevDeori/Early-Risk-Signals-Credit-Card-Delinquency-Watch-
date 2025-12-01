import numpy as np
import pandas as pd

def simulate_customers(n_customers=1000, months=6, seed=42):
    """Simulate simple monthly card activity for a cohort of customers.

    Returns a DataFrame with one row per customer-month.
    """
    rng = np.random.default_rng(seed)
    customers = np.arange(1, n_customers + 1)
    rows = []
    for c in customers:
        # baseline customer characteristics
        credit_limit = rng.integers(1000, 10000)
        base_spend = max(100.0, float(rng.normal(1500, 500)))
        base_payment_pct = rng.uniform(0.2, 1.0)
        essentials_share = rng.uniform(0.2, 0.6)
        for m in range(months):
            # month-to-month variation
            spend = max(0, rng.normal(base_spend * (1 - 0.02 * m), base_spend * 0.2))
            num_txn = max(1, int(rng.poisson(25)))
            cash_adv = rng.choice([0, 0, 0, 50, 100, 200], p=[0.7,0.1,0.08,0.05,0.04,0.03])
            # simulate some customers reducing spend over time to create signals
            if rng.random() < 0.05 + 0.01*m:
                spend *= rng.uniform(0.4, 0.8)
            utilization = min(1.0, spend / credit_limit)
            balance = spend * rng.uniform(0.3, 1.0) + rng.normal(0, 50)
            # payment behavior
            # some pay full, some pay minimum, some degrade over time
            if rng.random() < 0.02 + 0.02*m:
                payment_pct = rng.uniform(0.01, 0.05)  # min due only
            else:
                payment_pct = base_payment_pct * rng.uniform(0.6, 1.1)
            payment_amt = max(0, payment_pct * (balance + spend))
            min_due = max(25, 0.02 * (balance + spend))
            if rng.random() < 0.01 + 0.005*m:
                # occasional declined transactions increase
                declined = rng.integers(1, 5)
            else:
                declined = 0
            merchant_essentials_ratio = min(1.0, essentials_share * rng.uniform(0.8, 1.3))
            payment_day = int(rng.integers(1, 28))

            rows.append({
                "customer_id": c,
                "month": m,
                "credit_limit": credit_limit,
                "spend": float(spend),
                "num_txn": int(num_txn),
                "cash_adv": float(cash_adv),
                "balance": float(balance),
                "payment_amt": float(payment_amt),
                "min_due": float(min_due),
                "utilization": float(utilization),
                "declined_count": int(declined),
                "essentials_ratio": float(merchant_essentials_ratio),
                "payment_day": int(payment_day),
            })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = simulate_customers(500, months=6)
    print(df.groupby('month').agg({'spend':'mean','payment_amt':'mean'}))
