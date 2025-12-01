## Quick Start: Using Real Data

Your dataset has been successfully integrated! Here's how to use it:

### Run the Streamlit App

```powershell
pip install -r requirements.txt
streamlit run app_streamlit.py
```

The app now has **4 tabs**:

#### 1. **Real Data** (New!)
- Automatically loads `credit_card_data.csv`
- Shows tier distribution (MONITOR/ENGAGE/INTERVENE)
- Visualizes signal relationships:
  - Utilization vs Composite Score
  - Payment Ratio vs Composite Score
  - Cash Withdrawal vs Composite Score
- Lists flagged high-risk accounts
- **Outcome Analysis**: Shows how model scores correlate with actual DPD outcomes
- Download scored data

#### 2. **Explore**
- Simulate synthetic data
- See tier distributions
- Inspect flagged cohorts

#### 3. **Calibrate**
- Adjust signal weights
- Tune ENGAGE/INTERVENE thresholds
- See real-time impact

#### 4. **Backtest**
- Inject synthetic delinquency outcomes
- Measure precision, recall, ROC-AUC
- View confusion matrices

---

### Key Findings from Your Dataset

With your 100 customers:

| Metric | Value |
|--------|-------|
| Total Customers | 100 |
| MONITOR Tier | 52 |
| ENGAGE Tier | 46 |
| INTERVENE Tier | 2 |
| Delinquent (DPD >= 1) | 22 |
| On-Time (DPD = 0) | 78 |
| Mean Score (Delinquent) | 54.50 |
| Mean Score (On-Time) | 50.53 |

**Signal Strength**: The model shows a ~4 point score difference between delinquent and on-time cohorts, indicating moderate predictive power that will improve with calibration.

---

### Signal Mapping for Your Dataset

Your columns map to signal categories as follows:

| Category | Real Columns | Signal Logic |
|----------|--------------|--------------|
| **Payment Behavior (40%)** | Avg Payment Ratio, Min Due Frequency | Lower ratio / frequency = higher risk |
| **Spending Pattern (30%)** | Recent Spend Change %, Cash Withdrawal % | Negative spend decline, high cash withdrawals = risk |
| **Credit Stress (30%)** | Utilisation %, Credit Limit | High utilization, low limit = risk |
| **Anomalies (15%)** | Merchant Mix Index | High concentration = risk |
| **External (5%)** | Very high utilization | Extreme stress signal |

---

### Next Steps

1. **Review flagged accounts**: Look at the "Real Data" tab to see which customers are flagged as ENGAGE/INTERVENE
2. **Validate with your team**: Confirm that high-scoring customers align with your risk intuition
3. **Calibrate thresholds**: Use the "Calibrate" tab to adjust weights and see impact
4. **Deploy**: When satisfied, export scored data and integrate with your outreach systems

---

### File Structure

```
Credit Card/
├── app_streamlit.py           # Main UI (4 tabs)
├── credit_card_data.csv       # Your dataset
├── data_loader.py             # Real data handling
├── signal_model.py            # Signal scoring logic
├── backtest.py                # Validation metrics
├── simulate.py                # Synthetic data (for testing)
├── app.py                      # CLI batch scoring
├── explore_signals.ipynb       # Jupyter notebook
└── requirements.txt
```

---

### Command Line Usage

Score your dataset from command line:

```powershell
python -c "from data_loader import load_real_data, compute_real_data_signals; df = load_real_data('credit_card_data.csv'); scores = compute_real_data_signals(df); scores.to_csv('real_scores_output.csv', index=False); print('Saved to real_scores_output.csv')"
```

Or use the Python API directly:

```python
from data_loader import load_real_data, compute_real_data_signals

df = load_real_data('credit_card_data.csv')
scores = compute_real_data_signals(df)
print(scores[['customer_id', 'composite_score', 'tier']])
```

---

### Questions?

- **How are scores calculated?** See `data_loader.py` for the signal composition logic
- **Can I modify weights?** Yes, use the "Calibrate" tab in Streamlit
- **How do I measure accuracy?** Use the "Backtest" tab to run ROC/precision-recall analysis
- **Can I upload my own data?** Yes, the "Real Data" tab supports custom CSV uploads (must have same columns)

