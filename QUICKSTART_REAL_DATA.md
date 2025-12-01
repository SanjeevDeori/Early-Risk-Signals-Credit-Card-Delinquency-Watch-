## Quick Start: Using Real Data

Your dataset has been successfully integrated! Here's how to use it and the updated app features.

### Setup & Run

```powershell
cd "D:\Credit Card"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the Streamlit app from the venv:

```powershell
python -m streamlit run app_streamlit.py
```

### Tabs & Features

- **Real Data**: Load `credit_card_data.csv` or upload your CSV. View tier distributions, signal relationships, and download scored data.
- **Explore**: Simulate cohorts and inspect flagged accounts.
- **Calibrate**: Tune weights and thresholds interactively.
- **Backtest**: Inject synthetic outcomes and compute ROC/PR metrics and confusion matrices.
- **Drilldown**: Filter cohorts, inspect individual customer profiles, download profile PDFs, and send INTERVENE lists to a webhook.

### Useful Commands

Score your dataset from the command line:

```powershell
python -c "from data_loader import load_real_data, compute_real_data_signals; df = load_real_data('credit_card_data.csv'); scores = compute_real_data_signals(df); scores.to_csv('real_scores_output.csv', index=False); print('Saved to real_scores_output.csv')"
```

Run unit tests (after installing dev requirements):

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

### Notes

- To test webhook delivery locally, run a small HTTP listener (Flask or ngrok) and paste the URL into the Drilldown webhook input.
- PDF profile exports are available per-customer in the Drilldown tab.


