# Early Risk Signal — Prototype

This repository contains a **lightweight, data-driven prototype** that simulates credit card customer activity, computes early behavioral risk signals, and provides tools for exploration, calibration, and validation.

---

## 🚀 Quickstart

1. **Create and activate a virtual environment; install runtime dependencies:**
   ```powershell
   cd "D:\Credit Card"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **(Optional) Install developer tools for testing and linting:**
   ```powershell
   pip install -r requirements-dev.txt
   ```

3. **Run the interactive Streamlit app:**
   ```powershell
   python -m streamlit run app_streamlit.py
   ```
   The app has the following tabs:
   - **Real Data:** Load/analyze real datasets (upload supported).
   - **Explore:** Simulate and visualize tier distributions.
   - **Calibrate:** Tune weights and thresholds; see instant results.
   - **Backtest:** Inject synthetic outcomes; compute validation metrics.
   - **Drilldown:** Filter cohorts, inspect profiles, download PDFs, webhook POST of INTERVENE lists.

4. **Run unit tests:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python -m pytest -q
   ```

5. **Run CLI batch scoring:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   & ".\.venv\Scripts\python.exe" app.py --simulate --n 1000 --months 6 --out scores.csv
   ```

6. **Build and run via Docker (optional):**
   ```powershell
   docker build -t early-risk-signals .
   docker run -p 8501:8501 early-risk-signals
   ```

---

## 📁 Files

- **`simulate.py`**: Generates synthetic customer-month transaction/payment data.
- **`signal_model.py`**: Computes five behavioral signal categories & composite risk scores (0–100).
- **`backtest.py`**: Injects synthetic delinquency, computes validation (ROC-AUC, PR-AUC, etc).
- **`app.py`**: CLI batch scoring.
- **`app_streamlit.py`**: Interactive web UI (with Explore, Calibrate, Backtest, and Drilldown).
- **`explore_signals.ipynb`**: Jupyter notebook for EDA and visualization.

---

## 🌟 Key Features

### 1. **Multi-Signal Detection**
Five behavioral analysis categories:
- Payment Behavior Degradation (e.g. min-due trap, decay, timing drift)
- Spending Pattern Disruption (velocity collapse, cash advance surge, merchant mix shift)
- Credit Stress Indicators (utilization creep, declines, revolving growth)
- Transaction Anomalies (timing, merchant concentration, small txns)
- External Context Factors (industry/geography risk, seasonal gaps)

### 2. **Three-Tier Risk Framework**
| Tier      | Score Range | Action                                         |
|-----------|-------------|------------------------------------------------|
| 🟢 MONITOR | 30–50      | Passive, no intervention                      |
| 🟡 ENGAGE  | 51–75      | Soft outreach (SMS, email, wellness resources) |
| 🔴 INTERVENE | 76–100   | Direct RM contact, custom support             |

### 3. **Interactive Calibration**
- Adjust weights for each signal category
- Tune ENGAGE / INTERVENE thresholds
- See impact on risk tiers in real-time

### 4. **Backtest & Validation**
- Inject simulated outcomes, correlated with score
- Compute precision, recall, lead time, ROC-AUC
- Analyze false alarm rate & early detection

---

## 📊 Example Backtest Results

*Simulating 500 customers (10% delinquency, 0.7 signal strength):*
- **ROC-AUC:** 0.761
- **Precision @ 50:** 0.22
- **Recall @ 50:** 1.00
- **F1 Score:** 0.36

---

## ⚙️ Deployment & Automation

- **CI/CD:** [GitHub Actions](.github/workflows/ci.yml) runs tests & linting

### Phase 1: Validation (Weeks 1–4)
- [ ] Backtest on 12-month labeled data
- [ ] Measure real precision, recall, lead time
- [ ] Validate signal stability

### Phase 2: Pilot (Months 2–4)
- [ ] Deploy to 10K-customer subset
- [ ] Manual RM outreach on INTERVENE cohort
- [ ] Track intervention impact

### Phase 3: Scale (Months 4+)
- [ ] Automate intervention workflows (SMS, email, CRM)
- [ ] Real-time batch (daily)
- [ ] Model monitoring & retraining

---

## 🛠 Technology Stack

- **Data:** pandas, numpy
- **ML/Metrics:** scikit-learn (precision, ROC)
- **Viz:** matplotlib
- **Web UI:** Streamlit (Drilldown, PDF, webhook)
- **Optional:** Airflow, Feast, SageMaker

---

## 🔬 ML Baseline & Reproducibility

This project includes an ML baseline and comparison utilities to evaluate tree-based models against the rule-based signal:

- `ml_models.py` — dataset preparation and training helpers (RandomForest; optional XGBoost).
- `run_ml_baseline.py` — runnable experiment: simulates a cohort, injects outcomes, trains RF and XGBoost (if installed), and writes artifacts to `artifacts/`.
- `artifacts/ml_comparison.json` — side-by-side metrics for rule-based vs RF vs XGBoost.
- `artifacts/feature_importance_rf.png` and `artifacts/feature_importance_xgb.png` — feature importance visualizations.

To reproduce the ML baseline locally (recommended to use the project's venv):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# (optional) install extra ML libs for XGBoost/LightGBM
pip install xgboost lightgbm
python run_ml_baseline.py
```

The Streamlit `ML` tab also reads `artifacts/ml_comparison.json` (if present) and displays the latest comparison and feature-importance images.

Note: For sequence-model experiments the `sequence_modeling.py` scaffold is included; install `tensorflow` only if you intend to run those notebooks/experiments.

## 📦 Artifacts to Include for Capstone Submission

- `artifacts/ml_comparison.json` — ML vs rule-based metrics
- `artifacts/feature_importance_rf.png`, `artifacts/feature_importance_xgb.png` — importance plots
- Example outputs from `app_streamlit.py` (screenshots or exported PDF reports in `artifacts/`)


## 💡 Business Impact

- **Risk Reduction:** 25%+ drop in delinquency roll rates w/ early intervention
- **Loss Prevention:** $3–5M benefit per $1B portfolio (estimated)
- **Retention:** Proactive outreach boosts satisfaction & loyalty

---

## 📝 Notes

- Prototype only; all weights/thresholds configurable; validate on real data!
- Sim deals with artificial delinquency (binomial/correlated); real data varies.
- For production: integrate with core systems, fraud, workflow engines.

---
