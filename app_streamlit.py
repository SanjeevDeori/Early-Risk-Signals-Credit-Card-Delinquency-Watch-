"""Enhanced Streamlit prototype for Early Risk Signal exploration.

Run with:
  streamlit run app_streamlit.py

Features:
  - Real Data: Load and analyze real credit card dataset
  - Explore: Simulate data and visualize cohorts
  - Calibrate: Tune weights and thresholds
  - Backtest: Inject outcomes and measure model performance
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import io
import requests
from matplotlib.backends.backend_pdf import PdfPages
import os
import json

from simulate import simulate_customers
from signal_model import compute_customer_signals, DEFAULT_WEIGHTS
from backtest import inject_outcomes, compute_backtest_metrics
from data_loader import load_real_data, compute_real_data_signals
from ml_models import prepare_ml_dataset, train_random_forest, try_train_xgboost


@st.cache_data
def run_simulation(n_customers, months, seed):
    df = simulate_customers(n_customers=n_customers, months=months, seed=seed)
    scores = compute_customer_signals(df)
    return df, scores


def tier_with_thresholds(composite_score, t1, t2):
    """Assign tier based on custom thresholds."""
    if composite_score >= t2:
        return 'INTERVENE'
    elif composite_score >= t1:
        return 'ENGAGE'
    else:
        return 'MONITOR'


def main():
    st.set_page_config(page_title='Early Risk Signals', layout='wide', page_icon='🛡️')

    # Header / Branding
    st.markdown("""
    <style>
    .dashboard-title {font-size:30px; font-weight:700; color:#dbe9f4}
    .dashboard-sub {color:#9fb6d6; margin-top:4px}
    .card {background:rgba(255,255,255,0.03); padding:12px; border-radius:8px}
    </style>
    """, unsafe_allow_html=True)

    header_col, spacer_col, meta_col = st.columns([6, 1, 3])
    with header_col:
        st.markdown('<div class="dashboard-title">Early Risk Signals — Credit Card Delinquency Watch</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-sub">Lightweight behavioral signals to surface early risk and enable proactive outreach</div>', unsafe_allow_html=True)
    with meta_col:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write('**Quick Preview**')
            if st.button('Run Quick Preview (300 customers)', key='quick_preview'):
                with st.spinner('Simulating preview cohort...'):
                    df_prev = simulate_customers(n_customers=300, months=6, seed=123)
                    scores_prev = compute_customer_signals(df_prev)
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric('Total', len(scores_prev))
                    mc2.metric('ENGAGE', int((scores_prev['tier'] == 'ENGAGE').sum()))
                    mc3.metric('INTERVENE', int((scores_prev['tier'] == 'INTERVENE').sum()))
                    # store last run for drilldown
                    st.session_state['last_scores'] = scores_prev
                    st.session_state['last_df'] = df_prev
            else:
                st.write('Click to simulate a quick preview cohort')
            st.markdown('</div>', unsafe_allow_html=True)

    tab_real, tab1, tab2, tab3, tab_ml, tab_drilldown = st.tabs(['Real Data', 'Explore', 'Calibrate', 'Backtest', 'ML', 'Drilldown'])

    # ===== TAB 0: REAL DATA =====
    with tab_real:
        st.subheader('Real Credit Card Dataset Analysis')
        st.write('Load and analyze your real credit card delinquency dataset.')
        
        # Upload or use default
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader('Upload CSV', type=['csv', 'txt'])
        with col2:
            use_default = st.checkbox('Use default dataset (credit_card_data.csv)', value=True)

        filepath = None
        # Prefer repository copy of credit_card_data.csv; fall back to bundled sample
        if use_default:
            if os.path.exists('credit_card_data.csv'):
                filepath = 'credit_card_data.csv'
            elif os.path.exists('data/sample_credit_card_data.csv'):
                filepath = 'data/sample_credit_card_data.csv'
                st.warning('Default `credit_card_data.csv` not found — using bundled example dataset.')
            else:
                filepath = None
                st.warning('No default dataset found. Please upload a CSV or download the example.')
                # Offer the sample for download even if not present locally
                try:
                    with open('data/sample_credit_card_data.csv', 'rb') as fh:
                        sample_bytes = fh.read()
                    st.download_button('Download example CSV', data=sample_bytes, file_name='sample_credit_card_data.csv', mime='text/tab-separated-values')
                except Exception:
                    st.info('Example CSV not bundled; please upload your dataset.')
        elif uploaded_file:
            filepath = uploaded_file
        
        if filepath and st.button('Load & Analyze Real Data', key='real_data_run'):
            try:
                if isinstance(filepath, str):
                    df = load_real_data(filepath)
                else:
                    # Handle uploaded file
                    df = pd.read_csv(filepath, sep='\t')
                    df.columns = df.columns.str.strip()
                    column_mapping = {
                        'Customer ID': 'customer_id', 'Credit Limit': 'credit_limit',
                        'Utilisation %': 'utilization', 'Avg Payment Ratio': 'avg_payment_ratio',
                        'Min Due Paid Frequency': 'min_due_frequency', 'Merchant Mix Index': 'merchant_mix_index',
                        'Cash Withdrawal %': 'cash_withdrawal_pct', 'Recent Spend Change %': 'spend_change_pct',
                        'DPD Bucket Next Month': 'dpd_bucket_next_month',
                    }
                    df = df.rename(columns=column_mapping)
                
                with st.spinner('Computing risk signals...'):
                    scores = compute_real_data_signals(df)
                    # store for drilldown
                    st.session_state['last_scores'] = scores
                    st.session_state['last_df'] = df
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Total Customers', len(scores))
                col2.metric('MONITOR', len(scores[scores['tier'] == 'MONITOR']))
                col3.metric('ENGAGE', len(scores[scores['tier'] == 'ENGAGE']))
                col4.metric('INTERVENE', len(scores[scores['tier'] == 'INTERVENE']))
                
                # Visualizations
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader('Tier Distribution')
                    order = ['INTERVENE', 'ENGAGE', 'MONITOR']
                    counts = scores['tier'].value_counts().reindex(order).fillna(0).reset_index()
                    counts.columns = ['tier', 'count']
                    chart = alt.Chart(counts).mark_bar().encode(
                        y=alt.Y('tier:N', sort=order, title='Tier'),
                        x=alt.X('count:Q', title='Count'),
                        color=alt.Color('tier:N', scale=alt.Scale(domain=order, range=['#d7191c', '#fdae61', '#1a9641'])),
                        tooltip=['tier:N', 'count:Q']
                    ).properties(width=450, height=200)
                    st.altair_chart(chart, use_container_width=True)

                with col_right:
                    st.subheader('Score Distribution')
                    hist = alt.Chart(scores).mark_bar().encode(
                        alt.X('composite_score:Q', bin=alt.Bin(maxbins=30), title='Composite Score'),
                        y='count():Q',
                        tooltip=[alt.Tooltip('count():Q', title='Count')]
                    ).properties(width=450, height=200)
                    vlines = alt.Chart(pd.DataFrame({'x':[50,76],'label':['ENGAGE','INTERVENE']})).mark_rule(strokeDash=[4,4]).encode(x='x:Q', color=alt.Color('label:N', scale=alt.Scale(domain=['ENGAGE','INTERVENE'], range=['orange','red'])))
                    st.altair_chart((hist + vlines).interactive(), use_container_width=True)
                
                # Signal relationships
                st.subheader('Signal Relationships')
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    chart = alt.Chart(scores).mark_circle(size=60).encode(
                        x=alt.X('utilization:Q', title='Utilization'),
                        y=alt.Y('composite_score:Q', title='Composite Score'),
                        color=alt.Color('composite_score:Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=['customer_id', alt.Tooltip('composite_score:Q', format='.2f'), alt.Tooltip('utilization:Q', format='.2f')]
                    ).properties(width=300, height=240).interactive()
                    st.altair_chart(chart, use_container_width=True)
                
                with col2:
                    chart = alt.Chart(scores).mark_circle(size=60).encode(
                        x=alt.X('avg_payment_ratio:Q', title='Avg Payment Ratio'),
                        y=alt.Y('composite_score:Q', title='Composite Score'),
                        color=alt.Color('composite_score:Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=['customer_id', alt.Tooltip('composite_score:Q', format='.2f'), alt.Tooltip('avg_payment_ratio:Q', format='.2f')]
                    ).properties(width=300, height=240).interactive()
                    st.altair_chart(chart, use_container_width=True)
                
                with col3:
                    chart = alt.Chart(scores).mark_circle(size=60).encode(
                        x=alt.X('cash_withdrawal_pct:Q', title='Cash Withdrawal %'),
                        y=alt.Y('composite_score:Q', title='Composite Score'),
                        color=alt.Color('composite_score:Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=['customer_id', alt.Tooltip('composite_score:Q', format='.2f'), alt.Tooltip('cash_withdrawal_pct:Q', format='.2f')]
                    ).properties(width=300, height=240).interactive()
                    st.altair_chart(chart, use_container_width=True)
                
                # High-risk cohorts
                st.subheader('High-Risk Flagged Accounts')
                high_risk = scores[scores['tier'].isin(['INTERVENE', 'ENGAGE'])].sort_values('composite_score', ascending=False)
                st.dataframe(
                    high_risk[['customer_id', 'composite_score', 'tier', 'utilization', 'avg_payment_ratio',
                              'cash_withdrawal_pct', 'spend_change_pct']].head(30),
                    use_container_width=True
                )
                
                # Outcome analysis if available
                if 'dpd_bucket_next_month' in scores.columns:
                    with st.expander('Outcome Analysis (if labels available)'):
                        st.subheader('Score vs Actual DPD Outcome')
                        fig, ax = plt.subplots(figsize=(8, 5))
                        for dpd in sorted(scores['dpd_bucket_next_month'].unique()):
                            mask = scores['dpd_bucket_next_month'] == dpd
                            ax.scatter(scores[mask]['composite_score'], scores[mask]['dpd_bucket_next_month'],
                                     label=f'DPD={dpd}', alpha=0.6, s=50)
                        ax.set_xlabel('Composite Score')
                        ax.set_ylabel('DPD Bucket Next Month')
                        ax.set_title('Model Score vs Actual Outcome')
                        ax.legend()
                        ax.grid(alpha=0.2)
                        st.pyplot(fig)
                        
                        # Predictive power
                        delinquent = scores['dpd_bucket_next_month'] >= 1
                        st.write(f"Delinquent customers (DPD ≥ 1): {delinquent.sum()} / {len(scores)}")
                        st.write(f"Mean score of delinquent: {scores[delinquent]['composite_score'].mean():.2f}")
                        st.write(f"Mean score of on-time: {scores[~delinquent]['composite_score'].mean():.2f}")
                
                # Download
                csv = scores.to_csv(index=False).encode('utf-8')
                st.download_button('📥 Download Scored Data', csv, file_name='real_data_scores.csv', mime='text/csv')
                
            except Exception as e:
                st.error(f'Error loading data: {str(e)}')

    # ===== TAB 1: EXPLORE =====
    with tab1:
        st.subheader('Simulate and Explore Cohorts')
        col1, col2, col3 = st.columns(3)
        n = col1.slider('Customers', 100, 5000, 1000, step=100)
        months = col2.slider('Months', 2, 12, 6)
        seed = col3.number_input('Seed', value=123)

        if st.button('Run Simulation', key='explore_run'):
            with st.spinner('Simulating...'):
                df, scores = run_simulation(n, months, seed)
                # persist for drilldown
                st.session_state['last_scores'] = scores
                st.session_state['last_df'] = df

            # Tier distribution
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader('Tier Distribution')
                order = ['INTERVENE', 'ENGAGE', 'MONITOR']
                counts = scores['tier'].value_counts().reindex(order).fillna(0)
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(kind='barh', color=['#d7191c', '#fdae61', '#1a9641'], ax=ax)
                ax.set_xlabel('Count')
                st.pyplot(fig)

            with col_right:
                st.subheader('Score Distribution')
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(scores['composite_score'], bins=30, color='#2b8cbe', edgecolor='black')
                ax.set_xlabel('Composite Score')
                ax.set_ylabel('Count')
                ax.grid(alpha=0.2)
                st.pyplot(fig)

            # Multi-signal scatter
            st.subheader('Signal Relationships')
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(
                    scores['utilization'], scores['composite_score'],
                    c=scores['composite_score'], cmap='viridis', alpha=0.6, s=30
                )
                ax.set_xlabel('Utilization')
                ax.set_ylabel('Composite Score')
                ax.set_title('Utilization vs Score')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(
                    scores['velocity_decline_pct'], scores['composite_score'],
                    c=scores['composite_score'], cmap='viridis', alpha=0.6, s=30
                )
                ax.set_xlabel('Spend Velocity Decline %')
                ax.set_ylabel('Composite Score')
                ax.set_title('Spend Decline vs Score')
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)

            # Top flagged
            st.subheader('Top Flagged Accounts (INTERVENE tier)')
            top_flagged = scores[scores['tier'] == 'INTERVENE'].sort_values('composite_score', ascending=False).head(20)
            st.dataframe(top_flagged, use_container_width=True)

            # Download
            csv = scores.to_csv(index=False).encode('utf-8')
            st.download_button('📥 Download Scores CSV', csv, file_name='scores.csv', mime='text/csv')

    # ===== TAB 2: CALIBRATE =====
    with tab2:
        st.subheader('Tune Weights and Thresholds')

        # Load or use defaults
        with st.expander('Signal Weights', expanded=True):
            col1, col2 = st.columns(2)
            weights = {}
            weights['payment'] = col1.slider('Payment Behavior', 0.0, 1.0, DEFAULT_WEIGHTS['payment'], 0.01)
            weights['spending'] = col2.slider('Spending Pattern', 0.0, 1.0, DEFAULT_WEIGHTS['spending'], 0.01)
            weights['credit_stress'] = col1.slider('Credit Stress', 0.0, 1.0, DEFAULT_WEIGHTS['credit_stress'], 0.01)
            weights['anomaly'] = col2.slider('Anomaly', 0.0, 1.0, DEFAULT_WEIGHTS['anomaly'], 0.01)
            weights['external'] = col1.slider('External Factors', 0.0, 1.0, DEFAULT_WEIGHTS['external'], 0.01)

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
                st.info(f'Normalized weights sum to 1.0')

        with st.expander('Tier Thresholds', expanded=True):
            col1, col2 = st.columns(2)
            engage_threshold = col1.slider('ENGAGE threshold', 0, 100, 51)
            intervene_threshold = col2.slider('INTERVENE threshold', engage_threshold, 100, 76)

        if st.button('Apply Calibration', key='calibrate_run'):
            with st.spinner('Recalculating...'):
                n = st.session_state.get('n', 1000)
                months = st.session_state.get('months', 6)
                seed = st.session_state.get('seed', 123)
                df, _ = run_simulation(n, months, seed)
                scores = compute_customer_signals(df, weights=weights)

            # store calibrated scores for drilldown
            st.session_state['last_scores'] = scores
            st.session_state['last_df'] = df

            # Apply custom thresholds
            scores['tier'] = scores['composite_score'].apply(
                lambda x: tier_with_thresholds(x, engage_threshold, intervene_threshold)
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('New Tier Distribution')
                order = ['INTERVENE', 'ENGAGE', 'MONITOR']
                counts = scores['tier'].value_counts().reindex(order).fillna(0)
                fig, ax = plt.subplots(figsize=(6, 4))
                counts.plot(kind='bar', color=['#d7191c', '#fdae61', '#1a9641'], ax=ax)
                ax.set_ylabel('Count')
                st.pyplot(fig)

            with col2:
                st.metric('MONITOR', int(counts.get('MONITOR', 0)))
                st.metric('ENGAGE', int(counts.get('ENGAGE', 0)))
                st.metric('INTERVENE', int(counts.get('INTERVENE', 0)))

            csv = scores.to_csv(index=False).encode('utf-8')
            st.download_button('📥 Download Calibrated Scores', csv, file_name='scores_calibrated.csv', mime='text/csv')

    # ===== TAB 3: BACKTEST =====
    with tab3:
        st.subheader('Backtest Model Performance')

        col1, col2, col3 = st.columns(3)
        n = col1.number_input('Customers', 200, 5000, 500)
        delinq_rate = col2.slider('Delinquency Rate', 0.01, 0.20, 0.05, 0.01)
        signal_strength = col3.slider('Signal Strength', 0.0, 1.0, 0.7, 0.1)

        if st.button('Run Backtest', key='backtest_run'):
            with st.spinner('Running backtest...'):
                df, scores = run_simulation(n, 6, 123)
                # store for drilldown
                st.session_state['last_scores'] = scores
                st.session_state['last_df'] = df
                outcomes = inject_outcomes(scores, delinquency_rate=delinq_rate, signal_strength=signal_strength)
                metrics = compute_backtest_metrics(outcomes)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric('PR-AUC', f"{metrics['pr_auc']:.3f}")
            col2.metric('ROC-AUC', f"{metrics['roc_auc']:.3f}")
            col3.metric('Precision @ 50', f"{metrics['precision_at_50']:.3f}")
            col4.metric('Recall @ 50', f"{metrics['recall_at_50']:.3f}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Precision-Recall Curve')
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(metrics['recall'], metrics['precision'], marker='o', label=f"PR-AUC={metrics['pr_auc']:.3f}")
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.legend()
                ax.grid(alpha=0.2)
                st.pyplot(fig)

            with col2:
                st.subheader('ROC Curve')
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(metrics['fpr'], metrics['tpr'], marker='o', label=f"ROC-AUC={metrics['roc_auc']:.3f}")
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                ax.grid(alpha=0.2)
                st.pyplot(fig)

            # Confusion matrix
            st.subheader('Confusion Matrix @ Threshold 50')
            confusion_data = {
                'Predicted Negative': [metrics['true_negatives'], metrics['false_positives']],
                'Predicted Positive': [metrics['false_negatives'], metrics['true_positives']],
            }
            cm_df = pd.DataFrame(confusion_data, index=['Actual Negative', 'Actual Positive'])
            st.dataframe(cm_df)

            # Summary
            st.subheader('Summary')
            st.write(f"""
            - **Flagged (score ≥ 50)**: {metrics['flagged_count']} customers
            - **Actually Delinquent**: {metrics['delinquent_count']} customers
            - **True Positives**: {metrics['true_positives']} (caught early)
            - **False Positives**: {metrics['false_positives']} (false alarm)
            - **F1 Score @ 50**: {metrics['f1_at_50']:.3f}
            """)

    # ===== TAB X: ML Baseline =====
    with tab_ml:
        st.subheader('ML Baseline: RandomForest vs Rule-based')
        st.write('Train a Random Forest baseline on a simulated cohort and compare against the rule-based signals.')

        ml_sim_n = st.number_input('Cohort size for ML baseline', min_value=200, max_value=5000, value=2000)

        # If a previous run exists, show it immediately for quick inspection
        comp_path = 'artifacts/ml_comparison.json'
        rf_img = 'artifacts/feature_importance_rf.png'
        xgb_img = 'artifacts/feature_importance_xgb.png'
        if os.path.exists(comp_path):
            try:
                with open(comp_path, 'r') as fh:
                    comp = json.load(fh)
                st.markdown('**Latest ML vs Rule-based Comparison (from artifacts/ml_comparison.json)**')
                cols = st.columns(3)
                if comp.get('rule_based') is not None:
                    cols[0].metric('Rule ROC-AUC', f"{comp['rule_based'].get('roc_auc', 0):.3f}")
                if comp.get('random_forest') is not None:
                    cols[1].metric('RF ROC-AUC', f"{comp['random_forest'].get('roc_auc', 0):.3f}")
                if comp.get('xgboost') is not None:
                    cols[2].metric('XGB ROC-AUC', f"{comp['xgboost'].get('roc_auc', 0):.3f}")

                st.write('Full comparison JSON:')
                st.json(comp)
            except Exception as e:
                st.error(f'Error reading comparison JSON: {e}')

        if st.button('Run ML Baseline', key='ml_run'):
            with st.spinner('Training Random Forest...'):
                df_ml = simulate_customers(ml_sim_n, months=6, seed=123)
                scores_ml = compute_customer_signals(df_ml)

                X, y, full = prepare_ml_dataset(scores_ml, delinquency_rate=0.05, signal_strength=0.7, seed=42)
                try:
                    rf_res = train_random_forest(X, y)
                except Exception as e:
                    st.error(f'Error training RandomForest: {e}')
                    rf_res = None

                xgb_res = try_train_xgboost(X, y)

                # Compute rule-based metrics
                try:
                    rule_metrics = compute_backtest_metrics(full)
                except Exception as e:
                    st.error(f'Error computing rule-based metrics: {e}')
                    rule_metrics = None

                # Compose comparison
                comparison = {'rule_based': None, 'random_forest': None, 'xgboost': None}
                if rule_metrics is not None:
                    comparison['rule_based'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rule_metrics.items()}
                if rf_res is not None:
                    comparison['random_forest'] = rf_res['metrics']
                if xgb_res is not None:
                    comparison['xgboost'] = xgb_res['metrics']

                # Save comparison
                os.makedirs('artifacts', exist_ok=True)
                with open('artifacts/ml_comparison.json', 'w') as fh:
                    json.dump(comparison, fh, indent=2)

                st.success('ML baseline run complete — comparison saved to artifacts/ml_comparison.json')

                # Display summary metrics
                cols = st.columns(3)
                if rule_metrics is not None:
                    cols[0].metric('Rule-based ROC-AUC', f"{rule_metrics.get('roc_auc', 0):.3f}")
                if rf_res is not None:
                    cols[1].metric('RF ROC-AUC', f"{rf_res['metrics']['roc_auc']:.3f}")
                if xgb_res is not None:
                    cols[2].metric('XGB ROC-AUC', f"{xgb_res['metrics']['roc_auc']:.3f}")

                # Show RF feature importance image if available
                rf_img = 'artifacts/feature_importance_rf.png'
                xgb_img = 'artifacts/feature_importance_xgb.png'
                if os.path.exists(rf_img):
                    st.image(rf_img, caption='RF Feature Importance', use_column_width=True)
                if os.path.exists(xgb_img):
                    st.image(xgb_img, caption='XGBoost Feature Importance', use_column_width=True)

                with open('artifacts/ml_comparison.json', 'rb') as fh:
                    st.download_button('Download ML comparison JSON', fh.read(), file_name='ml_comparison.json', mime='application/json')

    # ===== TAB 4: DRILLDOWN =====
    with tab_drilldown:
        st.subheader('Customer Drill-down & Filters')

        if 'last_scores' not in st.session_state:
            st.info('No scored cohort available yet. Run a simulation, quick preview, or load real data to enable drill-down.')
            if st.button('Run Quick Preview (300)', key='dd_quick'):
                df_prev = simulate_customers(n_customers=300, months=6, seed=123)
                scores_prev = compute_customer_signals(df_prev)
                st.session_state['last_scores'] = scores_prev
                st.session_state['last_df'] = df_prev
                st.experimental_rerun()
        else:
            scores = st.session_state['last_scores']
            df = st.session_state.get('last_df', None)

            # Filters
            cols = st.columns([2, 2, 2, 4])
            with cols[0]:
                tier_filter = st.multiselect('Tiers', options=['MONITOR', 'ENGAGE', 'INTERVENE'], default=['INTERVENE','ENGAGE','MONITOR'])
            with cols[1]:
                score_min, score_max = st.slider('Score range', 0, 100, (0, 100))
            with cols[2]:
                top_n = st.number_input('Top N', min_value=5, max_value=200, value=20)
            with cols[3]:
                search_id = st.text_input('Search Customer ID')

            mask = scores['tier'].isin(tier_filter) & scores['composite_score'].between(score_min, score_max)
            if search_id:
                try:
                    sid = int(search_id)
                    mask = mask & (scores['customer_id'] == sid)
                except ValueError:
                    mask = mask & (scores['customer_id'].astype(str).str.contains(search_id))

            filtered = scores[mask].sort_values('composite_score', ascending=False).head(int(top_n))
            st.subheader('Filtered Cohort')
            st.dataframe(filtered[['customer_id','composite_score','tier','utilization']].reset_index(drop=True), use_container_width=True)

            # Select a customer for profile
            cust_ids = filtered['customer_id'].tolist()
            if cust_ids:
                selected = st.selectbox('Select customer for profile', options=cust_ids)
                profile = scores[scores['customer_id'] == selected].iloc[0]

                st.markdown('**Customer Profile Summary**')
                pcol1, pcol2, pcol3 = st.columns(3)
                pcol1.metric('Customer ID', profile['customer_id'])
                pcol2.metric('Composite Score', f"{profile['composite_score']:.1f}")
                pcol3.metric('Tier', profile['tier'])

                # show component scores if present
                comps = {k: profile[k] for k in ['payment_score','spending_score','credit_stress_score','anomaly_score'] if k in profile.index}
                if comps:
                    st.subheader('Signal Breakdown')
                    for k, v in comps.items():
                        st.write(f"- **{k.replace('_',' ').title()}**: {v:.2f}")

                # Time series if detailed df available
                if df is not None and 'month' in df.columns:
                    cust_hist = df[df['customer_id'] == selected].sort_values('month')
                    if not cust_hist.empty:
                        st.subheader('Customer Activity (over months)')
                        fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
                        ax[0].plot(cust_hist['month'], cust_hist['spend'], marker='o', label='Spend')
                        ax[0].plot(cust_hist['month'], cust_hist['payment_amt'], marker='x', label='Payment Amt')
                        ax[0].set_ylabel('Amount')
                        ax[0].legend()
                        ax[1].plot(cust_hist['month'], cust_hist['utilization'], marker='s', color='orange', label='Utilization')
                        ax[1].set_xlabel('Month')
                        ax[1].set_ylabel('Utilization')
                        ax[1].legend()
                        st.pyplot(fig)

                        csv = cust_hist.to_csv(index=False).encode('utf-8')
                        st.download_button('📥 Download Customer History', csv, file_name=f'customer_{selected}_history.csv', mime='text/csv')

                # Recommendations
                st.subheader('Recommended Intervention')
                if profile['tier'] == 'INTERVENE':
                    st.markdown('- Immediate RM outreach: call and offer payment plan or hardship assistance')
                    st.markdown('- Consider temporary credit limit increase or restructure')
                elif profile['tier'] == 'ENGAGE':
                    st.markdown('- Soft outreach: SMS/email with payment reminder and financial resources')
                    st.markdown('- Monitor next 30 days; consider targeted offers')
                else:
                    st.markdown('- Passive monitoring; no immediate action')

                # PDF export for profile
                def make_profile_pdf(profile, cust_hist=None):
                    buffer = io.BytesIO()
                    with PdfPages(buffer) as pdf:
                        # Page 1: Summary text
                        fig = plt.figure(figsize=(8.27, 11.69))  # A4
                        fig.clf()
                        txt = fig.text(0.1, 0.9, f"Customer Profile Report - ID: {profile['customer_id']}", fontsize=14)
                        lines = [
                            f"Composite Score: {profile['composite_score']:.2f}",
                            f"Tier: {profile['tier']}",
                        ]
                        # component scores
                        for k in ['payment_score', 'spending_score', 'credit_stress_score', 'anomaly_score']:
                            if k in profile.index:
                                lines.append(f"{k.replace('_',' ').title()}: {profile[k]:.2f}")

                        for i, line in enumerate(lines):
                            fig.text(0.1, 0.85 - i*0.04, line, fontsize=10)

                        pdf.savefig(fig)
                        plt.close(fig)

                        # Page 2: Time series plots if available
                        if cust_hist is not None and not cust_hist.empty:
                            fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), sharex=True)
                            ax[0].plot(cust_hist['month'], cust_hist['spend'], marker='o', label='Spend')
                            ax[0].plot(cust_hist['month'], cust_hist['payment_amt'], marker='x', label='Payment Amt')
                            ax[0].set_ylabel('Amount')
                            ax[0].legend()
                            ax[1].plot(cust_hist['month'], cust_hist['utilization'], marker='s', color='orange', label='Utilization')
                            ax[1].set_xlabel('Month')
                            ax[1].set_ylabel('Utilization')
                            ax[1].legend()
                            pdf.savefig(fig)
                            plt.close(fig)

                    buffer.seek(0)
                    return buffer.read()

                if st.button('📄 Download Profile PDF'):
                    cust_hist = None
                    if df is not None and ('month' in df.columns):
                        cust_hist = df[df['customer_id'] == selected].sort_values('month')
                    pdf_bytes = make_profile_pdf(profile, cust_hist)
                    st.download_button('Download PDF', data=pdf_bytes, file_name=f'customer_{selected}_profile.pdf', mime='application/pdf')

                # Alerting webhook
                st.subheader('Alerting / Export')
                webhook_url = st.text_input('Webhook URL (optional)', value='')
                if st.button('Send INTERVENE List to Webhook'):
                    intervene_list = scores[scores['tier'] == 'INTERVENE'].sort_values('composite_score', ascending=False)
                    payload = {
                        'source': 'early-risk-signals-prototype',
                        'count': int(len(intervene_list)),
                        'top_customers': intervene_list.head(50)[['customer_id','composite_score','utilization']].to_dict(orient='records')
                    }
                    if not webhook_url:
                        st.error('Please provide a webhook URL to send alerts.')
                    else:
                        try:
                            resp = requests.post(webhook_url, json=payload, timeout=10)
                            if resp.status_code >= 200 and resp.status_code < 300:
                                st.success(f'Successfully POSTed {len(intervene_list)} INTERVENE customers to webhook')
                            else:
                                st.error(f'Webhook returned status {resp.status_code}: {resp.text}')
                        except Exception as e:
                            st.error(f'Error sending webhook: {e}')



if __name__ == '__main__':
    main()

