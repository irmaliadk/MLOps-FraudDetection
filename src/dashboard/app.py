import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Fraud Detection System Dashboard")
st.caption("Real-time crypto transaction fraud detection — Kraken XBTUSD")
st.divider()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    st.info("**Data Source:** Kraken Public API\n\n**Symbol:** XBTUSD\n\n**Model:** RandomForest\n\n**F1 Score:** 0.8660")
    st.divider()
    st.header("🔗 Links")
    st.markdown("- [GitHub Repo](https://github.com/irmaliadk/MLOps-FraudDetection)")
    st.markdown("- [DagsHub MLflow](https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow)")
    st.markdown("- [Grafana Dashboard](http://localhost:3000)")

# ─── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Prediction",
    "📊 Data Overview",
    "🔬 Drift Detection",
    "📈 Model Performance"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🎯 Real-time Transaction Fraud Prediction")
    st.write("Masukkan data transaksi untuk diprediksi apakah fraud atau legitimate.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (USD)", min_value=0.0, value=78500.0, step=100.0)
        volume = st.number_input("Volume (BTC)", min_value=0.0, value=0.005, step=0.001, format="%.6f")
    with col2:
        hour    = st.slider("Hour (WIB)", 0, 23, 14)
        minute  = st.slider("Minute", 0, 59, 30)
        is_sell = st.selectbox("Transaction Type", options=[0, 1], format_func=lambda x: "Buy" if x == 0 else "Sell")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        try:
            scaler_amount = joblib.load("models/scalers/scaler_amount.pkl")
            scaler_volume = joblib.load("models/scalers/scaler_volume.pkl")
            model         = joblib.load("models/trained/fraud_model.pkl")

            amount_scaled = float(scaler_amount.transform([[amount]])[0][0])
            volume_scaled = float(scaler_volume.transform([[volume]])[0][0])

            input_df = pd.DataFrame([{
                "amount_scaled": amount_scaled,
                "volume_scaled": volume_scaled,
                "hour":          hour,
                "minute":        minute,
                "is_sell":       is_sell,
            }])

            prediction  = int(model.predict(input_df)[0])
            fraud_proba = round(float(model.predict_proba(input_df)[0][1]), 4)

            st.divider()
            if prediction == 1:
                st.error(f"⚠️ **FRAUD DETECTED!**\nFraud Probability: **{fraud_proba*100:.1f}%**")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**\nFraud Probability: **{fraud_proba*100:.1f}%**")

            col3, col4, col5 = st.columns(3)
            col3.metric("Prediction", "FRAUD" if prediction == 1 else "LEGITIMATE")
            col4.metric("Fraud Probability", f"{fraud_proba*100:.1f}%")
            col5.metric("Amount Scaled", f"{amount_scaled:.4f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Streaming Data Overview")

    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))

    if not files:
        st.warning("Belum ada data processed. Jalankan stream_generator.py terlebih dahulu.")
    else:
        dfs = [pd.read_csv(f) for f in files]
        df  = pd.concat(dfs, ignore_index=True).drop_duplicates()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Cases", f"{df['Class'].sum():,}")
        col3.metric("Fraud Rate", f"{df['Class'].mean()*100:.2f}%")
        col4.metric("Total Batches", len(files))

        st.divider()
        col5, col6 = st.columns(2)

        with col5:
            fraud_counts = df["Class"].value_counts().reset_index()
            fraud_counts.columns = ["Class", "Count"]
            fraud_counts["Label"] = fraud_counts["Class"].map({0: "Legitimate", 1: "Fraud"})
            fig = px.pie(fraud_counts, values="Count", names="Label",
                        title="Transaction Distribution",
                        color_discrete_map={"Legitimate": "#00cc96", "Fraud": "#ef553b"})
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            fig2 = px.histogram(df, x="amount_scaled", color=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
                               title="Amount Distribution by Class",
                               color_discrete_map={"Legitimate": "#00cc96", "Fraud": "#ef553b"},
                               barmode="overlay", opacity=0.7)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Latest Data Sample")
        st.dataframe(df.tail(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Drift Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 Data Drift Detection")

    drift_path = Path("reports/drift_result.json")
    if not drift_path.exists():
        st.warning("Belum ada hasil drift detection. Jalankan drift_detector.py terlebih dahulu.")
    else:
        with open(drift_path) as f:
            drift_result = json.load(f)

        drift_detected = drift_result.get("drift_detected", False)

        if drift_detected:
            st.error("⚠️ **DATA DRIFT TERDETEKSI!** Model perlu diretrain.")
        else:
            st.success("✅ **Tidak ada drift** — model masih relevan.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Drift Detected", "YES ⚠️" if drift_detected else "NO ✅")
        col2.metric("Drifted Columns",
                   f"{drift_result.get('drifted_columns', 0)}/{drift_result.get('total_columns', 0)}")
        col3.metric("Last Check", drift_result.get("timestamp", "N/A")[:19])

        st.divider()
        details = drift_result.get("drift_details", {})
        if details:
            st.subheader("📋 Drift Details per Column")
            rows = []
            for col, data in details.items():
                rows.append({
                    "Column":       col,
                    "KS Statistic": data.get("ks_statistic", 0),
                    "P-Value":      data.get("p_value", 1),
                    "Ref Mean":     data.get("ref_mean", 0),
                    "Current Mean": data.get("current_mean", 0),
                    "Mean Shift %": data.get("mean_shift_pct", 0),
                    "Drift":        "⚠️ YES" if data.get("drift_detected") else "✅ NO"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Model Performance History")

    comparison_path = Path("reports/model_comparison.json")
    if comparison_path.exists():
        with open(comparison_path) as f:
            comparison = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Champion F1",
                   f"{comparison.get('current_champion_f1', 0):.4f}")
        col2.metric("New Model F1",
                   f"{comparison.get('new_model_f1', 0):.4f}")
        col3.metric("Promoted",
                   "✅ YES" if comparison.get("promoted") else "❌ NO",
                   delta=f"{comparison.get('improvement', 0):+.4f}")

    st.divider()
    st.subheader("🏆 Model Experiments")
    st.markdown("Lihat semua eksperimen di [DagsHub MLflow](https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow/#/experiments/0)")

    metrics_data = {
        "Model": ["RandomForest_100trees", "RandomForest_200trees", "DecisionTree", "LogisticRegression"],
        "F1 Score": [0.8660, 0.8262, 0.8401, 0.4264],
        "ROC AUC":  [0.9131, 0.9067, 0.9307, 0.7282],
        "Precision":[0.8936, 0.8129, 0.7929, 0.3180],
        "Recall":   [0.8400, 0.8400, 0.8933, 0.6467],
    }
    df_metrics = pd.DataFrame(metrics_data)

    fig3 = px.bar(df_metrics, x="Model", y=["F1 Score", "ROC AUC", "Precision", "Recall"],
                  title="Model Comparison", barmode="group",
                  color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"])
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(df_metrics, use_container_width=True)