"""
ML Model Results Page - Evaluation Metrics and Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

st.set_page_config(
    page_title="Fraud Detection - ML Results",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ML Model Results & Evaluation")
st.markdown("---")

# Load evaluation results
# Get absolute path based on app location
app_dir = Path(__file__).parent.parent.parent
reports_dir = app_dir / 'reports'

# Check if results exist
results_csv = reports_dir / 'evaluation_results.csv'

if not results_csv.exists():
    st.warning("⚠️ Evaluation results not found. Please run the ML Evaluation notebook first.")
    st.info("Run `04_ML_Evaluation.ipynb` to generate evaluation results and visualizations.")
    st.stop()

# Load results
results_df = pd.read_csv(results_csv)

# Display metrics table
st.header("📊 Model Comparison")
st.dataframe(results_df, width='stretch', hide_index=True)

# Best model
best_model = results_df.loc[results_df['F1-Score'].idxmax()]
st.success(f"🏆 **Best Model**: {best_model['Model']} (F1-Score: {best_model['F1-Score']:.4f})")

# Metrics visualization
st.header("📈 Metrics Visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("F1-Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='F1-Score', kind='barh', ax=ax, color='steelblue')
    ax.set_title('F1-Score by Model', fontweight='bold')
    ax.set_xlabel('F1-Score')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("ROC-AUC Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(x='Model', y='ROC-AUC', kind='barh', ax=ax, color='green')
    ax.set_title('ROC-AUC by Model', fontweight='bold')
    ax.set_xlabel('ROC-AUC')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Load visualizations
st.header("📉 Evaluation Charts")

# ROC Curves
roc_path = reports_dir / 'roc_curves.png'
if roc_path.exists():
    st.subheader("ROC Curves")
    st.image(str(roc_path), width='stretch')

# PR Curves
pr_path = reports_dir / 'pr_curves.png'
if pr_path.exists():
    st.subheader("Precision-Recall Curves")
    st.image(str(pr_path), width='stretch')

# Confusion Matrices
cm_path = reports_dir / 'confusion_matrices.png'
if cm_path.exists():
    st.subheader("Confusion Matrices")
    st.image(str(cm_path), width='stretch')

# Metrics Comparison
metrics_path = reports_dir / 'metrics_comparison.png'
if metrics_path.exists():
    st.subheader("Metrics Comparison")
    st.image(str(metrics_path), width='stretch')

# Detailed metrics
st.header("📋 Detailed Metrics")

selected_model_detail = st.selectbox("Select Model for Details", results_df['Model'].tolist())

model_row = results_df[results_df['Model'] == selected_model_detail].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{model_row['Accuracy']:.4f}")
col2.metric("Precision", f"{model_row['Precision']:.4f}")
col3.metric("Recall", f"{model_row['Recall']:.4f}")
col4.metric("F1-Score", f"{model_row['F1-Score']:.4f}")

col1, col2 = st.columns(2)
col1.metric("ROC-AUC", f"{model_row['ROC-AUC']:.4f}")
col2.metric("PR-AUC", f"{model_row['PR-AUC']:.4f}")

