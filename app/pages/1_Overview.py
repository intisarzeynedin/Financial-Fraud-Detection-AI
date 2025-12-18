"""
Overview Page - Project Summary and Dashboard
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Fraud Detection - Overview",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Credit Card Fraud Detection Dashboard")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("""
- **Overview** - Project summary
- **Data Exploration** - Explore dataset
- **ML Prediction** - Real-time predictions
- **ML Results** - Model evaluation
- **Forecasting** - Future predictions
""")

# Main content
# Get absolute path for data
app_dir = Path(__file__).parent.parent.parent
data_dir = app_dir / 'data'

col1, col2, col3 = st.columns(3)

# Try to load actual data
try:
    cleaned_path = data_dir / 'creditcard_cleaned.csv'
    if cleaned_path.exists():
        df_temp = pd.read_csv(cleaned_path)
        total_txns = len(df_temp)
        fraud_count = df_temp['Class'].sum()
        fraud_rate = df_temp['Class'].mean() * 100
    else:
        total_txns = 283726
        fraud_count = 492
        fraud_rate = 0.17
except:
    total_txns = 283726
    fraud_count = 492
    fraud_rate = 0.17

with col1:
    st.metric("Total Transactions", f"{total_txns:,}", "Cleaned Dataset")
    
with col2:
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%", f"{fraud_count} fraud cases")
    
with col3:
    st.metric("Models Trained", "4", "Ready for prediction")

st.markdown("---")

# Project Overview
st.header("📋 Project Overview")
st.markdown("""
This project implements a machine learning system for detecting credit card fraud using advanced ML models.

### Key Features:
- **Data Cleaning**: Removed duplicates and handled data quality issues
- **Feature Engineering**: Created time-based and interaction features
- **ML Models**: Trained 4 different models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Real-time Prediction**: Make predictions on new transactions
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Future Forecasting**: Predict future fraud trends
""")

# Workflow
st.header("🔄 Workflow")
st.markdown("""
1. **Data Cleaning** → Remove duplicates and clean data
2. **EDA** → Exploratory data analysis
3. **Preprocessing** → Feature engineering and scaling
4. **Training** → Train multiple ML models
5. **Evaluation** → Evaluate and compare models
6. **Deployment** → Streamlit dashboard for predictions
""")

# Model Information
st.header("🤖 Trained Models")
models_info = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Type': ['Linear', 'Tree-based', 'Ensemble', 'Gradient Boosting'],
    'Status': ['✅ Trained', '✅ Trained', '✅ Trained', '✅ Trained']
})

st.dataframe(models_info, width='stretch', hide_index=True)

# Quick Stats
st.header("📈 Quick Statistics")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Info")
    st.markdown("""
    - **Original Size**: 284,807 transactions
    - **Cleaned Size**: 283,726 transactions
    - **Duplicates Removed**: 1,081
    - **Features**: 30 (V1-V28, Time, Amount)
    - **Target**: Class (0=Normal, 1=Fraud)
    """)

with col2:
    st.subheader("Model Performance")
    st.markdown("""
    - **Best Model**: Check ML Results page
    - **Metrics**: F1-Score, ROC-AUC, PR-AUC
    - **Evaluation**: Comprehensive metrics available
    - **Visualizations**: ROC curves, PR curves, confusion matrices
    """)

# Next Steps
st.header("🚀 Next Steps")
st.markdown("""
1. Navigate to **Data Exploration** to explore the dataset
2. Use **ML Prediction** to make real-time fraud predictions
3. View **ML Results** to see model performance metrics
4. Check **Forecasting** for future fraud trend predictions
""")

