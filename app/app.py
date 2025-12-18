"""
Main Streamlit App - Credit Card Fraud Detection Dashboard
Multi-page application for fraud detection visualization and prediction
"""

import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("---")

st.markdown("""
## Welcome to the Fraud Detection Dashboard!

This interactive dashboard provides comprehensive tools for fraud detection and analysis.

### 🎯 Features:

1. **📊 Overview** - Project summary and quick statistics
2. **🔍 Data Exploration** - Interactive data visualization and analysis
3. **🔮 ML Prediction** - Real-time fraud prediction on new transactions
4. **📈 ML Results** - Model evaluation metrics and performance charts
5. **🔮 Forecasting** - Future fraud trend predictions

### 🚀 Getting Started:

Use the sidebar to navigate between different pages and explore the fraud detection system.

### 📝 Note:

Make sure you have:
- ✅ Run the data cleaning notebook (01b_DataCleaning.ipynb)
- ✅ Run the preprocessing notebook (02_Preprocessing_FeatureEngineering.ipynb)
- ✅ Run the ML training notebook (03_ML_Training.ipynb)
- ✅ Run the ML evaluation notebook (04_ML_Evaluation.ipynb)

All models and results will be automatically loaded in this dashboard.
""")

# Quick stats
st.markdown("---")
st.header("📊 Quick Statistics")

col1, col2, col3, col4 = st.columns(4)

try:
    from pathlib import Path
    import pandas as pd
    
    cleaned_path = Path('data/creditcard_cleaned.csv')
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Normal", f"{len(df[df['Class']==0]):,}")
        col3.metric("Fraud", f"{len(df[df['Class']==1]):,}")
        col4.metric("Fraud Rate", f"{df['Class'].mean()*100:.2f}%")
    else:
        col1.metric("Status", "Data not loaded")
        col2.metric("Status", "Run notebooks first")
except:
    st.info("Run the data cleaning notebook to see statistics here.")

# Navigation
st.sidebar.title("📑 Navigation")
st.sidebar.markdown("""
### Pages:
1. **📊 Overview** - Project overview
2. **🔍 Data Exploration** - Explore dataset
3. **🔮 ML Prediction** - Make predictions
4. **📈 ML Results** - View model results
5. **🔮 Forecasting** - Future predictions
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Start with Overview to understand the project, then explore other pages!")

