"""
ML Prediction Page - Real-time Fraud Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.model_loader import ModelLoader

st.set_page_config(
    page_title="Fraud Detection - ML Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Real-time Fraud Prediction")
st.markdown("---")

# Initialize model loader
@st.cache_resource
def load_models():
    """Load models and scaler."""
    # Get absolute path based on app location
    app_dir = Path(__file__).parent.parent.parent
    models_dir = app_dir / 'models'
    
    model_loader = ModelLoader(models_dir=str(models_dir))
    try:
        model_loader.load_scaler()
        model_loader.load_feature_columns()
        available_models = model_loader.get_available_models()
        return model_loader, available_models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, []

model_loader, available_models = load_models()

if not available_models:
    st.warning("⚠️ No trained models found. Please run the ML Training notebook first.")
    st.stop()

# Sidebar - Model Selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose Model", available_models)

# Load selected model
@st.cache_resource
def get_model(model_name):
    """Load specific model."""
    if model_loader:
        return model_loader.load_model(model_name)
    return None

model = get_model(selected_model)

# Prediction Interface
st.header("📝 Transaction Input")

# Two input methods
input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=0.01)
        time = st.number_input("Time (seconds)", min_value=0.0, value=0.0, step=1.0)
    
    with col2:
        st.info("**V Features** (PCA transformed)")
        st.caption("Enter values for V1-V28 features")
    
    # V features input
    v_cols = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        v_values[f'V{i}'] = v_cols[col_idx].number_input(
            f"V{i}", value=0.0, step=0.01, key=f"v{i}"
        )
    
    # Create transaction data
    if st.button("🔍 Predict Fraud", type="primary"):
        transaction_data = {
            'Time': time,
            'Amount': amount,
            **v_values
        }
        
        try:
            # Make prediction
            result = model_loader.predict_single(selected_model, transaction_data)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['is_fraud']:
                    st.error(f"🚨 **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **NORMAL TRANSACTION**")
            
            with col2:
                st.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
            
            with col3:
                st.metric("Prediction", "Fraud" if result['is_fraud'] else "Normal")
            
            # Probability bar
            st.subheader("Prediction Confidence")
            st.progress(result['fraud_probability'])
            st.caption(f"Fraud Probability: {result['fraud_probability']*100:.2f}%")
            
            # Risk level
            if result['fraud_probability'] > 0.7:
                risk_level = "🔴 High Risk"
            elif result['fraud_probability'] > 0.4:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"
            
            st.info(f"**Risk Level**: {risk_level}")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), width='stretch')
            
            if st.button("🔍 Predict All Transactions", type="primary"):
                predictions = []
                probabilities = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    try:
                        result = model_loader.predict_single(selected_model, row.to_dict())
                        predictions.append(result['prediction'])
                        probabilities.append(result['fraud_probability'])
                    except Exception as e:
                        st.warning(f"Error predicting row {idx}: {e}")
                        predictions.append(0)
                        probabilities.append(0.0)
                    
                    progress_bar.progress((idx + 1) / len(df))
                    status_text.text(f"Processing {idx + 1}/{len(df)} transactions...")
                
                # Add predictions to dataframe
                df['Prediction'] = ['Fraud' if p == 1 else 'Normal' for p in predictions]
                df['Fraud_Probability'] = probabilities
                
                st.success(f"✅ Predicted {len(df)} transactions!")
                
                # Summary
                fraud_count = sum(predictions)
                st.metric("Fraud Detected", f"{fraud_count} / {len(df)}", 
                         f"{fraud_count/len(df)*100:.2f}%")
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(df, width='stretch')
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Model Information
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.info(f"**Selected Model**: {selected_model}")
st.sidebar.caption("Model trained on cleaned dataset with class imbalance handling")

