"""
Future Forecasting Page - Predict Future Fraud Trends
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_loader import DataLoader

st.set_page_config(
    page_title="Fraud Detection - Forecasting",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Future Fraud Forecasting")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load cleaned dataset."""
    # Get absolute path based on app location
    app_dir = Path(__file__).parent.parent.parent
    data_dir = app_dir / 'data'
    
    cleaned_path = data_dir / 'creditcard_cleaned.csv'
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        return df
    else:
        # Fallback to original data
        original_path = data_dir / 'creditcard.csv'
        if original_path.exists():
            df = pd.read_csv(original_path)
            return df.drop_duplicates()
        else:
            st.error(f"Data file not found at {data_dir}")
            return pd.DataFrame()

df = load_data()

# Convert Time to datetime-like for forecasting
df['Hour'] = (df['Time'] / 3600) % 24
df['Day'] = (df['Time'] / (3600 * 24)) % 7

# Sidebar
st.sidebar.header("Forecasting Parameters")
forecast_days = st.sidebar.slider("Forecast Period (days)", 1, 30, 7)
forecast_method = st.sidebar.selectbox("Forecasting Method", 
                                       ["Trend Analysis", "Moving Average", "Seasonal Pattern"])

# Historical Trends
st.header("📊 Historical Fraud Trends")

# Daily fraud rate
df['Day_Index'] = (df['Time'] / (3600 * 24)).astype(int)
daily_fraud = df.groupby('Day_Index')['Class'].agg(['sum', 'count'])
daily_fraud['Fraud_Rate'] = daily_fraud['sum'] / daily_fraud['count'] * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Fraud count over time
axes[0].plot(daily_fraud.index, daily_fraud['sum'], marker='o', linewidth=2, color='red')
axes[0].set_title('Daily Fraud Count Over Time', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Day Index')
axes[0].set_ylabel('Number of Fraud Cases')
axes[0].grid(alpha=0.3)

# Fraud rate over time
axes[1].plot(daily_fraud.index, daily_fraud['Fraud_Rate'], marker='o', linewidth=2, color='orange')
axes[1].set_title('Daily Fraud Rate Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day Index')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Forecasting
st.header("🔮 Future Fraud Predictions")

if forecast_method == "Trend Analysis":
    # Simple linear trend
    recent_days = daily_fraud.tail(30)
    x = np.array(recent_days.index)
    y = np.array(recent_days['sum'])
    
    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    future_days = np.arange(daily_fraud.index.max() + 1, daily_fraud.index.max() + forecast_days + 1)
    predictions = np.polyval(coeffs, future_days)
    
elif forecast_method == "Moving Average":
    # Moving average
    window = 7
    recent_fraud = daily_fraud['sum'].tail(window).mean()
    future_days = np.arange(daily_fraud.index.max() + 1, daily_fraud.index.max() + forecast_days + 1)
    predictions = np.full(len(future_days), recent_fraud)
    
else:  # Seasonal Pattern
    # Use hourly patterns
    hourly_fraud = df.groupby('Hour')['Class'].mean() * 100
    avg_daily_fraud = df['Class'].mean() * len(df) / len(daily_fraud)
    future_days = np.arange(daily_fraud.index.max() + 1, daily_fraud.index.max() + forecast_days + 1)
    predictions = np.full(len(future_days), avg_daily_fraud)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Day_Index': future_days,
    'Predicted_Fraud_Count': predictions,
    'Predicted_Fraud_Rate': predictions / (len(df) / len(daily_fraud)) * 100
})

# Display forecast
st.subheader(f"Forecast for Next {forecast_days} Days")
st.dataframe(forecast_df, width='stretch')

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Historical + Forecast
historical_days = daily_fraud.tail(30)
axes[0].plot(historical_days.index, historical_days['sum'], 
            marker='o', linewidth=2, label='Historical', color='blue')
axes[0].plot(forecast_df['Day_Index'], forecast_df['Predicted_Fraud_Count'], 
            marker='s', linewidth=2, linestyle='--', label='Forecast', color='red')
axes[0].set_title('Fraud Count: Historical vs Forecast', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Day Index')
axes[0].set_ylabel('Number of Fraud Cases')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Fraud rate forecast
axes[1].plot(historical_days.index, historical_days['Fraud_Rate'], 
            marker='o', linewidth=2, label='Historical Rate', color='green')
axes[1].plot(forecast_df['Day_Index'], forecast_df['Predicted_Fraud_Rate'], 
            marker='s', linewidth=2, linestyle='--', label='Forecast Rate', color='orange')
axes[1].set_title('Fraud Rate: Historical vs Forecast', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day Index')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Summary statistics
st.subheader("📊 Forecast Summary")
col1, col2, col3 = st.columns(3)

avg_predicted = forecast_df['Predicted_Fraud_Count'].mean()
total_predicted = forecast_df['Predicted_Fraud_Count'].sum()
avg_rate = forecast_df['Predicted_Fraud_Rate'].mean()

col1.metric("Avg Daily Fraud", f"{avg_predicted:.1f}")
col2.metric("Total Predicted", f"{total_predicted:.0f}")
col3.metric("Avg Fraud Rate", f"{avg_rate:.3f}%")

# Hourly pattern analysis
st.header("⏰ Hourly Fraud Patterns")
hourly_fraud = df.groupby('Hour')['Class'].agg(['sum', 'count', 'mean'])
hourly_fraud['Fraud_Rate'] = hourly_fraud['mean'] * 100

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(hourly_fraud.index, hourly_fraud['Fraud_Rate'], color='red', alpha=0.7)
ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Fraud Rate (%)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Download forecast
csv = forecast_df.to_csv(index=False)
st.download_button(
    label="📥 Download Forecast",
    data=csv,
    file_name=f"fraud_forecast_{forecast_days}days.csv",
    mime="text/csv"
)

