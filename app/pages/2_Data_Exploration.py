"""
Data Exploration Page - Interactive Data Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_loader import DataLoader

st.set_page_config(
    page_title="Fraud Detection - Data Exploration",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Data Exploration")
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

# Sidebar filters
st.sidebar.header("Filters")
fraud_filter = st.sidebar.selectbox("Transaction Type", ["All", "Normal Only", "Fraud Only"])

if fraud_filter == "Normal Only":
    df_filtered = df[df['Class'] == 0]
elif fraud_filter == "Fraud Only":
    df_filtered = df[df['Class'] == 1]
else:
    df_filtered = df

# Dataset Overview
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df_filtered):,}")
col2.metric("Normal", f"{len(df_filtered[df_filtered['Class']==0]):,}")
col3.metric("Fraud", f"{len(df_filtered[df_filtered['Class']==1]):,}")
col4.metric("Fraud Rate", f"{df_filtered['Class'].mean()*100:.2f}%")

# Display data
st.subheader("📋 Data Preview")
st.dataframe(df_filtered.head(100), width='stretch')

# Statistics
st.subheader("📈 Statistical Summary")
st.dataframe(df_filtered.describe(), width='stretch')

# Visualizations
st.header("📊 Visualizations")

# Class Distribution
st.subheader("Class Distribution")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_counts = df_filtered['Class'].value_counts()
class_labels = class_counts.index.map({0: 'Normal', 1: 'Fraud'})
class_colors = class_counts.index.map({0: 'blue', 1: 'red'})

axes[0].bar(class_labels, class_counts.values, color=class_colors)
axes[0].set_title('Class Distribution (Count)')
axes[0].set_ylabel('Count')

axes[1].pie(class_counts.values, labels=class_labels, autopct='%1.2f%%', 
            colors=class_colors, startangle=90)
axes[1].set_title('Class Distribution (Percentage)')

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Amount Distribution
st.subheader("Transaction Amount Analysis")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

normal_amounts = df_filtered[df_filtered['Class'] == 0]['Amount']
fraud_amounts = df_filtered[df_filtered['Class'] == 1]['Amount']

axes[0].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
axes[0].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
axes[0].set_title('Amount Distribution by Class')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(0, 5000)

sns.boxplot(data=df_filtered, x='Class', y='Amount', ax=axes[1])
axes[1].set_title('Amount Distribution (Box Plot)')
axes[1].set_yscale('log')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Normal', 'Fraud'])

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Feature Analysis
st.subheader("Feature Analysis (V1-V28)")
v_features = [f'V{i}' for i in range(1, 29)]
selected_features = st.multiselect("Select Features to Compare", v_features, default=v_features[:5])

if selected_features:
    fig, axes = plt.subplots(len(selected_features), 1, figsize=(12, 3*len(selected_features)))
    if len(selected_features) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(selected_features):
        df_filtered[df_filtered['Class'] == 0][feature].hist(bins=30, ax=axes[idx], 
                                                           alpha=0.5, label='Normal', color='blue')
        df_filtered[df_filtered['Class'] == 1][feature].hist(bins=30, ax=axes[idx], 
                                                             alpha=0.5, label='Fraud', color='red')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].legend()
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Time Analysis
st.subheader("Time-based Analysis")
df_filtered['Hour'] = (df_filtered['Time'] / 3600) % 24

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fraud_by_hour = df_filtered[df_filtered['Class'] == 1].groupby(df_filtered['Hour'].astype(int)).size()
normal_by_hour = df_filtered[df_filtered['Class'] == 0].groupby(df_filtered['Hour'].astype(int)).size()

axes[0].bar(fraud_by_hour.index, fraud_by_hour.values, color='red', alpha=0.7)
axes[0].set_title('Fraud Transactions by Hour')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Number of Fraud Transactions')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(normal_by_hour.index, normal_by_hour.values, color='blue', alpha=0.7)
axes[1].set_title('Normal Transactions by Hour')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Number of Normal Transactions')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

