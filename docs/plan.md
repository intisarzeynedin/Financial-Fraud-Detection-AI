# Credit Card Fraud Detection ML Project Plan

## Project Overview
This project aims to develop a machine learning model for detecting credit card fraud using the provided transaction datasets. The datasets include CSV, JSON, and unstructured log formats containing transaction features, amounts, and fraud labels.

## Folder Structure
```
transactions/
├── data/                 # Raw data files
│   ├── creditcard.csv
│   ├── transactions.json
│   └── unstructured_logs.txt
├── notebooks/            # Jupyter notebooks for exploration and prototyping
├── src/                  # Source code for data processing, training, and evaluation
├── models/               # Saved trained models
├── reports/              # Evaluation reports, charts, and metrics
├── app/                  # Streamlit application for visualization
└── docs/                 # Documentation
    └── plan.md
```

## Weekly Plan

### Week 3: Model Development – Implement ML Training
- **Objective**: Start with baseline models and handle imbalanced data.
- **Tasks**:
  - Load and preprocess data from CSV/JSON formats.
  - Handle class imbalance using SMOTE and class weights.
  - Implement baseline models: Logistic Regression, Decision Tree using Scikit-learn.
  - If data is too large, consider Spark MLlib for distributed training.
- **Deliverables**:
  - Initial model results (precision, recall, F1-score).
  - Analysis of sampling effects (SMOTE vs. class weights).
  - Code for data preprocessing and baseline training.

### Week 4: Model Tuning & Evaluation
- **Objective**: Compare advanced models and fine-tune for best performance.
- **Tasks**:
  - Implement advanced models: Random Forest, XGBoost.
  - Perform hyperparameter tuning using cross-validation.
  - Evaluate models using metrics: F1, AUC, AUPRC.
  - Select the best model and validate on test set.
- **Deliverables**:
  - Final model with evaluation metrics chart.
  - Code for model training and evaluation.
  - Streamlit app for visualizing model results and predictions.

## Technologies
- **Machine Learning**: Scikit-learn (Logistic Regression, Random Forest), XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Big Data (if needed)**: PySpark MLlib
- **Environment**: Python 3.x

## Next Steps
1. Set up Python environment and install dependencies.
2. Implement data loading and preprocessing scripts.
3. Train baseline models and evaluate.
4. Develop Streamlit app for visualization.
5. Fine-tune models and finalize.