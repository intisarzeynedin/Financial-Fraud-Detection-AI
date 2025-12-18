# ML Training Pipeline Documentation

## Overview
This directory contains the complete ML training pipeline for credit card fraud detection, as specified in the project plan.

## Project Structure

```
transactions/
├── data/                 # Raw data files
│   ├── creditcard.csv
│   ├── transactions.json
│   └── unstructured_logs.txt
├── src/                  # Source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── models.py        # ML model implementations
│   ├── evaluator.py     # Model evaluation and metrics
│   ├── train.py         # Main training script
│   └── model_loader.py  # Model loading utility for app
├── models/               # Saved trained models (created after training)
├── reports/              # Evaluation reports and charts (created after training)
└── app/                  # Streamlit application (to be created)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Run the main training script to train all models:

```bash
cd transactions
python src/train.py
```

This will:
1. Load and preprocess the credit card data
2. Train baseline models (Logistic Regression, Decision Tree)
3. Train advanced models (Random Forest, XGBoost)
4. Evaluate all models with metrics (F1, AUC, AUPRC)
5. Generate comparison charts and reports
6. Save trained models to `models/` directory

### Training Pipeline Details

#### 1. Data Loading (`data_loader.py`)
- Loads CSV/JSON data files
- Handles large files with chunking
- Preprocesses data: scaling, train/test split
- Maintains class distribution with stratified splitting

#### 2. Model Training (`models.py`)
- **Baseline Models:**
  - Logistic Regression (with class weights or SMOTE)
  - Decision Tree (with class weights or SMOTE)
  
- **Advanced Models:**
  - Random Forest (with hyperparameter tuning option)
  - XGBoost (with hyperparameter tuning option)

- **Class Imbalance Handling:**
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Class weights (balanced or custom)
  - Scale pos weight for XGBoost

#### 3. Model Evaluation (`evaluator.py`)
- Calculates metrics:
  - Accuracy, Precision, Recall
  - F1-Score
  - ROC-AUC (Area Under ROC Curve)
  - PR-AUC (Area Under Precision-Recall Curve)
- Generates visualizations:
  - ROC curves comparison
  - Precision-Recall curves
  - Confusion matrices
  - Metrics comparison bar charts
- Saves results to CSV

#### 4. Model Loading (`model_loader.py`)
- Utility for loading trained models in Streamlit app
- Handles feature scaling automatically
- Provides prediction interface

## Model Outputs

After training, you'll find:

### Models Directory (`models/`)
- `logistic_regression.pkl` - Logistic Regression with class weights
- `logistic_regression_smote.pkl` - Logistic Regression with SMOTE
- `decision_tree.pkl` - Decision Tree classifier
- `random_forest.pkl` - Random Forest classifier
- `xgboost.pkl` - XGBoost classifier
- `scaler.pkl` - Fitted StandardScaler
- `feature_columns.pkl` - Feature column names

### Reports Directory (`reports/`)
- `evaluation_results.csv` - Metrics comparison table
- `roc_curves.png` - ROC curves for all models
- `pr_curves.png` - Precision-Recall curves
- `confusion_matrices.png` - Confusion matrices
- `metrics_comparison.png` - Bar charts comparing metrics

## Model Selection

The training script automatically:
- Trains multiple models with different configurations
- Evaluates all models on test set
- Compares models and identifies best performer
- Saves all models for later use

Best model is selected based on F1-Score (recommended for imbalanced data).

## Customization

### Hyperparameter Tuning
To enable hyperparameter tuning, edit `src/train.py`:
```python
rf_model = models.random_forest(
    X_train, y_train, 
    tune_hyperparameters=True  # Enable tuning
)
```

### Adding New Models
1. Add model method to `FraudDetectionModels` class in `src/models.py`
2. Call it in `src/train.py`
3. Evaluate it using `evaluator.evaluate_model()`

### Using Different Data
Modify `src/train.py` to load different data:
```python
# Load JSON data instead
df = data_loader.load_json_data('transactions.json')
```

## Next Steps

After training models:
1. Review evaluation results in `reports/` directory
2. Select best model for production
3. Use `model_loader.py` in Streamlit app for predictions
4. Implement real-time prediction stream visualization

## Notes

- Models are trained on 80% of data, tested on 20%
- Class imbalance is handled using both SMOTE and class weights
- All models are saved for comparison and ensemble methods
- Feature scaling is applied consistently across all models

