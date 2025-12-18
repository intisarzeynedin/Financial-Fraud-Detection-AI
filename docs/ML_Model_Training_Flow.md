# ML Model Training Flow Documentation

## Overview
This document describes the complete machine learning training pipeline for credit card fraud detection, detailing how each notebook contributes to the model development process.

---

## Notebook Workflow

### 1. Data Cleaning (`01b_DataCleaning.ipynb`)

**Purpose**: Clean and prepare raw data for analysis and modeling.

**Process**:
1. **Data Loading**: Loads the original `creditcard.csv` file containing 284,807 transactions
2. **Duplicate Detection**: Identifies and counts duplicate rows in the dataset
3. **Duplicate Removal**: Removes duplicate entries, keeping the first occurrence
4. **Data Quality Checks**:
   - Missing value detection
   - Data type validation
   - Anomaly detection (negative amounts, invalid class values)
5. **Data Export**: Saves cleaned dataset to `creditcard_cleaned.csv`

**Output**:
- Cleaned dataset: 283,726 transactions (1,081 duplicates removed)
- File: `data/creditcard_cleaned.csv`

**Key Features**:
- Handles large datasets efficiently
- Provides data quality metrics
- Visualizes cleaning results

---

### 2. Exploratory Data Analysis (`01_EDA_Analysis.ipynb`)

**Purpose**: Understand data characteristics, distributions, and patterns.

**Process**:
1. **Data Overview**: 
   - Dataset shape and structure
   - Column information
   - Statistical summaries
2. **Target Variable Analysis**:
   - Class distribution (Normal vs Fraud)
   - Imbalance ratio calculation
   - Visualization of class proportions
3. **Feature Analysis**:
   - Transaction amount statistics by class
   - V1-V28 feature distributions
   - Correlation analysis
4. **Time-based Analysis**:
   - Hourly transaction patterns
   - Fraud occurrence by time of day
5. **Insights Generation**: Identifies key patterns and recommendations

**Key Findings**:
- Highly imbalanced dataset (0.17% fraud rate)
- V features are PCA-transformed (already scaled)
- Amount feature shows different distributions for fraud vs normal
- Temporal patterns exist in transaction data

---

### 3. Preprocessing & Feature Engineering (`02_Preprocessing_FeatureEngineering.ipynb`)

**Purpose**: Transform raw data into features suitable for ML models.

**Process**:
1. **Data Loading**: Loads cleaned dataset from previous step
2. **Feature Engineering**:
   - **Time-based features**:
     - Hour of day (0-23)
     - Day of week (0-6)
     - Cyclical encoding (sin/cos transformations)
   - **Amount transformations**:
     - Log transformation
     - Square root transformation
   - **Interaction features**:
     - V_Sum: Sum of all V features
     - V_Mean: Mean of V features
     - V_Std: Standard deviation of V features
3. **Data Splitting**:
   - Train/Test split (80/20)
   - Stratified splitting to maintain class distribution
   - Random state for reproducibility
4. **Feature Scaling**:
   - StandardScaler applied to all features
   - Fit on training data only
   - Transform both train and test sets

**Output**:
- Training set: ~227,000 samples
- Test set: ~56,700 samples
- Scaled features ready for modeling

**Key Features**:
- Maintains class distribution in splits
- Handles feature scaling consistently
- Creates informative derived features

---

### 4. ML Model Training (`03_ML_Training.ipynb`)

**Purpose**: Train multiple machine learning models for fraud detection.

**Process**:

#### 4.1 Data Preparation
- Loads cleaned and preprocessed data
- Verifies data quality
- Prepares train/test splits

#### 4.2 Baseline Models

**Logistic Regression**:
- **Method**: Linear classifier with regularization
- **Class Imbalance Handling**: 
  - Option 1: Class weights (`class_weight='balanced'`)
  - Option 2: SMOTE oversampling
- **Hyperparameters**: 
  - `max_iter=1000`
  - `random_state=42`
- **Use Case**: Fast, interpretable baseline model

**Decision Tree**:
- **Method**: Tree-based classifier
- **Class Imbalance Handling**: Class weights or SMOTE
- **Hyperparameters**:
  - `max_depth=10` (prevents overfitting)
  - `random_state=42`
- **Use Case**: Non-linear patterns, feature importance

#### 4.3 Advanced Models

**Random Forest**:
- **Method**: Ensemble of decision trees
- **Class Imbalance Handling**: Class weights or SMOTE
- **Hyperparameters**:
  - `n_estimators=100` (number of trees)
  - `class_weight='balanced'`
  - Optional: GridSearchCV for tuning
- **Use Case**: Robust, handles non-linearity well

**XGBoost**:
- **Method**: Gradient boosting framework
- **Class Imbalance Handling**: 
  - `scale_pos_weight` parameter
  - Or SMOTE oversampling
- **Hyperparameters**:
  - `n_estimators=100`
  - `max_depth=5`
  - `learning_rate=0.1`
  - Optional: GridSearchCV for tuning
- **Use Case**: High performance, handles complex patterns

#### 4.4 Model Persistence
- All trained models saved to `models/` directory
- Scaler saved for inference
- Feature columns saved for consistency

**Output**:
- `models/logistic_regression.pkl`
- `models/decision_tree.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/scaler.pkl`
- `models/feature_columns.pkl`

---

### 5. ML Model Evaluation (`04_ML_Evaluation.ipynb`)

**Purpose**: Comprehensive evaluation of all trained models.

**Process**:

#### 5.1 Model Loading
- Loads all trained models from `models/` directory
- Loads test data with same preprocessing

#### 5.2 Metrics Calculation
For each model, calculates:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve (important for imbalanced data)
- **Confusion Matrix**: TP, TN, FP, FN counts

#### 5.3 Visualizations Generated
1. **ROC Curves**: Comparison of all models
2. **Precision-Recall Curves**: Better for imbalanced data
3. **Confusion Matrices**: Visual representation of predictions
4. **Metrics Comparison**: Bar charts comparing all metrics

#### 5.4 Model Comparison
- Ranks models by F1-Score (primary metric for imbalanced data)
- Identifies best performing model
- Saves comparison table to CSV

**Output**:
- `reports/evaluation_results.csv`
- `reports/roc_curves.png`
- `reports/pr_curves.png`
- `reports/confusion_matrices.png`
- `reports/metrics_comparison.png`

---

### 6. Forecasting Analysis (`05_Forecasting_Analysis.ipynb`)

**Purpose**: Predict future fraud trends using time series analysis.

**Process**:

#### 6.1 Time Series Preparation
- Converts Time feature to day indices
- Calculates daily fraud statistics:
  - Daily fraud count
  - Daily fraud rate
  - Total transactions per day

#### 6.2 Historical Trend Analysis
- Visualizes fraud patterns over time
- Identifies trends and patterns
- Analyzes seasonal variations

#### 6.3 Forecasting Methods

**Moving Average**:
- Uses recent N-day average
- Simple and stable predictions
- Good for short-term forecasts

**Linear Trend**:
- Fits linear regression to historical data
- Extrapolates future values
- Captures overall trend direction

**Seasonal Patterns**:
- Analyzes hourly patterns
- Identifies peak fraud hours
- Uses patterns for predictions

#### 6.4 Forecast Generation
- Predicts next 7 days (configurable)
- Generates forecast dataframe
- Visualizes historical vs forecast

**Output**:
- `reports/forecasting_trends.png`
- `reports/forecast_visualization.png`
- `reports/future_forecast.csv`

---

## Model Training Pipeline Summary

### Data Flow:
```
Raw Data (creditcard.csv)
    ↓
[01b_DataCleaning] → Cleaned Data (creditcard_cleaned.csv)
    ↓
[01_EDA_Analysis] → Insights & Understanding
    ↓
[02_Preprocessing] → Scaled Features (X_train, X_test, y_train, y_test)
    ↓
[03_ML_Training] → Trained Models (*.pkl files)
    ↓
[04_ML_Evaluation] → Evaluation Metrics & Charts
    ↓
[05_Forecasting] → Future Predictions
```

### Key Design Decisions:

1. **Class Imbalance Handling**:
   - Primary: Class weights (faster, no data augmentation)
   - Alternative: SMOTE (synthetic data generation)
   - XGBoost: Uses `scale_pos_weight` parameter

2. **Feature Engineering**:
   - Time-based features capture temporal patterns
   - Amount transformations handle skewness
   - Interaction features capture relationships

3. **Model Selection**:
   - Multiple models for comparison
   - F1-Score as primary metric (suitable for imbalanced data)
   - ROC-AUC and PR-AUC for comprehensive evaluation

4. **Reproducibility**:
   - Fixed random seeds (random_state=42)
   - Stratified splitting maintains class distribution
   - Consistent preprocessing pipeline

---

## Model Architecture Details

### Logistic Regression
- **Algorithm**: Linear classification with L2 regularization
- **Advantages**: Fast, interpretable, good baseline
- **Limitations**: Assumes linear relationships

### Decision Tree
- **Algorithm**: Recursive binary splitting
- **Advantages**: Non-linear, interpretable, feature importance
- **Limitations**: Can overfit, less stable

### Random Forest
- **Algorithm**: Ensemble of decision trees with bagging
- **Advantages**: Robust, handles non-linearity, feature importance
- **Limitations**: Less interpretable, slower than single tree

### XGBoost
- **Algorithm**: Gradient boosting with regularization
- **Advantages**: High performance, handles complex patterns
- **Limitations**: More complex, requires tuning

---

## Evaluation Strategy

### Metrics Used:
1. **F1-Score**: Primary metric (balances precision and recall)
2. **ROC-AUC**: Overall model performance
3. **PR-AUC**: Better for imbalanced data
4. **Precision**: Important to minimize false positives
5. **Recall**: Important to catch all fraud cases

### Why These Metrics?
- **Imbalanced Data**: F1 and PR-AUC are more informative than accuracy
- **Business Context**: Both precision and recall matter (minimize false alarms while catching fraud)
- **Comprehensive**: Multiple metrics provide complete picture

---

## Model Deployment

### Saved Artifacts:
- **Models**: All trained models in pickle format
- **Scaler**: StandardScaler for feature normalization
- **Feature Columns**: List of feature names for consistency

### Usage in Streamlit App:
1. Load model using `ModelLoader`
2. Load and apply scaler
3. Preprocess input data
4. Make predictions
5. Return probabilities and classifications

---

## Best Practices Implemented

1. **Data Validation**: Checks at each step
2. **Reproducibility**: Fixed random seeds
3. **Modularity**: Separate notebooks for each stage
4. **Documentation**: Clear code comments and structure
5. **Error Handling**: Graceful fallbacks
6. **Visualization**: Comprehensive charts and graphs
7. **Version Control**: Saved models and results

---

## Next Steps for Production

1. **Model Monitoring**: Track model performance over time
2. **A/B Testing**: Compare model versions
3. **Retraining Pipeline**: Periodic model updates
4. **Feature Store**: Centralized feature management
5. **Model Registry**: Version control for models
6. **API Development**: RESTful API for predictions
7. **Real-time Monitoring**: Alert on performance degradation

---

## Conclusion

This ML training pipeline provides a complete, reproducible workflow for fraud detection. Each notebook serves a specific purpose in the machine learning lifecycle, from data cleaning to model evaluation and forecasting. The modular design allows for easy updates and maintenance, while comprehensive evaluation ensures model quality.

