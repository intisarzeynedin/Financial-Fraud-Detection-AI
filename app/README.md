# Streamlit Fraud Detection Dashboard

Multi-page Streamlit application for credit card fraud detection visualization and prediction.

## Pages

1. **Overview** (`1_Overview.py`) - Project summary and quick statistics
2. **Data Exploration** (`2_Data_Exploration.py`) - Interactive data visualization
3. **ML Prediction** (`3_ML_Prediction.py`) - Real-time fraud prediction
4. **ML Results** (`4_ML_Results.py`) - Model evaluation metrics and charts
5. **Forecasting** (`5_Forecasting.py`) - Future fraud trend predictions

## Running the App

```bash
cd transactions
streamlit run app/app.py
```

## Requirements

- All notebooks must be run first:
  1. `01b_DataCleaning.ipynb` - Clean data
  2. `02_Preprocessing_FeatureEngineering.ipynb` - Preprocess data
  3. `03_ML_Training.ipynb` - Train models
  4. `04_ML_Evaluation.ipynb` - Evaluate models
  5. `05_Forecasting_Analysis.ipynb` - Generate forecasts (optional)

## Features

- **Real-time Predictions**: Make fraud predictions on new transactions
- **Interactive Visualizations**: Explore data with interactive charts
- **Model Comparison**: Compare all trained models
- **Future Forecasting**: Predict future fraud trends
- **Export Results**: Download predictions and forecasts

