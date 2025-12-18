"""
Model Loader Utility
Loads trained models and scaler for use in Streamlit app.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class ModelLoader:
    """Utility class to load trained models and make predictions."""
    
    def __init__(self, models_dir='models'):
        """
        Initialize ModelLoader.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        
    def load_scaler(self):
        """Load the fitted scaler."""
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        return self.scaler
    
    def load_feature_columns(self):
        """Load feature column names."""
        feature_cols_path = self.models_dir / 'feature_columns.pkl'
        if feature_cols_path.exists():
            self.feature_columns = joblib.load(feature_cols_path)
            print(f"Feature columns loaded from {feature_cols_path}")
        else:
            raise FileNotFoundError(f"Feature columns not found at {feature_cols_path}")
        return self.feature_columns
    
    def load_model(self, model_name):
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model (without .pkl extension)
            
        Returns:
            Loaded model
        """
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_path} not found")
        
        model = joblib.load(model_path)
        self.models[model_name] = model
        print(f"Model {model_name} loaded from {model_path}")
        return model
    
    def load_all_models(self):
        """Load all available models."""
        model_files = list(self.models_dir.glob("*.pkl"))
        model_files = [f for f in model_files if f.stem not in ['scaler', 'feature_columns']]
        
        for model_file in model_files:
            model_name = model_file.stem
            self.load_model(model_name)
        
        return self.models
    
    def predict(self, model_name, X, return_proba=False):
        """
        Make predictions using a loaded model.
        
        Args:
            model_name: Name of the model to use
            X: Input features (DataFrame or array)
            return_proba: Whether to return probability scores
            
        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        if model_name not in self.models:
            self.load_model(model_name)
        
        model = self.models[model_name]
        
        # Ensure X is a DataFrame with correct columns
        if isinstance(X, np.ndarray):
            if self.feature_columns is None:
                self.load_feature_columns()
            X = pd.DataFrame(X, columns=self.feature_columns)
        
        # Scale features if scaler is available
        if self.scaler is None:
            self.load_scaler()
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns
        )
        
        if return_proba:
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities
        else:
            predictions = model.predict(X_scaled)
            return predictions
    
    def predict_single(self, model_name, transaction_data):
        """
        Predict for a single transaction.
        
        Args:
            model_name: Name of the model to use
            transaction_data: Dictionary or Series with transaction features
            
        Returns:
            Dictionary with prediction and probability
        """
        # Convert to DataFrame
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, pd.Series):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data
        
        # Ensure feature columns are in correct order
        if self.feature_columns is None:
            self.load_feature_columns()
        
        # Reorder columns to match training data
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")
        
        df = df[self.feature_columns]
        
        # Make prediction
        prediction, probability = self.predict(model_name, df, return_proba=True)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'is_fraud': bool(prediction[0] == 1),
            'fraud_probability': float(probability[0])
        }
    
    def get_available_models(self):
        """Get list of available model files."""
        model_files = list(self.models_dir.glob("*.pkl"))
        model_files = [f.stem for f in model_files if f.stem not in ['scaler', 'feature_columns']]
        return model_files

