"""
Data Loading and Preprocessing Module
Handles loading data from CSV/JSON formats and preprocessing for ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json


class DataLoader:
    """Load and preprocess transaction data for fraud detection."""
    
    def __init__(self, data_dir='data'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_csv_data(self, filename='creditcard.csv'):
        """
        Load data from CSV file.
        
        Args:
            filename: Name of CSV file
            
        Returns:
            DataFrame with transaction data
        """
        filepath = self.data_dir / filename
        print(f"Loading data from {filepath}...")
        
        # Load CSV in chunks if too large
        try:
            df = pd.read_csv(filepath)
        except MemoryError:
            print("File too large, loading in chunks...")
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=10000):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        
        print(f"Loaded {len(df)} transactions")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        
        return df
    
    def load_json_data(self, filename='transactions.json'):
        """
        Load data from JSON file.
        
        Args:
            filename: Name of JSON file
            
        Returns:
            DataFrame with transaction data
        """
        filepath = self.data_dir / filename
        print(f"Loading data from {filepath}...")
        
        # Try to load JSON - handle large files
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        except (MemoryError, json.JSONDecodeError) as e:
            print(f"Error loading JSON: {e}")
            # Try reading line by line
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            df = pd.DataFrame(data)
        
        print(f"Loaded {len(df)} transactions from JSON")
        return df
    
    def preprocess_data(self, df, target_col='Class', test_size=0.2, random_state=42):
        """
        Preprocess data for ML training.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test, feature_columns
        """
        print("\nPreprocessing data...")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Get feature columns (exclude Time and Class)
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'Time']]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        print(f"Features: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols[:5]}... (showing first 5)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Maintain class distribution
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"  - Fraud: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"Test set: {len(X_test)} samples")
        print(f"  - Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
        
        # Scale features (fit on training data only)
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def get_scaler(self):
        """Get the fitted scaler for later use."""
        return self.scaler
    
    def transform_new_data(self, X):
        """
        Transform new data using fitted scaler.
        
        Args:
            X: New data to transform
            
        Returns:
            Scaled data
        """
        if self.feature_columns is None:
            raise ValueError("Scaler not fitted. Call preprocess_data first.")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns
        )
        return X_scaled

