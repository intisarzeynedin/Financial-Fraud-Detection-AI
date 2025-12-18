"""
Machine Learning Models Module
Implements baseline and advanced models for fraud detection.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import numpy as np
import joblib
from pathlib import Path


class FraudDetectionModels:
    """Collection of ML models for fraud detection."""
    
    def __init__(self, models_dir='models'):
        """
        Initialize models.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.trained_models = {}
        self.model_results = {}
        
    def logistic_regression(self, X_train, y_train, use_smote=False, class_weight='balanced'):
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE oversampling
            class_weight: Class weight strategy ('balanced' or None)
            
        Returns:
            Trained model
        """
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        if use_smote:
            print("Using SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Samples: {len(X_resampled)}, Fraud: {y_resampled.sum()}")
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight=None
            )
            model.fit(X_resampled, y_resampled)
        else:
            print(f"Using class_weight={class_weight}")
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight=class_weight
            )
            model.fit(X_train, y_train)
        
        self.trained_models['logistic_regression'] = model
        return model
    
    def decision_tree(self, X_train, y_train, use_smote=False, class_weight='balanced'):
        """
        Train Decision Tree model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE oversampling
            class_weight: Class weight strategy ('balanced' or None)
            
        Returns:
            Trained model
        """
        print("\n" + "="*50)
        print("Training Decision Tree...")
        print("="*50)
        
        if use_smote:
            print("Using SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Samples: {len(X_resampled)}, Fraud: {y_resampled.sum()}")
            model = DecisionTreeClassifier(
                random_state=42,
                class_weight=None,
                max_depth=10
            )
            model.fit(X_resampled, y_resampled)
        else:
            print(f"Using class_weight={class_weight}")
            model = DecisionTreeClassifier(
                random_state=42,
                class_weight=class_weight,
                max_depth=10
            )
            model.fit(X_train, y_train)
        
        self.trained_models['decision_tree'] = model
        return model
    
    def random_forest(self, X_train, y_train, use_smote=False, 
                     class_weight='balanced', tune_hyperparameters=False):
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE oversampling
            class_weight: Class weight strategy ('balanced' or None)
            tune_hyperparameters: Whether to perform grid search
            
        Returns:
            Trained model
        """
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            }
            
            base_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, 
                scoring='f1', n_jobs=-1, verbose=1
            )
            
            if use_smote:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                grid_search.fit(X_resampled, y_resampled)
            else:
                grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            if use_smote:
                print("Using SMOTE for class balancing...")
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Samples: {len(X_resampled)}, Fraud: {y_resampled.sum()}")
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight=None,
                    n_jobs=-1
                )
                model.fit(X_resampled, y_resampled)
            else:
                print(f"Using class_weight={class_weight}")
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight=class_weight,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
        
        self.trained_models['random_forest'] = model
        return model
    
    def xgboost_model(self, X_train, y_train, use_smote=False, tune_hyperparameters=False):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE oversampling
            tune_hyperparameters: Whether to perform grid search
            
        Returns:
            Trained model
        """
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        if tune_hyperparameters:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'scale_pos_weight': [1, len(y_train[y_train==0])/len(y_train[y_train==1])]
            }
            
            base_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3,
                scoring='f1', n_jobs=-1, verbose=1
            )
            
            if use_smote:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                grid_search.fit(X_resampled, y_resampled)
            else:
                grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
            
            if use_smote:
                print("Using SMOTE for class balancing...")
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE - Samples: {len(X_resampled)}, Fraud: {y_resampled.sum()}")
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                model.fit(X_resampled, y_resampled)
            else:
                print(f"Using scale_pos_weight={scale_pos_weight:.2f}")
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                model.fit(X_train, y_train)
        
        self.trained_models['xgboost'] = model
        return model
    
    def save_model(self, model_name, model):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model
            model: Trained model object
        """
        filepath = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name):
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model
        """
        filepath = self.models_dir / f"{model_name}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Model {filepath} not found")
        
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_model(self, model_name):
        """Get trained model by name."""
        return self.trained_models.get(model_name)

