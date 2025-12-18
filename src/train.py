"""
Main Training Script
Trains multiple models for fraud detection and evaluates them.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.models import FraudDetectionModels
from src.evaluator import ModelEvaluator


def main():
    """Main training pipeline."""
    print("="*70)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Initialize components
    data_loader = DataLoader(data_dir='data')
    models = FraudDetectionModels(models_dir='models')
    evaluator = ModelEvaluator(reports_dir='reports')
    
    # Load and preprocess data
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    df = data_loader.load_csv_data('creditcard.csv')
    X_train, X_test, y_train, y_test, feature_cols = data_loader.preprocess_data(df)
    
    # Train baseline models
    print("\n" + "="*70)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("="*70)
    
    # Logistic Regression - with class weights
    lr_model = models.logistic_regression(
        X_train, y_train, 
        use_smote=False, 
        class_weight='balanced'
    )
    models.save_model('logistic_regression', lr_model)
    
    # Logistic Regression - with SMOTE
    lr_smote_model = models.logistic_regression(
        X_train, y_train, 
        use_smote=True, 
        class_weight=None
    )
    models.save_model('logistic_regression_smote', lr_smote_model)
    
    # Decision Tree - with class weights
    dt_model = models.decision_tree(
        X_train, y_train, 
        use_smote=False, 
        class_weight='balanced'
    )
    models.save_model('decision_tree', dt_model)
    
    # Train advanced models
    print("\n" + "="*70)
    print("STEP 3: TRAINING ADVANCED MODELS")
    print("="*70)
    
    # Random Forest
    rf_model = models.random_forest(
        X_train, y_train, 
        use_smote=False, 
        class_weight='balanced',
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )
    models.save_model('random_forest', rf_model)
    
    # XGBoost
    xgb_model = models.xgboost_model(
        X_train, y_train, 
        use_smote=False,
        tune_hyperparameters=False  # Set to True for hyperparameter tuning
    )
    models.save_model('xgboost', xgb_model)
    
    # Evaluate all models
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    evaluator.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression (Class Weights)')
    evaluator.evaluate_model(lr_smote_model, X_test, y_test, 'Logistic Regression (SMOTE)')
    evaluator.evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
    evaluator.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    evaluator.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    
    # Compare models
    print("\n" + "="*70)
    print("STEP 5: MODEL COMPARISON")
    print("="*70)
    
    comparison_df = evaluator.compare_models()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*70)
    
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metrics_comparison()
    
    # Save results
    evaluator.save_results()
    
    # Save scaler for later use
    import joblib
    scaler = data_loader.get_scaler()
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    print("\nScaler and feature columns saved to models/")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved in: models/")
    print("Reports saved in: reports/")
    print("\nBest model based on F1-Score:")
    if comparison_df is not None and len(comparison_df) > 0:
        best_model = comparison_df.iloc[0]
        print(f"  Model: {best_model['Model']}")
        print(f"  F1-Score: {best_model['F1-Score']:.4f}")
        print(f"  ROC-AUC: {best_model['ROC-AUC']:.4f}")


if __name__ == '__main__':
    main()

