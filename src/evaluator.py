"""
Model Evaluation Module
Evaluates models using various metrics: F1, AUC, AUPRC, precision, recall.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """Evaluate ML models for fraud detection."""
    
    def __init__(self, reports_dir='reports'):
        """
        Initialize evaluator.
        
        Args:
            reports_dir: Directory to save evaluation reports
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*50}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Print results
        print(f"\nMetrics for {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def compare_models(self):
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            print("No models evaluated yet.")
            return None
        
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        
        return df
    
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for all models."""
        if not self.results:
            print("No models evaluated yet.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            filepath = self.reports_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {filepath}")
        else:
            plt.savefig(self.reports_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_pr_curves(self, save_path=None):
        """Plot Precision-Recall curves for all models."""
        if not self.results:
            print("No models evaluated yet.")
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            filepath = self.reports_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"PR curves saved to {filepath}")
        else:
            plt.savefig(self.reports_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for all models."""
        if not self.results:
            print("No models evaluated yet.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['metrics']['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'])
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            filepath = self.reports_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {filepath}")
        else:
            plt.savefig(self.reports_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot bar chart comparing metrics across models."""
        if not self.results:
            print("No models evaluated yet.")
            return
        
        comparison_df = self.compare_models()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            comparison_df.plot(x='Model', y=metric, kind='barh', ax=ax, color='steelblue')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('')
            ax.grid(alpha=0.3, axis='x')
            ax.legend().remove()
        
        plt.tight_layout()
        
        if save_path:
            filepath = self.reports_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {filepath}")
        else:
            plt.savefig(self.reports_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save_results(self, filename='evaluation_results.csv'):
        """Save evaluation results to CSV."""
        comparison_df = self.compare_models()
        if comparison_df is not None:
            filepath = self.reports_dir / filename
            comparison_df.to_csv(filepath, index=False)
            print(f"Results saved to {filepath}")
            return filepath
        return None

