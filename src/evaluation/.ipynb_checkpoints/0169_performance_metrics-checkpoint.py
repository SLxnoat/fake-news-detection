import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
import os

class PerformanceAnalyzer:
    def __init__(self, class_names=None):
        self.class_names = class_names or ['true', 'mostly-true', 'half-true', 
                                          'barely-true', 'false', 'pants-fire']
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """Calculate comprehensive performance metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names,
                                     output_dict=True, zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Add ROC AUC if probabilities provided
        if y_proba is not None:
            try:
                # For multiclass, calculate macro-average ROC AUC
                roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                results['roc_auc'] = roc_auc
            except ValueError:
                results['roc_auc'] = None
        
        self.results[model_name] = results
        return results
    
    def plot_confusion_matrix(self, model_name, save_path=None, figsize=(10, 8)):
        """Plot confusion matrix for a model"""
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
            annotations.append(row)
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_class_metrics(self, model_name, save_path=None):
        """Plot per-class precision, recall, and F1-score"""
        if model_name not in self.results:
            print(f"No results found for {model_name}")
            return
        
        results = self.results[model_name]
        
        # Create DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': results['precision_per_class'],
            'Recall': results['recall_per_class'],
            'F1-Score': results['f1_per_class']
        })
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, metrics_df['Precision'], width, 
                      label='Precision', alpha=0.8)
        bars2 = ax.bar(x, metrics_df['Recall'], width, 
                      label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, 
                      label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance Metrics - {model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_models(self, save_path=None):
        """Compare all evaluated models"""
        if not self.results:
            print("No models to compare")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision_weighted'],
                'Recall': results['recall_weighted'],
                'F1-Score': results['f1_weighted']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i//2, i%2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=color, alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return comparison_df
    
    def generate_report(self, save_path=None):
        """Generate comprehensive performance report"""
        if not self.results:
            print("No results to report")
            return
        
        report = "# Model Performance Report\n\n"
        
        for model_name, results in self.results.items():
            report += f"## {model_name}\n\n"
            report += f"**Overall Performance:**\n"
            report += f"- Accuracy: {results['accuracy']:.4f}\n"
            report += f"- Weighted Precision: {results['precision_weighted']:.4f}\n"
            report += f"- Weighted Recall: {results['recall_weighted']:.4f}\n"
            report += f"- Weighted F1-Score: {results['f1_weighted']:.4f}\n"
            
            if 'roc_auc' in results and results['roc_auc'] is not None:
                report += f"- ROC AUC: {results['roc_auc']:.4f}\n"
            
            report += "\n**Per-Class Performance:**\n"
            for i, class_name in enumerate(self.class_names):
                if i < len(results['precision_per_class']):
                    report += f"- {class_name}:\n"
                    report += f"  - Precision: {results['precision_per_class'][i]:.4f}\n"
                    report += f"  - Recall: {results['recall_per_class'][i]:.4f}\n"
                    report += f"  - F1-Score: {results['f1_per_class'][i]:.4f}\n"
                    report += f"  - Support: {results['support_per_class'][i]}\n"
            
            report += "\n---\n\n"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report