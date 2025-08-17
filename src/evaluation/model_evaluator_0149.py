import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or [
            'pants-fire', 'false', 'barely-true', 
            'half-true', 'mostly-true', 'true'
        ]
        self.results = {}
        self.cv_results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive single model evaluation"""
        
        print(f"\\n=== EVALUATING {model_name.upper()} ===")
        
        # Basic predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        return self.results[model_name]
    
    def cross_validate_model(self, model, X, y, model_name, cv=5):
        """Perform cross-validation evaluation"""
        
        print(f"\\n=== CROSS-VALIDATION: {model_name.upper()} ===")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Different scoring metrics
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_scores[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{metric.replace('_', ' ').title()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        self.cv_results[model_name] = cv_scores
        return cv_scores
    
    def compare_models(self, save_comparison=True):
        """Compare all evaluated models"""
        
        if not self.results:
            print("No models evaluated yet!")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision_weighted'],
                'Recall': results['recall_weighted'],
                'F1-Score': results['f1_weighted']
            }
            
            # Add CV results if available
            if model_name in self.cv_results:
                cv = self.cv_results[model_name]
                row['CV_Accuracy_Mean'] = cv['accuracy']['mean']
                row['CV_Accuracy_Std'] = cv['accuracy']['std']
                row['CV_F1_Mean'] = cv['f1_weighted']['mean']
                row['CV_F1_Std'] = cv['f1_weighted']['std']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_comparison:
            comparison_df.to_csv('results/reports/model_comparison_0149.csv', index=False)
        
        print("\\n=== MODEL COMPARISON ===")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_confusion_matrices(self, save_plot=True):
        """Plot confusion matrices for all models"""
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\\nAccuracy: {results["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('results/figures/confusion_matrices_0149.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_comparison(self, save_plot=True):
        """Create performance comparison visualization"""
        
        if not self.results:
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [self.results[model][metric] for model in models]
            model_names = [model.replace('_', ' ').title() for model in models]
            
            bars = axes[idx].bar(model_names, values, alpha=0.7, 
                               color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(models)])
            axes[idx].set_title(f'{metric_name} Comparison')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_ylim(0, 1)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('results/figures/performance_comparison_0149.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_detailed_report(self):
        """Generate detailed evaluation report"""
        
        report_content = []
        report_content.append("# Baseline Models Evaluation Report")
        report_content.append(f"**Evaluator**: Member 0149")
        report_content.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        report_content.append("\\n## Model Performance Summary")
        
        for model_name, results in self.results.items():
            report_content.append(f"\\n### {model_name.replace('_', ' ').title()}")
            report_content.append(f"- **Accuracy**: {results['accuracy']:.4f}")
            report_content.append(f"- **Weighted Precision**: {results['precision_weighted']:.4f}")
            report_content.append(f"- **Weighted Recall**: {results['recall_weighted']:.4f}")
            report_content.append(f"- **Weighted F1-Score**: {results['f1_weighted']:.4f}")
            
            if model_name in self.cv_results:
                cv = self.cv_results[model_name]
                report_content.append(f"- **CV Accuracy**: {cv['accuracy']['mean']:.4f} (±{cv['accuracy']['std']:.4f})")
                report_content.append(f"- **CV F1-Score**: {cv['f1_weighted']['mean']:.4f} (±{cv['f1_weighted']['std']:.4f})")
        
        report_content.append("\\n## Per-Class Performance")
        for model_name, results in self.results.items():
            report_content.append(f"\\n### {model_name.replace('_', ' ').title()} - Per Class")
            for i, class_name in enumerate(self.class_names):
                if i < len(results['precision_per_class']):
                    p = results['precision_per_class'][i]
                    r = results['recall_per_class'][i] 
                    f1 = results['f1_per_class'][i]
                    report_content.append(f"- **{class_name}**: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        
        # Save report
        with open('results/reports/detailed_evaluation_report_0149.md', 'w') as f:
            f.write('\\n'.join(report_content))
        
        print("Detailed report saved to: results/reports/detailed_evaluation_report_0149.md")
        return '\\n'.join(report_content)

# Usage example
if __name__ == "__main__":
    # This would be called from your main training script
    pass