import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import pickle
import json
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """Advanced performance analytics for fake news detection models"""
    
    def __init__(self, models_dir="../../models", results_dir="../../results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.performance_history = []
        
    def load_models(self):
        """Load all available models"""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} not found")
            return
        
        for file in os.listdir(self.models_dir):
            if file.endswith('.pkl') and file != 'preprocessor.pkl':
                model_name = file.replace('.pkl', '')
                try:
                    with open(os.path.join(self.models_dir, file), 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {str(e)}")
    
    def comprehensive_model_evaluation(self, X_test, y_test, model_names=None):
        """Perform comprehensive evaluation of all models"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        
        for model_name in model_names:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found")
                continue
            
            logger.info(f"Evaluating {model_name}...")
            
            model = self.models[model_name]
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            roc_data = None
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
            
            # Precision-Recall curve
            pr_data = None
            if y_pred_proba is not None:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                pr_data = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': pr_auc}
            
            # Calibration curve
            calibration_data = None
            if y_pred_proba is not None:
                fraction_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10
                )
                calibration_data = {
                    'fraction_positives': fraction_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'roc_curve': roc_data,
                'precision_recall_curve': pr_data,
                'calibration_curve': calibration_data,
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        
        return results
    
    def create_performance_dashboard(self, evaluation_results):
        """Create comprehensive performance dashboard"""
        
        model_names = list(evaluation_results.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Model Accuracy Comparison', 'ROC Curves',
                'Precision-Recall Curves', 'Confusion Matrices',
                'Calibration Plots', 'F1-Score Comparison'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Model Accuracy Comparison
        accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy',
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. ROC Curves
        for model_name in model_names:
            roc_data = evaluation_results[model_name].get('roc_curve')
            if roc_data:
                fig.add_trace(
                    go.Scatter(
                        x=roc_data['fpr'], y=roc_data['tpr'],
                        mode='lines',
                        name=f"{model_name} (AUC={roc_data['auc']:.3f})"
                    ),
                    row=1, col=2
                )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'),
                      name='Random', showlegend=False),
            row=1, col=2
        )
        
        # 3. Precision-Recall Curves
        for model_name in model_names:
            pr_data = evaluation_results[model_name].get('precision_recall_curve')
            if pr_data:
                fig.add_trace(
                    go.Scatter(
                        x=pr_data['recall'], y=pr_data['precision'],
                        mode='lines',
                        name=f"{model_name} (AUC={pr_data['auc']:.3f})",
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Confusion Matrix (show first model as example)
        if model_names:
            cm = evaluation_results[model_names[0]]['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues',
                          text=cm, texttemplate="%{text}",
                          showscale=False),
                row=2, col=2
            )
        
        # 5. Calibration Plots
        for model_name in model_names:
            cal_data = evaluation_results[model_name].get('calibration_curve')
            if cal_data:
                fig.add_trace(
                    go.Scatter(
                        x=cal_data['mean_predicted_value'],
                        y=cal_data['fraction_positives'],
                        mode='lines+markers',
                        name=f"{model_name}",
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash'), name='Perfect Calibration',
                      showlegend=False),
            row=3, col=1
        )
        
        # 6. F1-Score Comparison
        f1_scores = [evaluation_results[name]['f1_score'] for name in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name='F1-Score',
                  marker_color='lightgreen'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Model Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def analyze_prediction_errors(self, X_test, y_test, evaluation_results, sample_texts=None):
        """Analyze prediction errors to identify patterns"""
        
        error_analysis = {}
        
        for model_name, results in evaluation_results.items():
            predictions = np.array(results['predictions'])
            probabilities = results.get('prediction_probabilities')
            
            # Find misclassified samples
            errors = y_test != predictions
            error_indices = np.where(errors)[0]
            
            # False positives and false negatives
            false_positives = np.where((y_test == 0) & (predictions == 1))[0]
            false_negatives = np.where((y_test == 1) & (predictions == 0))[0]
            
            # Confidence analysis for errors
            error_confidences = []
            if probabilities is not None:
                probs = np.array(probabilities)
                for idx in error_indices:
                    pred_conf = probs[idx] if predictions[idx] == 1 else 1 - probs[idx]
                    error_confidences.append(pred_conf)
            
            error_analysis[model_name] = {
                'total_errors': len(error_indices),
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives),
                'error_rate': len(error_indices) / len(y_test),
                'avg_error_confidence': np.mean(error_confidences) if error_confidences else None,
                'error_indices': error_indices.tolist(),
                'fp_indices': false_positives.tolist(),
                'fn_indices': false_negatives.tolist()
            }
        
        return error_analysis
    
    def create_error_analysis_report(self, error_analysis, sample_texts=None):
        """Create detailed error analysis report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'detailed_analysis': error_analysis
        }
        
        # Summary statistics
        total_models = len(error_analysis)
        avg_error_rate = np.mean([data['error_rate'] for data in error_analysis.values()])
        
        best_model = min(error_analysis.keys(), 
                        key=lambda x: error_analysis[x]['error_rate'])
        worst_model = max(error_analysis.keys(), 
                         key=lambda x: error_analysis[x]['error_rate'])
        
        report['summary'] = {
            'total_models_analyzed': total_models,
            'average_error_rate': avg_error_rate,
            'best_performing_model': best_model,
            'worst_performing_model': worst_model,
            'best_error_rate': error_analysis[best_model]['error_rate'],
            'worst_error_rate': error_analysis[worst_model]['error_rate']
        }
        
        return report
    
    def model_drift_analysis(self, current_results, historical_results):
        """Analyze model drift over time"""
        
        drift_analysis = {}
        
        for model_name in current_results.keys():
            if model_name not in historical_results:
                continue
            
            current_acc = current_results[model_name]['accuracy']
            historical_acc = [r['accuracy'] for r in historical_results[model_name]]
            
            current_f1 = current_results[model_name]['f1_score']
            historical_f1 = [r['f1_score'] for r in historical_results[model_name]]
            
            # Calculate drift metrics
            acc_drift = current_acc - np.mean(historical_acc)
            f1_drift = current_f1 - np.mean(historical_f1)
            
            # Statistical significance (simplified)
            acc_std = np.std(historical_acc) if len(historical_acc) > 1 else 0
            f1_std = np.std(historical_f1) if len(historical_f1) > 1 else 0
            
            drift_analysis[model_name] = {
                'accuracy_drift': acc_drift,
                'f1_drift': f1_drift,
                'accuracy_drift_significant': abs(acc_drift) > 2 * acc_std if acc_std > 0 else False,
                'f1_drift_significant': abs(f1_drift) > 2 * f1_std if f1_std > 0 else False,
                'drift_severity': 'high' if abs(acc_drift) > 0.1 or abs(f1_drift) > 0.1 else 'low'
            }
        
        return drift_analysis
    
    def generate_performance_insights(self, evaluation_results, error_analysis):
        """Generate actionable insights from performance analysis"""
        
        insights = []
        
        # Model ranking
        model_ranking = sorted(evaluation_results.keys(), 
                              key=lambda x: evaluation_results[x]['f1_score'], 
                              reverse=True)
        
        insights.append({
            'type': 'ranking',
            'title': 'Model Performance Ranking',
            'content': f"Best performing model: {model_ranking[0]} (F1: {evaluation_results[model_ranking[0]]['f1_score']:.3f})",
            'action': 'Consider using this model for production deployment'
        })
        
        # High error rate warning
        high_error_models = [name for name, data in error_analysis.items() 
                           if data['error_rate'] > 0.2]
        
        if high_error_models:
            insights.append({
                'type': 'warning',
                'title': 'High Error Rate Alert',
                'content': f"Models with >20% error rate: {', '.join(high_error_models)}",
                'action': 'Consider retraining or feature engineering'
            })
        
        # False positive/negative analysis
        for model_name, data in error_analysis.items():
            fp_rate = data['false_positives'] / (data['false_positives'] + data['false_negatives']) if (data['false_positives'] + data['false_negatives']) > 0 else 0
            
            if fp_rate > 0.6:
                insights.append({
                    'type': 'analysis',
                    'title': f'{model_name} - High False Positive Rate',
                    'content': f"Model tends to classify real news as fake ({fp_rate:.1%} of errors)",
                    'action': 'Adjust classification threshold or retrain with balanced data'
                })
        
        # Calibration insights
        for model_name, results in evaluation_results.items():
            cal_data = results.get('calibration_curve')
            if cal_data:
                # Check if model is well calibrated (simplified check)
                mean_diff = np.mean(np.abs(np.array(cal_data['fraction_positives']) - 
                                         np.array(cal_data['mean_predicted_value'])))
                
                if mean_diff > 0.1:
                    insights.append({
                        'type': 'calibration',
                        'title': f'{model_name} - Poor Calibration',
                        'content': f"Model confidence scores don't match actual accuracy",
                        'action': 'Consider probability calibration (Platt scaling or isotonic regression)'
                    })
        
        return insights
    
    def export_comprehensive_report(self, evaluation_results, error_analysis, 
                                  insights, filename=None):
        """Export comprehensive performance report"""
        
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_evaluated': list(evaluation_results.keys()),
                'report_type': 'comprehensive_performance_analysis'
            },
            'evaluation_results': evaluation_results,
            'error_analysis': error_analysis,
            'insights': insights,
            'summary_metrics': {
                'best_model': max(evaluation_results.keys(), 
                                key=lambda x: evaluation_results[x]['f1_score']),
                'average_accuracy': np.mean([r['accuracy'] for r in evaluation_results.values()]),
                'average_f1': np.mean([r['f1_score'] for r in evaluation_results.values()]),
                'total_insights': len(insights)
            }
        }
        
        # Save report
        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report exported to {filepath}")
        return filepath

# Example usage and testing
if __name__ == "__main__":
    # Initialize analytics
    analytics = PerformanceAnalytics()
    
    # Generate sample data for testing
    logger.info("Generating sample test data...")
    
    n_samples = 1000
    X_test = np.random.randn(n_samples, 100)  # 100 features
    y_test = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Create mock evaluation results
    mock_results = {}
    model_names = ['logistic_regression', 'random_forest', 'neural_network', 'ensemble']
    
    for model_name in model_names:
        # Simulate predictions
        y_pred = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
        y_pred_proba = np.random.beta(2, 2, size=n_samples)  # More realistic probability distribution
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Mock other data
        mock_results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Add ROC curve data
        if len(np.unique(y_test)) > 1:  # Need both classes for ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            mock_results[model_name]['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            }
        
        # Add PR curve data
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        mock_results[model_name]['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': pr_auc
        }
    
    # Perform error analysis
    logger.info("Performing error analysis...")
    error_analysis = analytics.analyze_prediction_errors(X_test, y_test, mock_results)
    
    # Generate insights
    logger.info("Generating performance insights...")
    insights = analytics.generate_performance_insights(mock_results, error_analysis)
    
    # Create performance dashboard
    logger.info("Creating performance dashboard...")
    dashboard = analytics.create_performance_dashboard(mock_results)
    
    # Print results
    print("\n=== Performance Analysis Results ===")
    for model_name, results in mock_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        if 'roc_curve' in results:
            print(f"  ROC-AUC: {results['roc_curve']['auc']:.4f}")
    
    print("\n=== Error Analysis ===")
    for model_name, analysis in error_analysis.items():
        print(f"\n{model_name}:")
        print(f"  Error Rate: {analysis['error_rate']:.4f}")
        print(f"  False Positives: {analysis['false_positives']}")
        print(f"  False Negatives: {analysis['false_negatives']}")
    
    print(f"\n=== Generated Insights ({len(insights)}) ===")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. [{insight['type'].upper()}] {insight['title']}")
        print(f"   {insight['content']}")
        print(f"   Action: {insight['action']}")
    
    # Export comprehensive report
    report_path = analytics.export_comprehensive_report(
        mock_results, error_analysis, insights
    )
    print(f"\nComprehensive report saved to: {report_path}")