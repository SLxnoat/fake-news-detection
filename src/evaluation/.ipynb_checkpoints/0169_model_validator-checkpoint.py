import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class ModelValidator:
    def __init__(self, cv_folds=5, test_size=0.2, random_state=42):
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        
    def cross_validate(self, model, X, y, model_name):
        """Perform cross-validation on a model"""
        print(f"Cross-validating {model_name}...")
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # Get cross-validation scores
        cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted')
        cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted')
        
        # Store results
        self.results[model_name] = {
            'cv_accuracy': cv_accuracy,
            'cv_f1': cv_f1,
            'cv_precision': cv_precision,
            'cv_recall': cv_recall,
            'mean_accuracy': cv_accuracy.mean(),
            'std_accuracy': cv_accuracy.std(),
            'mean_f1': cv_f1.mean(),
            'std_f1': cv_f1.std()
        }
        
        print(f"Results for {model_name}:")
        print(f"  Accuracy: {cv_accuracy.mean():.3f} (+/- {cv_accuracy.std()*2:.3f})")
        print(f"  F1-Score: {cv_f1.mean():.3f} (+/- {cv_f1.std()*2:.3f})")
        
        return self.results[model_name]
    
    def final_evaluation(self, model, X_test, y_test, model_name, class_names=None):
        """Final evaluation on test set"""
        print(f"Final evaluation for {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        final_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }
        
        # Store in results
        if model_name in self.results:
            self.results[model_name].update(final_results)
        else:
            self.results[model_name] = final_results
            
        return final_results
    
    def save_results(self, filepath):
        """Save validation results"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filepath}")