import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    """Text preprocessing for baseline models"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_texts(self, texts):
        """Preprocess a list of texts"""
        return [self.clean_text(text) for text in texts]

class BaselineModels:
    """Baseline models for fake news detection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.preprocessor = TextPreprocessor()
        self.label_mapping = {
            'true': 5, 'mostly-true': 4, 'half-true': 3,
            'barely-true': 2, 'false': 1, 'pants-fire': 0
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def prepare_data(self, train_df, test_df, valid_df):
        """Prepare data for training"""
        
        # Combine train and validation for training (as per common practice)
        train_combined = pd.concat([train_df, valid_df], ignore_index=True)
        
        # Preprocess text
        print("Preprocessing text...")
        X_train_text = self.preprocessor.preprocess_texts(train_combined['statement'])
        X_test_text = self.preprocessor.preprocess_texts(test_df['statement'])
        
        # Encode labels
        y_train = [self.label_mapping.get(label, 0) for label in train_combined['label']]
        y_test = [self.label_mapping.get(label, 0) for label in test_df['label']]
        
        return X_train_text, X_test_text, y_train, y_test
    
    def create_tfidf_logistic_model(self):
        """Create TF-IDF + Logistic Regression pipeline"""
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'tfidf__max_features': [3000, 5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        return grid_search
    
    def create_tfidf_rf_model(self):
        """Create TF-IDF + Random Forest pipeline"""
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'tfidf__max_features': [3000, 5000],
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=5, scoring='f1_weighted',
            n_jobs=-1, verbose=1
        )
        
        return grid_search
    
    def train_all_models(self, X_train, y_train):
        """Train all baseline models"""
        
        print("=== TRAINING BASELINE MODELS ===")
        
        # Create and train TF-IDF + Logistic Regression
        print("\\n1. Training TF-IDF + Logistic Regression...")
        tfidf_lr = self.create_tfidf_logistic_model()
        tfidf_lr.fit(X_train, y_train)
        self.models['tfidf_logistic'] = tfidf_lr
        print(f"Best parameters: {tfidf_lr.best_params_}")
        print(f"Best cross-validation score: {tfidf_lr.best_score_:.4f}")
        
        # Create and train TF-IDF + Random Forest
        print("\\n2. Training TF-IDF + Random Forest...")
        tfidf_rf = self.create_tfidf_rf_model()
        tfidf_rf.fit(X_train, y_train)
        self.models['tfidf_rf'] = tfidf_rf
        print(f"Best parameters: {tfidf_rf.best_params_}")
        print(f"Best cross-validation score: {tfidf_rf.best_score_:.4f}")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        
        print("\\n=== MODEL EVALUATION ===")
        class_names = list(self.label_mapping.keys())
        
        for name, model in self.models.items():
            print(f"\\n--- {name.upper()} RESULTS ---")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=class_names,
                                      zero_division=0))
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'model': model
            }
        
        return self.results
    
    def plot_results(self, y_test):
        """Create visualizations for model results"""
        
        class_names = list(self.label_mapping.keys())
        n_models = len(self.results)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, results) in enumerate(self.results.items()):
            
            # Confusion Matrix
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[0, idx])
            axes[0, idx].set_title(f'{name.replace("_", " ").title()}\\nConfusion Matrix')
            axes[0, idx].set_xlabel('Predicted')
            axes[0, idx].set_ylabel('True')
            
            # Accuracy comparison
            accuracy = results['accuracy']
            axes[1, idx].bar(['Accuracy'], [accuracy], color='skyblue', alpha=0.7)
            axes[1, idx].set_ylim(0, 1)
            axes[1, idx].set_title(f'{name.replace("_", " ").title()}\\nAccuracy: {accuracy:.3f}')
            axes[1, idx].set_ylabel('Score')
            
            # Add text annotation
            axes[1, idx].text(0, accuracy + 0.02, f'{accuracy:.3f}', 
                             ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/baseline_models_evaluation_0149.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models"""
        import os
        os.makedirs('models/baseline', exist_ok=True)
        
        for name, model in self.models.items():
            filename = f'models/baseline/{name}_0149.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved: {filename}")
    
    def create_performance_report(self):
        """Create detailed performance report"""
        
        report_data = []
        for name, results in self.results.items():
            report_data.append({
                'Model': name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Best_CV_Score': self.models[name].best_score_
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_df.to_csv('results/reports/baseline_performance_0149.csv', index=False)
        print("\\nPerformance Report:")
        print(report_df.to_string(index=False))
        
        return report_df
    
    def load_models(self):
        """Load pre-trained models from disk"""
        import os
        
        model_files = {
            'tfidf_logistic': 'models/baseline/tfidf_logistic_0149.pkl',
            'tfidf_rf': 'models/baseline/tfidf_rf_0149.pkl'
        }
        
        loaded_models = {}
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        model = pickle.load(f)
                    loaded_models[name] = model
                    print(f"✅ Loaded {name} model from {filepath}")
                except Exception as e:
                    print(f"❌ Error loading {name} model: {e}")
            else:
                print(f"⚠️ Model file not found: {filepath}")
        
        if loaded_models:
            self.models.update(loaded_models)
            print(f"✅ Successfully loaded {len(loaded_models)} models")
        else:
            print("⚠️ No models were loaded. You may need to train models first.")
        
        return loaded_models
    
    def predict(self, texts):
        """Make predictions using loaded models"""
        if not self.models:
            print("❌ No models loaded. Please load models first.")
            return None
        
        # Use the first available model for prediction
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        
        try:
            # Preprocess texts
            processed_texts = self.preprocessor.preprocess_texts(texts)
            
            # Make predictions
            predictions = model.predict(processed_texts)
            
            # Convert numeric predictions back to labels
            label_predictions = [self.reverse_label_mapping.get(pred, 'unknown') for pred in predictions]
            
            return label_predictions
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None

# Usage example
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('data/processed/train_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv') 
    valid_df = pd.read_csv('data/processed/valid_processed.csv')
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Prepare data
    X_train, X_test, y_train, y_test = baseline.prepare_data(train_df, test_df, valid_df)
    
    # Train models
    models = baseline.train_all_models(X_train, y_train)
    
    # Evaluate models
    results = baseline.evaluate_models(X_test, y_test)
    
    # Create visualizations
    baseline.plot_results(y_test)
    
    # Save models and create report
    baseline.save_models()
    baseline.create_performance_report()