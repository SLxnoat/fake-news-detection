import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFakeNewsProcessor:
    """
    Advanced processing pipeline for fake news detection
    Includes feature engineering, model selection, and ensemble methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.preprocessors = {}
        self.models = {}
        self.ensemble_model = None
        self.feature_names = []
        self.performance_metrics = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration parameters"""
        return {
            'text_processing': {
                'max_features': 10000,
                'ngram_range': (1, 3),
                'min_df': 2,
                'max_df': 0.95,
                'sublinear_tf': True
            },
            'feature_engineering': {
                'include_readability': True,
                'include_sentiment': True,
                'include_pos_tags': True,
                'include_named_entities': True
            },
            'models': {
                'logistic_regression': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                },
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'gradient_boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                },
                'neural_network': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'ensemble': {
                'voting': 'soft',
                'use_stacking': True
            },
            'validation': {
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            }
        }
    
    def advanced_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced features from text and metadata"""
        logger.info("Starting advanced feature extraction...")
        
        features_df = df.copy()
        
        # Text-based features
        if 'statement' in df.columns:
            # Basic text statistics
            features_df['text_length'] = df['statement'].str.len()
            features_df['word_count'] = df['statement'].str.split().str.len()
            features_df['sentence_count'] = df['statement'].str.count(r'\.')
            features_df['exclamation_count'] = df['statement'].str.count('!')
            features_df['question_count'] = df['statement'].str.count(r'\?')
            features_df['capital_ratio'] = df['statement'].apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )
            
            # Advanced linguistic features
            features_df['avg_word_length'] = df['statement'].apply(
                lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
            )
            features_df['unique_word_ratio'] = df['statement'].apply(
                lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
            )
            
            # Readability features (simplified)
            features_df['readability_score'] = df['statement'].apply(self._calculate_readability)
            
            # Sentiment features (simplified)
            features_df['sentiment_polarity'] = df['statement'].apply(self._calculate_sentiment)
            
            # Named entity features (simplified)
            features_df['entity_count'] = df['statement'].apply(self._count_entities)
            
        # Metadata features
        if 'speaker' in df.columns:
            features_df['has_speaker'] = (~df['speaker'].isna()).astype(int)
            features_df['speaker_length'] = df['speaker'].fillna('').str.len()
        
        if 'job' in df.columns:
            # Job categories
            political_jobs = ['senator', 'governor', 'congressman', 'mayor', 'politician']
            media_jobs = ['journalist', 'reporter', 'anchor', 'correspondent']
            
            features_df['is_politician'] = df['job'].fillna('').str.lower().apply(
                lambda x: any(job in x for job in political_jobs)
            ).astype(int)
            
            features_df['is_media'] = df['job'].fillna('').str.lower().apply(
                lambda x: any(job in x for job in media_jobs)
            ).astype(int)
        
        # Context features
        if 'context' in df.columns:
            context_mapping = {
                'debate': 1, 'interview': 2, 'rally': 3, 
                'statement': 4, 'social-media': 5
            }
            features_df['context_numeric'] = df['context'].map(context_mapping).fillna(0)
        
        # Party affiliation features
        if 'party' in df.columns:
            features_df['is_democrat'] = (df['party'] == 'democrat').astype(int)
            features_df['is_republican'] = (df['party'] == 'republican').astype(int)
            features_df['is_independent'] = (df['party'] == 'none').astype(int)
        
        logger.info(f"Feature extraction completed. Shape: {features_df.shape}")
        return features_df
    
    def _calculate_readability(self, text: str) -> float:
        """Simplified readability score"""
        if not text:
            return 0
        
        words = text.split()
        sentences = text.split('.')
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simplified Flesch score
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
        return max(0, min(100, readability))
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simplified sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0
        
        return (positive_count - negative_count) / len(words)
    
    def _count_entities(self, text: str) -> int:
        """Simplified named entity counting"""
        # Simple heuristic: count capitalized words that aren't at sentence start
        import re
        
        sentences = text.split('.')
        entity_count = 0
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 1:  # Skip first word of each sentence
                entity_count += sum(1 for word in words[1:] if word[0].isupper())
        
        return entity_count
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create comprehensive preprocessing pipeline"""
        logger.info("Creating preprocessing pipeline...")
        
        # Identify column types
        text_columns = ['statement']
        numeric_columns = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        categorical_columns = [col for col in X.columns if col not in text_columns + numeric_columns]
        
        # Text preprocessing
        text_transformer = TfidfVectorizer(
            max_features=self.config['text_processing']['max_features'],
            ngram_range=self.config['text_processing']['ngram_range'],
            min_df=self.config['text_processing']['min_df'],
            max_df=self.config['text_processing']['max_df'],
            sublinear_tf=self.config['text_processing']['sublinear_tf'],
            stop_words='english'
        )
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, text_columns[0] if text_columns else []),
                ('num', StandardScaler(), numeric_columns),
                ('cat', 'passthrough', categorical_columns)  # Will be handled separately
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def train_individual_models(self, X_processed, y) -> Dict[str, Any]:
        """Train individual models with hyperparameter optimization"""
        logger.info("Training individual models...")
        
        models = {}
        
        # Logistic Regression
        if 'logistic_regression' in self.config['models']:
            logger.info("Training Logistic Regression...")
            lr_params = self.config['models']['logistic_regression']
            lr = LogisticRegression(random_state=self.config['validation']['random_state'])
            lr_grid = GridSearchCV(lr, lr_params, cv=self.config['validation']['cv_folds'], 
                                  scoring='roc_auc', n_jobs=-1)
            lr_grid.fit(X_processed, y)
            models['logistic_regression'] = lr_grid.best_estimator_
            logger.info(f"LR best params: {lr_grid.best_params_}")
        
        # Random Forest
        if 'random_forest' in self.config['models']:
            logger.info("Training Random Forest...")
            rf_params = self.config['models']['random_forest']
            rf = RandomForestClassifier(random_state=self.config['validation']['random_state'])
            rf_grid = GridSearchCV(rf, rf_params, cv=self.config['validation']['cv_folds'], 
                                  scoring='roc_auc', n_jobs=-1)
            rf_grid.fit(X_processed, y)
            models['random_forest'] = rf_grid.best_estimator_
            logger.info(f"RF best params: {rf_grid.best_params_}")
        
        # Gradient Boosting
        if 'gradient_boosting' in self.config['models']:
            logger.info("Training Gradient Boosting...")
            gb_params = self.config['models']['gradient_boosting']
            gb = GradientBoostingClassifier(random_state=self.config['validation']['random_state'])
            gb_grid = GridSearchCV(gb, gb_params, cv=self.config['validation']['cv_folds'], 
                                  scoring='roc_auc', n_jobs=-1)
            gb_grid.fit(X_processed, y)
            models['gradient_boosting'] = gb_grid.best_estimator_
            logger.info(f"GB best params: {gb_grid.best_params_}")
        
        # Neural Network (with reduced search space for speed)
        if 'neural_network' in self.config['models']:
            logger.info("Training Neural Network...")
            nn_params = self.config['models']['neural_network']
            nn = MLPClassifier(random_state=self.config['validation']['random_state'], max_iter=500)
            # Use smaller param grid for NN to avoid long training time
            nn_params_reduced = {k: v[:2] if isinstance(v, list) else v for k, v in nn_params.items()}
            nn_grid = GridSearchCV(nn, nn_params_reduced, cv=3,  # Reduced CV folds for NN
                                  scoring='roc_auc', n_jobs=-1)
            nn_grid.fit(X_processed, y)
            models['neural_network'] = nn_grid.best_estimator_
            logger.info(f"NN best params: {nn_grid.best_params_}")
        
        return models
    
    def create_ensemble_model(self, individual_models: Dict[str, Any]) -> VotingClassifier:
        """Create ensemble model from individual models"""
        logger.info("Creating ensemble model...")
        
        estimators = [(name, model) for name, model in individual_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.config['ensemble']['voting']
        )
        
        return ensemble
    
    def evaluate_models(self, models: Dict[str, Any], X_test, y_test) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Classification metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
            }
            
            logger.info(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
                       f"F1: {results[name]['f1_score']:.4f}, "
                       f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        return results
    
    def save_models(self, models: Dict[str, Any], preprocessor: ColumnTransformer, 
                   output_dir: str = "../../models") -> None:
        """Save trained models and preprocessor"""
        logger.info(f"Saving models to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
        
        # Save individual models
        for name, model in models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))
        
        # Save ensemble if exists
        if self.ensemble_model:
            joblib.dump(self.ensemble_model, os.path.join(output_dir, 'ensemble.pkl'))
        
        # Save configuration
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Models saved successfully!")
    
    def full_pipeline(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Execute full training pipeline"""
        logger.info("Starting full training pipeline...")
        start_time = time.time()
        
        # Feature extraction
        train_features = self.advanced_feature_extraction(train_data)
        
        # Prepare target variable
        if 'label' in train_features.columns:
            # Convert labels to binary
            label_map = {'true': 0, 'mostly-true': 0, 'half-true': 0, 
                        'barely-true': 1, 'false': 1, 'pants-fire': 1}
            y = train_features['label'].map(label_map)
            X = train_features.drop('label', axis=1)
        else:
            raise ValueError("No 'label' column found in training data")
        
        # Split if no test data provided
        if test_data is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['validation']['test_size'],
                random_state=self.config['validation']['random_state'],
                stratify=y
            )
        else:
            X_train, y_train = X, y
            test_features = self.advanced_feature_extraction(test_data)
            y_test = test_features['label'].map(label_map)
            X_test = test_features.drop('label', axis=1)
        
        # Create preprocessor and fit
        preprocessor = self.create_preprocessing_pipeline(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train individual models
        individual_models = self.train_individual_models(X_train_processed, y_train)
        
        # Create ensemble
        ensemble_model = self.create_ensemble_model(individual_models)
        ensemble_model.fit(X_train_processed, y_train)
        
        # Add ensemble to models dict
        all_models = individual_models.copy()
        all_models['ensemble'] = ensemble_model
        
        # Evaluate all models
        performance = self.evaluate_models(all_models, X_test_processed, y_test)
        
        # Save models
        self.save_models(all_models, preprocessor)
        
        # Store results
        self.models = all_models
        self.preprocessors['main'] = preprocessor
        self.performance_metrics = performance
        self.ensemble_model = ensemble_model
        
        total_time = time.time() - start_time
        logger.info(f"Full pipeline completed in {total_time:.2f} seconds")
        
        return {
            'models': all_models,
            'preprocessor': preprocessor,
            'performance': performance,
            'training_time': total_time,
            'feature_count': X_train_processed.shape[1]
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = AdvancedFakeNewsProcessor()
    
    # Load sample data (replace with actual LIAR dataset)
    try:
        train_data = pd.read_csv("../../data/processed/train_processed.csv")
        test_data = pd.read_csv("../../data/processed/test_processed.csv")
        
        # Run full pipeline
        results = processor.full_pipeline(train_data, test_data)
        
        print("\n=== Training Results ===")
        for model_name, metrics in results['performance'].items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")
        
        print(f"\nTotal features: {results['feature_count']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        
    except FileNotFoundError:
        logger.warning("Processed data files not found. Please run data preprocessing first.")
        
        # Create sample data for testing
        logger.info("Creating sample data for testing...")
        sample_data = pd.DataFrame({
            'statement': [
                'This is a true news statement',
                'This is completely fake and false information',
                'Politicians are making important decisions',
                'The economy is performing well according to experts'
            ] * 100,  # Repeat for larger dataset
            'label': ['true', 'false', 'half-true', 'mostly-true'] * 100,
            'speaker': ['John Doe'] * 400,
            'job': ['politician'] * 400,
            'context': ['debate'] * 400
        })
        
        # Run pipeline with sample data
        results = processor.full_pipeline(sample_data)
        
        print("\n=== Sample Data Training Results ===")
        for model_name, metrics in results['performance'].items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")