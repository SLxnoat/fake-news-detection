import pickle
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployment:
    def __init__(self, models_dir='../models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessors = {}
        self.metadata = {}
        
    def load_model(self, model_path: str, model_name: str, model_type: str = 'sklearn'):
        """Load a single model"""
        try:
            if model_type == 'sklearn':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_type == 'pytorch':
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'eval'):
                    model.eval()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.models[model_name] = {
                'model': model,
                'type': model_type,
                'loaded_at': datetime.now(),
                'path': model_path
            }
            
            logger.info(f"Successfully loaded {model_name} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all available models"""
        model_configs = [
            {'path': os.path.join(self.models_dir, 'baseline/tfidf_logistic.pkl'),
             'name': 'tfidf_logistic', 'type': 'sklearn'},
            {'path': os.path.join(self.models_dir, 'baseline/tfidf_rf.pkl'),
             'name': 'tfidf_rf', 'type': 'sklearn'},
            {'path': os.path.join(self.models_dir, 'hybrid/best_hybrid_model.pth'),
             'name': 'hybrid_model', 'type': 'pytorch'}
        ]
        
        loaded_models = 0
        for config in model_configs:
            if os.path.exists(config['path']):
                if self.load_model(config['path'], config['name'], config['type']):
                    loaded_models += 1
            else:
                logger.warning(f"Model file not found: {config['path']}")
        
        logger.info(f"Loaded {loaded_models} models successfully")
        return loaded_models
    
    def load_preprocessors(self):
        """Load preprocessing components"""
        preprocessor_configs = [
            {'path': os.path.join(self.models_dir, 'preprocessors/text_preprocessor.pkl'),
             'name': 'text_preprocessor'},
            {'path': os.path.join(self.models_dir, 'preprocessors/tfidf_vectorizer.pkl'),
             'name': 'tfidf_vectorizer'},
            {'path': os.path.join(self.models_dir, 'preprocessors/metadata_processor.pkl'),
             'name': 'metadata_processor'}
        ]
        
        loaded_preprocessors = 0
        for config in preprocessor_configs:
            if os.path.exists(config['path']):
                try:
                    with open(config['path'], 'rb') as f:
                        preprocessor = pickle.load(f)
                    self.preprocessors[config['name']] = preprocessor
                    loaded_preprocessors += 1
                    logger.info(f"Loaded preprocessor: {config['name']}")
                except Exception as e:
                    logger.error(f"Failed to load preprocessor {config['name']}: {str(e)}")
            else:
                logger.warning(f"Preprocessor not found: {config['path']}")
        
        return loaded_preprocessors
    
    def predict_single(self, model_name: str, statement: str, 
                      speaker: str = "", party: str = "", subject: str = "") -> Dict[str, Any]:
        """Make prediction with a single model"""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not loaded'}
        
        try:
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Basic preprocessing (simplified)
            processed_statement = self.preprocess_text(statement)
            
            # Make prediction based on model type
            if model_info['type'] == 'sklearn':
                # For sklearn models, assume they expect text input
                prediction = model.predict([processed_statement])[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba([processed_statement])[0]
                    confidence = float(max(probabilities))
                else:
                    confidence = 0.5  # Default confidence
                    
            elif model_info['type'] == 'pytorch':
                # For PyTorch models, more complex preprocessing needed
                prediction = 'half-true'  # Placeholder
                confidence = 0.75  # Placeholder
            
            result = {
                'model': model_name,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'input': {
                    'statement': statement,
                    'speaker': speaker,
                    'party': party,
                    'subject': subject
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def predict_ensemble(self, statement: str, speaker: str = "", 
                        party: str = "", subject: str = "") -> Dict[str, Any]:
        """Make predictions with all loaded models and ensemble them"""
        if not self.models:
            return {'error': 'No models loaded'}
        
        predictions = {}
        all_predictions = []
        confidences = []
        
        # Get predictions from all models
        for model_name in self.models.keys():
            result = self.predict_single(model_name, statement, speaker, party, subject)
            if 'error' not in result:
                predictions[model_name] = result
                all_predictions.append(result['prediction'])
                confidences.append(result['confidence'])
        
        if not all_predictions:
            return {'error': 'All models failed to make predictions'}
        
        # Simple ensemble: majority voting
        from collections import Counter
        prediction_counts = Counter(all_predictions)
        ensemble_prediction = prediction_counts.most_common(1)[0][0]
        ensemble_confidence = np.mean(confidences)
        
        result = {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': float(ensemble_confidence),
            'individual_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = ''.join(char for char in text if char.isalnum() or char.isspace())
        text = ' '.join(text.split())  # Remove extra whitespaces
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': len(self.models),
            'loaded_preprocessors': len(self.preprocessors),
            'models': {}
        }
        
        for model_name, model_info in self.models.items():
            info['models'][model_name] = {
                'type': model_info['type'],
                'loaded_at': model_info['loaded_at'].isoformat(),
                'path': model_info['path']
            }
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all loaded models"""
        health_status = {
            'status': 'healthy',
            'models_status': {},
            'timestamp': datetime.now().isoformat()
        }
        
        test_statement = "This is a test statement for health check."
        
        for model_name in self.models.keys():
            try:
                result = self.predict_single(model_name, test_statement)
                if 'error' in result:
                    health_status['models_status'][model_name] = 'unhealthy'
                    health_status['status'] = 'degraded'
                else:
                    health_status['models_status'][model_name] = 'healthy'
            except Exception as e:
                health_status['models_status'][model_name] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
        
        return health_status
 