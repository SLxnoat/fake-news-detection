import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time
import logging
from .model_deployment import ModelDeployment

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, models_dir='../models'):
        self.deployment = ModelDeployment(models_dir)
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_response_time': 0.0,
            'response_times': []
        }
    
    def initialize(self):
        """Initialize the inference pipeline"""
        logger.info("Initializing inference pipeline...")
        
        # Load models and preprocessors
        models_loaded = self.deployment.load_all_models()
        preprocessors_loaded = self.deployment.load_preprocessors()
        
        logger.info(f"Loaded {models_loaded} models and {preprocessors_loaded} preprocessors")
        
        # Perform health check
        health = self.deployment.health_check()
        logger.info(f"Health check status: {health['status']}")
        
        return {
            'models_loaded': models_loaded,
            'preprocessors_loaded': preprocessors_loaded,
            'health_status': health['status']
        }
    
    def predict(self, statement: str, speaker: str = "", party: str = "", 
               subject: str = "", use_ensemble: bool = True) -> Dict[str, Any]:
        """Make prediction with performance tracking"""
        start_time = time.time()
        
        try:
            if use_ensemble:
                result = self.deployment.predict_ensemble(statement, speaker, party, subject)
            else:
                # Use the first available model
                if not self.deployment.models:
                    result = {'error': 'No models available'}
                else:
                    model_name = list(self.deployment.models.keys())[0]
                    result = self.deployment.predict_single(model_name, statement, speaker, party, subject)
            
            response_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['response_times'].append(response_time)
            
            if 'error' not in result:
                self.performance_stats['successful_predictions'] += 1
            else:
                self.performance_stats['failed_predictions'] += 1
            
            # Update average response time
            self.performance_stats['average_response_time'] = np.mean(
                self.performance_stats['response_times'][-100:]  # Last 100 predictions
            )
            
            # Add performance info to result
            result['performance'] = {
                'response_time': response_time,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['failed_predictions'] += 1
            self.performance_stats['response_times'].append(response_time)
            
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'error': str(e),
                'performance': {
                    'response_time': response_time,
                    'timestamp': time.time()
                }
            }
    
    def batch_predict(self, statements: List[Dict[str, str]], 
                     use_ensemble: bool = True) -> List[Dict[str, Any]]:
        """Make predictions for multiple statements"""
        results = []
        
        logger.info(f"Processing batch of {len(statements)} statements")
        
        for i, item in enumerate(statements):
            statement = item.get('statement', '')
            speaker = item.get('speaker', '')
            party = item.get('party', '')
            subject = item.get('subject', '')
            
            result = self.predict(statement, speaker, party, subject, use_ensemble)
            result['batch_index'] = i
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(statements)} statements")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_predictions']
            stats['failure_rate'] = stats['failed_predictions'] / stats['total_predictions']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Calculate percentiles for response times
        if stats['response_times']:
            stats['response_time_p50'] = np.percentile(stats['response_times'], 50)
            stats['response_time_p95'] = np.percentile(stats['response_times'], 95)
            stats['response_time_p99'] = np.percentile(stats['response_times'], 99)
        
        # Remove raw response times from output (too much data)
        stats_output = {k: v for k, v in stats.items() if k != 'response_times'}
        stats_output['response_times_count'] = len(stats['response_times'])
        
        return stats_output
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_response_time': 0.0,
            'response_times': []
        }
        logger.info("Performance statistics reset")