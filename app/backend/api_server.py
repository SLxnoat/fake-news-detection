from flask import Flask, request, jsonify, send_file
try:
    from flask_cors import CORS
except ImportError:
    # Fallback no-op CORS to allow server startup without flask-cors installed
    def CORS(app=None, *args, **kwargs):
        return app
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
import logging
import time
from functools import wraps
import hashlib
try:
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    class _DummyJWT:
        """Insecure JWT stub for development when PyJWT is missing."""
        @staticmethod
        def encode(payload, key, algorithm='HS256'):
            return f"dummy.{payload.get('username', 'user')}"

        @staticmethod
        def decode(token, key, algorithms=None):
            if isinstance(token, bytes):
                token = token.decode('utf-8', errors='ignore')
            if isinstance(token, str) and token.startswith('dummy.'):
                return {'username': token.split('.', 1)[1]}
            raise Exception('Invalid token')

    pyjwt = _DummyJWT()
from werkzeug.security import generate_password_hash, check_password_hash

# Add src to path (robust to current working directory)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from preprocessing.text_preprocessor_0148 import TextPreprocessor
from preprocessing.metadata_processor_0148 import MetadataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production

if not JWT_AVAILABLE:
    logger.warning("PyJWT is not installed. Using an insecure dummy token for development.")

class FakeNewsAPI:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.metadata_processor = MetadataProcessor()
        self.models = {}
        self.prediction_cache = {}
        self.api_usage_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0
        }
        
        # Load models
        self.load_models()
        
        # Demo users (replace with proper database in production)
        self.users = {
            'admin': generate_password_hash('password123'),
            'api_user': generate_password_hash('apikey123')
        }

    def load_models(self):
        """Load available models"""
        model_dir = "../../models"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('.pkl', '')
                    try:
                        with open(os.path.join(model_dir, file), 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {str(e)}")

    def generate_cache_key(self, statement: str, metadata: dict = None) -> str:
        """Generate cache key for predictions"""
        content = statement + str(sorted((metadata or {}).items()))
        return hashlib.md5(content.encode()).hexdigest()

    def authenticate_token(self, token: str) -> dict:
        """Authenticate JWT token"""
        try:
            payload = pyjwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            return payload
        except Exception:
            return None

api = FakeNewsAPI()

# Decorators
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            payload = api.authenticate_token(token)
            if not payload:
                return jsonify({'error': 'Token is invalid'}), 401
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    return decorated

def log_request(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api.api_usage_stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            result = f(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Log successful request
            logger.info(f"API call: {request.endpoint}, Time: {processing_time:.3f}s")
            
            return result
        except Exception as e:
            logger.error(f"API error in {request.endpoint}: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    
    return decorated

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(api.models),
        'version': '1.0.0'
    })

@app.route('/api/login', methods=['POST'])
def login():
    """User authentication"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if username in api.users and check_password_hash(api.users[username], password):
        token = pyjwt.encode({
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'token': token,
            'username': username,
            'expires_in': 24 * 3600
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/predict', methods=['POST'])
@require_auth
@log_request
def predict():
    """Single prediction endpoint"""
    try:
        data = request.get_json()
        start_time = time.time()
        statement = data.get('statement', '')
        metadata = data.get('metadata', {})
        use_cache = data.get('use_cache', True)
        
        if not statement.strip():
            return jsonify({'error': 'Statement cannot be empty'}), 400
        
        # Check cache
        cache_key = api.generate_cache_key(statement, metadata)
        if use_cache and cache_key in api.prediction_cache:
            api.api_usage_stats['cache_hits'] += 1
            cached_result = api.prediction_cache[cache_key]
            cached_result['from_cache'] = True
            return jsonify(cached_result)
        
        # Process text (basic preprocessing)
        try:
            processed_text = api.text_processor.process_single_text(statement)
        except Exception:
            processed_text = statement
        
        # Make prediction (simulate with multiple models)
        predictions = {}
        confidences = {}
        
        for model_name in ['logistic_regression', 'random_forest', 'neural_network']:
            # Simulate model predictions
            pred = np.random.choice(['real', 'fake'])
            conf = np.random.uniform(0.6, 0.95)
            predictions[model_name] = pred
            confidences[model_name] = conf
        
        # Ensemble prediction
        fake_votes = sum(1 for p in predictions.values() if p == 'fake')
        final_prediction = 'fake' if fake_votes > len(predictions) / 2 else 'real'
        final_confidence = np.mean(list(confidences.values()))
        
        # Prepare response
        result = {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'individual_models': {
                model: {'prediction': pred, 'confidence': confidences[model]}
                for model, pred in predictions.items()
            },
            'text_features': {
                'word_count': len(processed_text.split()),
                'char_count': len(processed_text),
                'sentence_count': len(processed_text.split('.')),
                'avg_word_length': float(np.mean([len(word) for word in processed_text.split()]) or 0)
            },
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'from_cache': False
        }
        
        # Cache result
        if use_cache:
            api.prediction_cache[cache_key] = result.copy()
        
        api.api_usage_stats['successful_predictions'] += 1
        return jsonify(result)
        
    except Exception as e:
        api.api_usage_stats['failed_predictions'] += 1
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Failed to process prediction'}), 500

@app.route('/api/predict/batch', methods=['POST'])
@require_auth
@log_request
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        statements = data.get('statements', [])
        metadata_list = data.get('metadata', [])
        
        if not statements:
            return jsonify({'error': 'No statements provided'}), 400
        
        if len(statements) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 100)'}), 400
        
        results = []
        processing_start = time.time()
        
        for i, statement in enumerate(statements):
            metadata = metadata_list[i] if i < len(metadata_list) else {}
            
            # Individual prediction (reuse logic from single predict)
            pred = np.random.choice(['real', 'fake'])
            conf = np.random.uniform(0.6, 0.95)
            
            result = {
                'id': i,
                'statement': statement,
                'prediction': pred,
                'confidence': conf,
                'processing_time': np.random.uniform(0.1, 0.5)
            }
            results.append(result)
        
        total_time = time.time() - processing_start
        
        # Summary statistics
        fake_count = sum(1 for r in results if r['prediction'] == 'fake')
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        response = {
            'results': results,
            'summary': {
                'total_processed': len(results),
                'fake_predictions': fake_count,
                'real_predictions': len(results) - fake_count,
                'average_confidence': avg_confidence,
                'total_processing_time': total_time,
                'avg_time_per_item': total_time / len(results)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        api.api_usage_stats['successful_predictions'] += len(results)
        return jsonify(response)
        
    except Exception as e:
        api.api_usage_stats['failed_predictions'] += 1
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Failed to process batch prediction'}), 500

@app.route('/api/models', methods=['GET'])
@require_auth
def get_models():
    """Get available models information"""
    model_info = {}
    for model_name in api.models.keys():
        # Simulate model metadata
        model_info[model_name] = {
            'name': model_name,
            'type': 'classifier',
            'accuracy': np.random.uniform(0.85, 0.95),
            'precision': np.random.uniform(0.80, 0.93),
            'recall': np.random.uniform(0.82, 0.94),
            'f1_score': np.random.uniform(0.81, 0.93),
            'training_date': '2024-01-15',
            'version': '1.0'
        }
    
    return jsonify({
        'models': model_info,
        'default_model': 'ensemble',
        'total_models': len(model_info)
    })

@app.route('/api/stats', methods=['GET'])
@require_auth
def get_stats():
    """API usage statistics"""
    return jsonify({
        'usage_stats': api.api_usage_stats,
        'cache_size': len(api.prediction_cache),
        'uptime': 'N/A',  # Would calculate actual uptime
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/feedback', methods=['POST'])
@require_auth
@log_request
def submit_feedback():
    """Submit prediction feedback"""
    try:
        data = request.get_json()
        statement = data.get('statement')
        prediction = data.get('prediction')
        actual_label = data.get('actual_label')
        confidence = data.get('confidence')
        feedback_score = data.get('feedback_score')  # 1-5 rating
        comments = data.get('comments', '')
        
        # Store feedback (in production, save to database)
        feedback_entry = {
            'statement': statement,
            'prediction': prediction,
            'actual_label': actual_label,
            'confidence': confidence,
            'feedback_score': feedback_score,
            'comments': comments,
            'timestamp': datetime.now().isoformat(),
            'user': 'api_user'  # Would get from JWT token
        }
        
        # Log feedback
        logger.info(f"Feedback received: {feedback_entry}")
        
        return jsonify({
            'message': 'Feedback submitted successfully',
            'feedback_id': hashlib.md5(str(feedback_entry).encode()).hexdigest()[:8]
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({'error': 'Failed to submit feedback'}), 500

@app.route('/api/export/predictions', methods=['GET'])
@require_auth
def export_predictions():
    """Export prediction history"""
    try:
        # In production, get from database
        # For demo, create sample data
        sample_data = []
        for i in range(50):
            sample_data.append({
                'id': i,
                'statement': f'Sample news statement {i}',
                'prediction': np.random.choice(['real', 'fake']),
                'confidence': np.random.uniform(0.6, 0.95),
                'timestamp': datetime.now().isoformat(),
                'processing_time': np.random.uniform(0.1, 0.5)
            })
        
        df = pd.DataFrame(sample_data)
        
        # Save to CSV
        filename = f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': 'Failed to export data'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)