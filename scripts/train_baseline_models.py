#!/usr/bin/env python3
"""
Script to train and save baseline models for the unified app
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path (robust to current working directory)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from models.baseline_models_0149 import BaselineModels
    print("✅ Successfully imported BaselineModels")
except ImportError as e:
    print(f"❌ Error importing BaselineModels: {e}")
    sys.exit(1)

def train_and_save_models():
    """Train baseline models and save them"""
    
    print("🚀 Training Baseline Models...")
    
    # Check if data files exist
    data_files = {
        'train': 'data/processed/train_processed.csv',
        'test': 'data/processed/test_processed.csv',
        'valid': 'data/processed/valid_processed.csv'
    }
    
    # Check for raw data if processed data doesn't exist
    raw_data_files = {
        'train': 'data/raw/train.tsv',
        'test': 'data/raw/test.tsv',
        'valid': 'data/raw/valid.tsv'
    }
    
    # Try to load data
    train_df = None
    test_df = None
    valid_df = None
    
    # First try processed data
    for split, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                if split == 'train':
                    train_df = pd.read_csv(filepath)
                elif split == 'test':
                    test_df = pd.read_csv(filepath)
                elif split == 'valid':
                    valid_df = pd.read_csv(filepath)
                print(f"✅ Loaded {split} data from {filepath}")
            except Exception as e:
                print(f"❌ Error loading {split} data: {e}")
    
    # If processed data not available, try raw data
    if train_df is None or test_df is None or valid_df is None:
        print("⚠️ Processed data not found, trying raw data...")
        
        for split, filepath in raw_data_files.items():
            if os.path.exists(filepath):
                try:
                    if split == 'train':
                        train_df = pd.read_csv(filepath, sep='\t')
                    elif split == 'test':
                        test_df = pd.read_csv(filepath, sep='\t')
                    elif split == 'valid':
                        valid_df = pd.read_csv(filepath, sep='\t')
                    print(f"✅ Loaded {split} raw data from {filepath}")
                except Exception as e:
                    print(f"❌ Error loading {split} raw data: {e}")
    
    # Check if we have the required data
    if train_df is None or test_df is None or valid_df is None:
        print("❌ Could not load required data files")
        print("Please ensure the following files exist:")
        for filepath in data_files.values():
            print(f"  - {filepath}")
        print("Or the raw data files:")
        for filepath in raw_data_files.values():
            print(f"  - {filepath}")
        return False
    
    # Check if we have the required columns
    required_columns = ['statement', 'label']
    for df_name, df in [('train', train_df), ('test', test_df), ('valid', valid_df)]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ {df_name} data missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return False
    
    print(f"📊 Data loaded successfully:")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    print(f"  - Valid: {len(valid_df)} samples")
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    try:
        # Prepare data
        print("🔧 Preparing data...")
        X_train, X_test, y_train, y_test = baseline.prepare_data(train_df, test_df, valid_df)
        
        # Train models
        print("🤖 Training models...")
        models = baseline.train_all_models(X_train, y_train)
        
        # Evaluate models
        print("📊 Evaluating models...")
        results = baseline.evaluate_models(X_test, y_test)
        
        # Save models
        print("💾 Saving models...")
        baseline.save_models()
        
        # Create performance report
        print("📈 Creating performance report...")
        report_df = baseline.create_performance_report()
        
        print("✅ Baseline models training completed successfully!")
        print(f"📁 Models saved to: models/baseline/")
        print(f"📊 Performance report saved to: results/reports/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dummy_models():
    """Create dummy models for testing if training fails"""
    
    print("🔄 Creating dummy models for testing...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import pickle
        
        # Create a simple dummy model
        dummy_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Fit on dummy data
        dummy_texts = [
            "This is a true statement",
            "This is a false statement",
            "This is a half true statement"
        ]
        dummy_labels = [5, 1, 3]  # true, false, half-true
        
        dummy_pipeline.fit(dummy_texts, dummy_labels)
        
        # Save dummy model
        os.makedirs('models/baseline', exist_ok=True)
        
        dummy_model_path = 'models/baseline/tfidf_logistic_0149.pkl'
        with open(dummy_model_path, 'wb') as f:
            pickle.dump(dummy_pipeline, f)
        
        print(f"✅ Dummy model created and saved to: {dummy_model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating dummy models: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Baseline Models Training Script")
    print("=" * 50)
    
    # Try to train real models first
    success = train_and_save_models()
    
    # If training fails, create dummy models
    if not success:
        print("\n⚠️ Training failed, creating dummy models for testing...")
        create_dummy_models()
    
    print("\n🎯 Script completed!")
    print("You can now run the unified app with: python launcher.py")
