#!/usr/bin/env python3
"""
Script to create the missing processed_liar_dataset.csv file
This fixes the FileNotFoundError in multiple notebooks
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from preprocessing.text_preprocessor_0148 import TextPreprocessor
    from preprocessing.metadata_processor_0148 import MetadataProcessor
    print("✅ Successfully imported preprocessing modules")
except ImportError as e:
    print(f"⚠️ Warning: Could not import preprocessing modules: {e}")
    print("Continuing with basic processing...")

def load_and_process_data():
    """Load raw data and create processed dataset"""
    
    # Check if raw data exists
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Load training data (largest dataset)
    train_file = data_dir / 'train.tsv'
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        return False
    
    print(f"📊 Loading data from: {train_file}")
    
    try:
        # Load the TSV file
        df = pd.read_csv(train_file, sep='\t', header=None)
        
        # Define column names based on LIAR dataset structure
        columns = [
            'id', 'label', 'statement', 'subject', 'speaker', 'speaker_job',
            'state_info', 'party_affiliation', 'barely_true_counts',
            'false_counts', 'half_true_counts', 'mostly_true_counts',
            'pants_fire_counts', 'context'
        ]
        
        # Rename columns
        if len(df.columns) >= len(columns):
            df.columns = columns + [f'extra_{i}' for i in range(len(df.columns) - len(columns))]
        else:
            print(f"⚠️ Warning: Expected {len(columns)} columns, got {len(df.columns)}")
            # Pad with extra columns if needed
            while len(df.columns) < len(columns):
                df[f'extra_{len(df.columns)}'] = np.nan
            df.columns = columns
        
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Basic data cleaning
        print("🧹 Cleaning data...")
        
        # Handle missing values
        df['statement'] = df['statement'].fillna('')
        df['speaker'] = df['speaker'].fillna('unknown')
        df['party_affiliation'] = df['party_affiliation'].fillna('unknown')
        df['subject'] = df['subject'].fillna('general')
        
        # Convert credibility counts to numeric, handling any non-numeric values
        credibility_cols = ['barely_true_counts', 'false_counts', 'half_true_counts',
                          'mostly_true_counts', 'pants_fire_counts']
        
        for col in credibility_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
        
        # Create derived features
        print("🔧 Creating derived features...")
        
        # Text features
        df['text_length'] = df['statement'].str.len()
        df['word_count'] = df['statement'].str.split().str.len()
        
        # Credibility features
        df['total_statements'] = df[credibility_cols].sum(axis=1)
        df['credibility_score'] = (
            (df['mostly_true_counts'] * 1.0 + 
             df['half_true_counts'] * 0.5 + 
             df['barely_true_counts'] * 0.25) / 
            (df['total_statements'] + 1e-5)  # Avoid division by zero
        )
        
        # Deception score
        df['deception_score'] = (
            (df['false_counts'] * 1.0 + 
             df['pants_fire_counts'] * 1.5) / 
            (df['total_statements'] + 1e-5)
        )
        
        # Speaker credibility history
        df['speaker_credibility'] = df.groupby('speaker')['credibility_score'].transform('mean')
        df['speaker_deception'] = df.groupby('speaker')['deception_score'].transform('mean')
        
        # Fill NaN values for new speakers
        df['speaker_credibility'] = df['speaker_credibility'].fillna(0.5)
        df['speaker_deception'] = df['speaker_deception'].fillna(0.5)
        
        # Party-level features
        df['party_avg_credibility'] = df.groupby('party_affiliation')['credibility_score'].transform('mean')
        df['party_avg_deception'] = df.groupby('party_affiliation')['deception_score'].transform('mean')
        
        # Subject complexity
        df['subject_complexity'] = df.groupby('subject')['text_length'].transform('mean')
        df['subject_complexity'] = df['subject_complexity'].fillna(df['text_length'].mean())
        
        # Create results directory
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Save processed dataset
        output_file = results_dir / 'processed_liar_dataset.csv'
        df.to_csv(output_file, index=False)
        
        print(f"✅ Processed dataset saved to: {output_file}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🔍 Sample columns: {list(df.columns[:10])}...")
        
        # Save summary statistics
        summary_file = results_dir / 'dataset_summary.json'
        summary = {
            'total_statements': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'party_distribution': df['party_affiliation'].value_counts().to_dict(),
            'subject_distribution': df['subject'].value_counts().head(10).to_dict(),
            'text_length_stats': {
                'mean': float(df['text_length'].mean()),
                'std': float(df['text_length'].std()),
                'min': int(df['text_length'].min()),
                'max': int(df['text_length'].max())
            },
            'credibility_score_stats': {
                'mean': float(df['credibility_score'].mean()),
                'std': float(df['credibility_score'].std()),
                'min': float(df['credibility_score'].min()),
                'max': float(df['credibility_score'].max())
            }
        }
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Summary statistics saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Creating Processed Dataset...")
    print("=" * 50)
    
    success = load_and_process_data()
    
    if success:
        print("\n✅ Dataset processing completed successfully!")
        print("\n📁 Files created:")
        print("- results/processed_liar_dataset.csv")
        print("- results/dataset_summary.json")
        print("\n🎯 You can now run the notebooks that depend on this dataset.")
    else:
        print("\n❌ Dataset processing failed!")
        print("Please check the error messages above and ensure:")
        print("1. Raw data files exist in data/raw/")
        print("2. Required packages are installed")
        print("3. File permissions are correct")

if __name__ == "__main__":
    main()
