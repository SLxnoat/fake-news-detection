#!/usr/bin/env python3
"""
Quick Start Script for Fake News Detection Project
Member: ITBIN-2211-0184 (EDA & Documentation)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_environment():
    """Check if all required packages are installed"""
    print("🔍 Checking Environment Setup...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'sklearn', 'nltk', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n🎉 All required packages are installed!")
        return True

def check_directory_structure():
    """Check if all required directories exist"""
    print("\n📁 Checking Directory Structure...")
    
    required_dirs = [
        'data/raw', 'data/processed', 'notebooks', 'src',
        'models', 'results/plots', 'results/reports', 
        'app/backend', 'app/frontend'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            missing_dirs.append(directory)
            print(f"❌ {directory} - MISSING")
            try:
                os.makedirs(directory)
                print(f"   🔧 Created: {directory}")
            except Exception as e:
                print(f"   ❌ Failed to create {directory}: {e}")
    
    if not missing_dirs:
        print("\n🎉 All directories are properly set up!")
    return len(missing_dirs) == 0

def check_dataset():
    """Check if dataset files are available"""
    print("\n📊 Checking Dataset Files...")
    
    dataset_files = [
        'data/raw/train.tsv',
        'data/raw/test.tsv', 
        'data/raw/valid.tsv'
    ]
    
    missing_files = []
    total_samples = 0
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            try:
                # Try to read the file to check if it's valid
                df = pd.read_csv(file_path, sep='\t', nrows=5)  # Just read first 5 rows
                file_size = os.path.getsize(file_path)
                print(f"✅ {file_path} ({file_size/1024:.1f} KB)")
            except Exception as e:
                print(f"⚠️ {file_path} - EXISTS but may be corrupted: {e}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - MISSING")
    
    if missing_files:
        print(f"\n📥 Missing dataset files. Please download from:")
        print("   https://github.com/thiagorainmaker77/liar_dataset")
        print("   Files needed:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("\n🎉 All dataset files are available!")
        return True

def run_basic_eda_test():
    """Run a basic EDA test to verify everything works"""
    print("\n🔬 Running Basic EDA Test...")
    
    try:
        # Try to load and analyze a small sample
        if os.path.exists('data/raw/train.tsv'):
            df = pd.read_csv('data/raw/train.tsv', sep='\t', nrows=100, header=None)
            
            print(f"✅ Successfully loaded {len(df)} sample rows")
            print(f"✅ Dataset shape: {df.shape}")
            print(f"✅ Columns: {df.shape[1]}")
            
            # Try basic visualization
            plt.figure(figsize=(8, 4))
            df[0].value_counts().plot(kind='bar', title='Sample Label Distribution')
            plt.tight_layout()
            
            # Save test plot
            os.makedirs('results/plots', exist_ok=True)
            plt.savefig('results/plots/test_visualization.png')
            plt.close()
            
            print("✅ Basic visualization created and saved")
            print("✅ EDA pipeline is working correctly!")
            
            return True
            
    except Exception as e:
        print(f"❌ EDA test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    print("\n⚙️ Creating Sample Configuration...")
    
    config_content = """# Fake News Detection Project Configuration
# Member: ITBIN-2211-0184

# Data Configuration
DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
RESULTS_PATH = "results/"

# EDA Configuration
FIGURE_SIZE = (20, 24)
DPI = 300
COLOR_PALETTE = "husl"

# Dataset Configuration
DATASET_FILES = {
    "train": "data/raw/train.tsv",
    "test": "data/raw/test.tsv", 
    "valid": "data/raw/valid.tsv"
}

# Column Names for LIAR Dataset
COLUMNS = [
    'label', 'statement', 'subject', 'speaker', 'speaker_job',
    'state_info', 'party_affiliation', 'barely_true_counts',
    'false_counts', 'half_true_counts', 'mostly_true_counts', 
    'pants_fire_counts', 'context'
]

# Truth Label Mapping
TRUTH_MAPPING = {
    'true': 1.0,
    'mostly-true': 0.8,
    'half-true': 0.5,
    'barely-true': 0.2,
    'false': 0.0,
    'pants-fire': 0.0
}"""
    
    try:
        with open('config.py', 'w') as f:
            f.write(config_content)
        print("✅ Configuration file created: config.py")
        return True
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False

def main():
    """Main function to run all checks"""
    print("=" * 60)
    print("🚀 FAKE NEWS DETECTION PROJECT - QUICK START")
    print("   Member: ITBIN-2211-0184 (EDA & Documentation)")
    print("=" * 60)
    
    # Run all checks
    env_ok = check_environment()
    dirs_ok = check_directory_structure()
    data_ok = check_dataset()
    
    # Create configuration
    config_ok = create_sample_config()
    
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY:")
    print(f"   Environment: {'Ready' if env_ok else 'Issues'}")
    print(f"   Directories: {'Ready' if dirs_ok else 'Issues'}")
    print(f"   Dataset: {'Ready' if data_ok else 'Missing files'}")
    print(f"   Configuration: {'Created' if config_ok else 'Issues'}")
    
    if env_ok and dirs_ok and data_ok:
        print("\n🎉 EVERYTHING IS READY!")
        print("\nNext Steps:")
        print("1. Open Jupyter Notebook: jupyter notebook")
        print("2. Create: notebooks/01_EDA_and_Data_Understanding.ipynb") 
        print("3. Copy the EDA code provided earlier")
        print("4. Run the analysis!")
        
        # Run basic test if dataset is available
        if data_ok:
            test_ok = run_basic_eda_test()
            if test_ok:
                print("\n✅ Basic EDA test passed - you're ready to go!")
    else:
        print("\n⚠️ SETUP INCOMPLETE")
        print("Please fix the issues above before proceeding.")
        
        if not data_ok:
            print("\n📥 TO DOWNLOAD DATASET:")
            print("1. Go to: https://github.com/thiagorainmaker77/liar_dataset")
            print("2. Download: train.tsv, test.tsv, valid.tsv")
            print("3. Place in: data/raw/")
    
    print("=" * 60)

if __name__ == "__main__":
    main()