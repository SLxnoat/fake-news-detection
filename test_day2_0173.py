import sys
import os
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
from models.hybrid_model_0173 import HybridFakeNewsDetector, test_hybrid_model
from models.bert_extractor_0173 import test_feature_fusion

def quick_test_day2():
    """Quick test of Day 2 implementation"""
    print("Day 2 Quick Test - Member 0173")
    print("=" * 50)
    
    # Test 1: Hybrid Model Architecture
    print("\n1️ Testing Hybrid Model Architecture...")
    try:
        model = test_hybrid_model()
        print("Hybrid model architecture working!")
    except Exception as e:
        print(f" Hybrid model test failed: {e}")
        return False
    
    # Test 2: Feature Fusion Pipeline  
    print("\n2️ Testing Feature Fusion Pipeline...")
    try:
        pipeline, train_dataset, valid_dataset,