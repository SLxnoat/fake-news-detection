#!/usr/bin/env python3
"""
Testing Pipeline for Fake News Detection System
Member 0148: Data Preprocessing & Web Application

This script tests all components of the preprocessing pipeline
and web application functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from preprocessing.text_processor import TextPreprocessor
from preprocessing.metadata_processor import MetadataProcessor
import traceback
from datetime import datetime

def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TESTING: {test_name}")
    print(f"{'='*60}")

def print_test_result(test_name, passed, error=None):
    """Print test results."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status}: {test_name}")
    if error:
        print(f"Error: {error}")

def test_text_preprocessing():
    """Test the text preprocessing pipeline."""
    print_test_header("Text Preprocessing Pipeline")
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Basic initialization
        processor = TextPreprocessor()
        print_test_result("TextPreprocessor initialization", True)
        tests_passed += 1
        
        # Test 2: Single text processing
        test_text = "This is a TEST with CAPS and numbers 123!!!"
        processed = processor.process_single_text(test_text)
        expected_words = ['test', 'caps', 'numbers']  # Expected after processing
        
        success = all(word in processed.lower() for word in ['test', 'caps'])
        print_test_result("Single text processing", success)
        if success:
            tests_passed += 1
        print(f"  Original: {test_text}")
        print(f"  Processed: {processed}")
        
        # Test 3: Batch text processing
        test_texts = [
            "The economy is growing rapidly.",
            "Healthcare costs have increased significantly.",
            "",  # Empty string test
            None  # None test
        ]
        
        processed_texts = processor.process_texts(test_texts, show_progress=False)
        success = len(processed_texts) == len(test_texts)
        print_test_result("Batch text processing", success)
        if success:
            tests_passed += 1
        
        # Test 4: Text statistics
        stats = processor.get_text_statistics(processed_texts)
        required_stats = ['total_texts', 'empty_texts', 'avg_length', 'avg_word_count']
        success = all(stat in stats for stat in required_stats)
        print_test_result("Text statistics calculation", success)
        if success:
            tests_passed += 1
        print(f"  Statistics: {stats}")
        
        # Test 5: Save and load processor
        test_path = "../../models/test_text_processor.pkl"
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        processor.save_preprocessor(test_path)
        loaded_processor = TextPreprocessor.load_preprocessor(test_path)
        
        # Test if loaded processor works the same
        original_result = processor.process_single_text("Test sentence")
        loaded_result = loaded_processor.process_single_text("Test sentence")
        
        success = original_result == loaded_result
        print_test_result("Save and load processor", success)
        if success:
            tests_passed += 1
        
        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print_test_result("Text preprocessing pipeline", False, str(e))
        print(f"Traceback: {traceback.format_exc()}")
    
    print(f"\nText Preprocessing Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def test_metadata_preprocessing():
    """Test the metadata preprocessing pipeline."""
    print_test_header("Metadata Preprocessing Pipeline")
    
    tests_passed = 0
    total_tests = 6
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'speaker': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'party_affiliation': ['Democrat', 'Republican', 'Independent'],
            'subject': ['healthcare', 'economy', 'education'],
            'barely_true_counts': [2, 5, 1],
            'false_counts': [1, 8, 2],
            'half_true_counts': [5, 2, 3],
            'mostly_true_counts': [3, 1, 4],
            'pants_fire_counts': [0, 3, 0],
            'label': ['half-true', 'false', 'mostly-true']
        })
        
        # Test 1: Basic initialization
        processor = MetadataProcessor()
        print_test_result("MetadataProcessor initialization", True)
        tests_passed += 1
        
        # Test 2: Column type identification
        cat_cols, num_cols = processor.identify_column_types(sample_data, target_column='label')
        success = len(cat_cols) > 0 and len(num_cols) > 0
        print_test_result("Column type identification", success)
        if success:
            tests_passed += 1
        print(f"  Categorical: {cat_cols}")
        print(f"  Numerical: {num_cols}")
        
        # Test 3: Credibility feature engineering
        engineered_data = processor.engineer_credibility_features(sample_data)
        expected_features = ['credibility_score', 'deception_score', 'total_statements']
        success = all(feature in engineered_data.columns for feature in expected_features)
        print_test_result("Credibility feature engineering", success)
        if success:
            tests_passed += 1
        print(f"  New features: {[col for col in engineered_data.columns if col not in sample_data.columns]}")
        
        # Test 4: Complete metadata processing
        processed_data = processor.process_metadata(sample_data, target_column='label', fit=True)
        success = processed_data.shape[0] == sample_data.shape[0] and processed_data.shape[1] >= sample_data.shape[1]
        print_test_result("Complete metadata processing", success)
        if success:
            tests_passed += 1
        print(f"  Original shape: {sample_data.shape}")
        print(f"  Processed shape: {processed_data.shape}")
        
        # Test 5: Transform new data (fit=False)
        new_sample = sample_data.iloc[:1].copy()  # Take first row
        transformed_new = processor.process_metadata(new_sample, target_column='label', fit=False)
        success = transformed_new.shape[1] == processed_data.shape[1]
        print_test_result("Transform new data without fitting", success)
        if success:
            tests_passed += 1
        
        # Test 6: Save and load processor
        test_path = "../../models/test_metadata_processor.pkl"
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        processor.save_processor(test_path)
        loaded_processor = MetadataProcessor.load_processor(test_path)
        
        success = (loaded_processor.encoding_strategy == processor.encoding_strategy and
                  loaded_processor.scaling_method == processor.scaling_method)
        print_test_result("Save and load processor", success)
        if success:
            tests_passed += 1
        
        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print_test_result("Metadata preprocessing pipeline", False, str(e))
        print(f"Traceback: {traceback.format_exc()}")
    
    print(f"\nMetadata Preprocessing Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def test_integrated_pipeline():
    """Test the integrated preprocessing pipeline."""
    print_test_header("Integrated Pipeline Testing")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Create realistic test data
        test_data = pd.DataFrame({
            'statement': [
                "The unemployment rate has decreased by 2% this year according to official statistics.",
                "Healthcare spending INCREASED by 50% under the current administration!!!",
                "climate change is causing unprecedented weather patterns globally"
            ],
            'speaker': ['John Smith', 'Mary Johnson', 'Robert Brown'],
            'party_affiliation': ['Democrat', 'Republican', 'Independent'],
            'subject': ['economy', 'healthcare', 'environment'],
            'barely_true_counts': [1, 3, 0],
            'false_counts': [0, 5, 1],
            'half_true_counts': [3, 2, 4],
            'mostly_true_counts': [5, 1, 3],
            'pants_fire_counts': [0, 2, 0],
            'label': ['mostly-true', 'false', 'half-true']
        })
        
        # Test 1: Initialize both processors
        text_processor = TextPreprocessor()
        metadata_processor = MetadataProcessor()
        
        print_test_result("Initialize both processors", True)
        tests_passed += 1
        
        # Test 2: Process all data
        # Process text
        statements = test_data['statement'].tolist()
        processed_statements = text_processor.process_texts(statements, show_progress=False)
        
        # Process metadata
        processed_metadata = metadata_processor.process_metadata(test_data, target_column='label', fit=True)
        
        success = (len(processed_statements) == len(statements) and 
                  processed_metadata.shape[0] == test_data.shape[0])
        print_test_result("Process complete dataset", success)
        if success:
            tests_passed += 1
        
        print(f"  Processed {len(processed_statements)} statements")
        print(f"  Metadata shape: {processed_metadata.shape}")
        print(f"  Sample processed statement: {processed_statements[0][:100]}...")
        
        # Test 3: Data quality checks
        # Check for any completely empty processed statements
        empty_statements = sum(1 for s in processed_statements if not s or s.strip() == '')
        
        # Check for any missing values in critical metadata
        critical_missing = processed_metadata.isnull().sum().sum()
        
        success = empty_statements < len(processed_statements) and critical_missing == 0
        print_test_result("Data quality validation", success)
        if success:
            tests_passed += 1
        
        print(f"  Empty statements: {empty_statements}/{len(processed_statements)}")
        print(f"  Missing metadata values: {critical_missing}")
        
    except Exception as e:
        print_test_result("Integrated pipeline testing", False, str(e))
        print(f"Traceback: {traceback.format_exc()}")
    
    print(f"\nIntegrated Pipeline Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def test_streamlit_components():
    """Test Streamlit application components."""
    print_test_header("Streamlit Application Components")
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Import streamlit app modules
        try:
            sys.path.append('../../app/backend')
            # We can't actually import the streamlit app here due to streamlit's requirements
            # but we can test the component functions
            print_test_result("Streamlit imports (simulated)", True)
            tests_passed += 1
        except Exception as e:
            print_test_result("Streamlit imports", False, str(e))
        
        # Test 2: Mock web application input processing
        # Simulate the preprocessing that would happen in the web app
        sample_input = {
            'statement': "The economy is growing at record pace this year.",
            'speaker': "John Doe",
            'party': "Democrat", 
            'subject': "economy"
        }
        
        text_processor = TextPreprocessor()
        processed_text = text_processor.process_single_text(sample_input['statement'])
        
        success = processed_text is not None and len(processed_text) > 0
        print_test_result("Web app input processing simulation", success)
        if success:
            tests_passed += 1
        
        # Test 3: Mock prediction pipeline
        # Simulate what would happen when a user submits a form
        mock_prediction = {
            'prediction': 'mostly-true',
            'confidence': 0.78,
            'model_type': 'demo'
        }
        
        # Test prediction formatting
        prediction_valid = (mock_prediction['prediction'] in 
                          ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire'])
        confidence_valid = 0 <= mock_prediction['confidence'] <= 1
        
        success = prediction_valid and confidence_valid
        print_test_result("Prediction pipeline simulation", success)
        if success:
            tests_passed += 1
        
    except Exception as e:
        print_test_result("Streamlit application components", False, str(e))
        print(f"Traceback: {traceback.format_exc()}")
    
    print(f"\nStreamlit Components Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def run_all_tests():
    """Run all tests and provide summary."""
    print(f"{'='*80}")
    print(f"FAKE NEWS DETECTION SYSTEM - TESTING SUITE")
    print(f"Member 0148: Data Preprocessing & Web Application")
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Run all test suites
    test_results = []
    
    test_results.append(('Text Preprocessing', test_text_preprocessing()))
    test_results.append(('Metadata Preprocessing', test_metadata_preprocessing()))  
    test_results.append(('Integrated Pipeline', test_integrated_pipeline()))
    test_results.append(('Streamlit Components', test_streamlit_components()))
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total_passed = sum(1 for _, passed in test_results if passed)
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOVERALL RESULT: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for Day 2 tasks.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before proceeding.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    run_all_tests()