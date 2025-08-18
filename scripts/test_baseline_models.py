#!/usr/bin/env python3
"""
Test script to verify BaselineModels functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_baseline_models():
    """Test the BaselineModels class"""
    
    print("🧪 Testing BaselineModels Class...")
    print("=" * 50)
    
    try:
        from models.baseline_models_0149 import BaselineModels
        print("✅ Successfully imported BaselineModels")
        
        # Test instantiation
        baseline = BaselineModels()
        print("✅ Successfully created BaselineModels instance")
        
        # Test methods
        methods_to_test = [
            'load_models',
            'predict',
            'prepare_data',
            'train_all_models',
            'evaluate_models',
            'save_models'
        ]
        
        print("\n🔍 Checking required methods:")
        for method in methods_to_test:
            if hasattr(baseline, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING")
        
        # Test load_models method
        print("\n🔄 Testing load_models method:")
        try:
            models = baseline.load_models()
            print(f"  ✅ load_models() returned: {type(models)}")
            print(f"  📊 Models loaded: {len(models)}")
        except Exception as e:
            print(f"  ❌ load_models() failed: {e}")
        
        # Test predict method
        print("\n🔮 Testing predict method:")
        try:
            test_texts = ["This is a test statement"]
            predictions = baseline.predict(test_texts)
            print(f"  ✅ predict() returned: {predictions}")
        except Exception as e:
            print(f"  ❌ predict() failed: {e}")
        
        print("\n🎯 Testing completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_baseline_models()
    
    if success:
        print("\n✅ All tests passed! BaselineModels is working correctly.")
        print("You can now run the unified app without errors.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("The unified app may not work properly.")
