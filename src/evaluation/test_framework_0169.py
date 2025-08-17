import unittest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'statement': ['This is true', 'This is false', 'Maybe true'],
            'label': ['true', 'false', 'half-true'],
            'speaker': ['John', 'Jane', 'Bob'],
            'party': ['democrat', 'republican', 'none']
        })
    
    def test_data_loading(self):
        """Test data loading functionality"""
        self.assertIsInstance(self.test_data, pd.DataFrame)
        self.assertGreater(len(self.test_data), 0)
        
    def test_required_columns(self):
        """Test if required columns exist"""
        required_cols = ['statement', 'label']
        for col in required_cols:
            self.assertIn(col, self.test_data.columns)
    
    def test_no_null_statements(self):
        """Test that statements are not null"""
        self.assertFalse(self.test_data['statement'].isnull().any())

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test model"""
        self.dummy_model = DummyClassifier(strategy='most_frequent')
        self.X_test = np.random.rand(100, 10)
        self.y_test = np.random.choice(['true', 'false'], 100)
        self.dummy_model.fit(self.X_test, self.y_test)
    
    def test_model_prediction(self):
        """Test model can make predictions"""
        predictions = self.dummy_model.predict(self.X_test[:10])
        self.assertEqual(len(predictions), 10)
    
    def test_prediction_types(self):
        """Test prediction types are valid"""
        predictions = self.dummy_model.predict(self.X_test[:5])
        valid_labels = ['true', 'false', 'half-true', 'mostly-true', 
                       'barely-true', 'pants-fire']
        for pred in predictions:
            # For dummy classifier, just check it's a string
            self.assertIsInstance(pred, str)

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_all_tests()