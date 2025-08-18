import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import os

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        tokens = word_tokenize(text)
        filtered_tokens = [w for w in tokens if w.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def process_single_text(self, text):
        """Process a single text input (required by web app)"""
        if pd.isna(text) or text == "":
            return ""
        
        cleaned = self.clean_text(text)
        no_stopwords = self.remove_stopwords(cleaned)
        return no_stopwords
    
    def process_batch(self, texts):
        """Process multiple texts efficiently"""
        processed = []
        for text in texts:
            cleaned = self.clean_text(text)
            no_stopwords = self.remove_stopwords(cleaned)
            processed.append(no_stopwords)
        return processed
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load a saved preprocessor from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                preprocessor = pickle.load(f)
            return preprocessor
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessor: {str(e)}")
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save preprocessor: {str(e)}")