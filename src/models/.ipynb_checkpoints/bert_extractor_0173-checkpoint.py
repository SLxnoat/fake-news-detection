import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BERTFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        """
        Initialize BERT Feature Extractor
        
        Args:
            model_name: BERT model variant to use
            max_length: Maximum sequence length for tokenization
        """
        print(f"Initializing BERT Feature Extractor with {model_name}")
        self.model_name = model_name
        self.max_length = max_length
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("BERT model loaded successfully!")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise e
    
    def preprocess_text(self, texts):
        """
        Preprocess texts for BERT input
        
        Args:
            texts: List of text strings
            
        Returns:
            List of cleaned texts
        """
        processed_texts = []
        for text in texts:
            if pd.isna(text) or text is None:
                processed_texts.append("")
            else:
                # Basic preprocessing
                text = str(text).strip()
                # Remove excessive whitespace
                text = ' '.join(text.split())
                processed_texts.append(text)
        
        return processed_texts
    
    def extract_features_batch(self, texts, batch_size=16):
        """
        Extract BERT features from texts using batching for efficiency
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of BERT embeddings (num_samples, 768)
        """
        # Preprocess texts
        texts = self.preprocess_text(texts)
        
        all_features = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT features"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Extract features
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Use [CLS] token representation (first token)
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_features.append(batch_features)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                # Create zero features for failed batch
                batch_features = np.zeros((len(batch_texts), 768))
                all_features.append(batch_features)
        
        # Combine all features
        features = np.vstack(all_features)
        print(f"Extracted features shape: {features.shape}")
        
        return features
    
    def save_features(self, features, filepath):
        """Save extracted features to file"""
        try:
            np.save(filepath, features)
            print(f"Features saved to {filepath}")
        except Exception as e:
            print(f"Error saving features: {e}")
    
    def load_features(self, filepath):
        """Load features from file"""
        try:
            features = np.load(filepath)
            print(f"Features loaded from {filepath}, shape: {features.shape}")
            return features
        except Exception as e:
            print(f"Error loading features: {e}")
            return None

class LIARDataset:
    """Class to handle LIAR dataset loading and preprocessing"""
    
    def __init__(self, data_path="data/raw"):
        self.data_path = data_path
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
        # Label mapping
        self.label_mapping = {
            'pants-fire': 0,
            'false': 1,
            'barely-true': 2,
            'half-true': 3,
            'mostly-true': 4,
            'true': 5
        }
        
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def load_data(self):
        """Load all LIAR dataset files"""
        try:
            # Define column names based on LIAR dataset structure
            columns = [
                'id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job',
                'state_info', 'party_affiliation', 'barely_true_counts',
                'false_counts', 'half_true_counts', 'mostly_true_counts',
                'pants_fire_counts', 'context'
            ]
            
            # Load datasets
            self.train_df = pd.read_csv(
                os.path.join(self.data_path, 'train.tsv'), 
                sep='\t', 
                header=None, 
                names=columns
            )
            
            self.valid_df = pd.read_csv(
                os.path.join(self.data_path, 'valid.tsv'), 
                sep='\t', 
                header=None, 
                names=columns
            )
            
            self.test_df = pd.read_csv(
                os.path.join(self.data_path, 'test.tsv'), 
                sep='\t', 
                header=None, 
                names=columns
            )
            
            print(f"Train set: {len(self.train_df)} samples")
            print(f"Validation set: {len(self.valid_df)} samples")
            print(f"Test set: {len(self.test_df)} samples")
            
            # Encode labels
            for df in [self.train_df, self.valid_df, self.test_df]:
                df['label_encoded'] = df['label'].map(self.label_mapping)
            
            print("Data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_basic_stats(self):
        """Get basic statistics about the dataset"""
        if self.train_df is None:
            print("Please load data first using load_data()")
            return
        
        print("\n=== DATASET STATISTICS ===")
        
        # Label distribution
        print("\nLabel Distribution (Train):")
        print(self.train_df['label'].value_counts())
        
        # Text length statistics
        self.train_df['text_length'] = self.train_df['statement'].str.len()
        print(f"\nText Length Statistics:")
        print(f"Mean: {self.train_df['text_length'].mean():.2f}")
        print(f"Median: {self.train_df['text_length'].median():.2f}")
        print(f"Max: {self.train_df['text_length'].max()}")
        print(f"Min: {self.train_df['text_length'].min()}")
        
        # Missing values
        print(f"\nMissing Values:")
        missing = self.train_df.isnull().sum()
        print(missing[missing > 0])

# Test and Demo Functions
def test_bert_extractor():
    """Test BERT feature extractor with sample data"""
    print("=== TESTING BERT FEATURE EXTRACTOR ===")
    
    # Sample texts
    sample_texts = [
        "The president announced new policies today.",
        "Healthcare costs have increased significantly.",
        "Climate change is affecting weather patterns.",
        "",  # Test empty string
        "Short text.",
        "This is a much longer text that should test the tokenizer's ability to handle various lengths of input text and ensure proper truncation when necessary."
    ]
    
    # Initialize extractor
    extractor = BERTFeatureExtractor(max_length=64)  # Smaller for testing
    
    # Extract features
    features = extractor.extract_features_batch(sample_texts, batch_size=2)
    
    print(f"Sample features shape: {features.shape}")
    print(f"Feature statistics - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    
    return extractor, features

def main():
    """Main function to test BERT setup"""
    print("Starting BERT Feature Extraction Setup - Member 0173")
    print("=" * 60)
    
    # Test 1: BERT Feature Extractor
    try:
        extractor, features = test_bert_extractor()
        print(" BERT Feature Extractor test passed!")
    except Exception as e:
        print(f" BERT Feature Extractor test failed: {e}")
        return
    
    # Test 2: Dataset Loading
    print("\n=== TESTING DATASET LOADING ===")
    try:
        dataset = LIARDataset()
        success = dataset.load_data()
        
        if success:
            dataset.get_basic_stats()
            print(" Dataset loading test passed!")
        else:
            print(" Dataset files not found. Please ensure files are in data/raw/")
            print("Required files: train.tsv, valid.tsv, test.tsv")
    except Exception as e:
        print(f" Dataset loading test failed: {e}")
    
    print(f"\n Day 1 Tasks Completed:")
    print(" BERT environment setup")
    print(" Feature extraction pipeline")
    print(" Dataset loading utilities")
    print(" Basic testing framework")
    
    print(f"\n Next Steps for Day 2:")
    print("- Design hybrid neural network architecture")
    print("- Implement feature fusion layers")
    print("- Create training pipeline")
    print("- Test model forward pass")

if __name__ == "__main__":
    main()