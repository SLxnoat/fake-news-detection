import pandas as pd
from .text_processor import TextPreprocessor
from .metadata_processor import MetadataProcessor

class FeaturePipeline:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.metadata_processor = MetadataProcessor()
        
    def fit_transform(self, df):
        """Complete feature engineering pipeline"""
        df_processed = df.copy()
        
        # Text preprocessing
        df_processed['statement_clean'] = self.text_processor.process_batch(
            df_processed['statement'].fillna('')
        )
        
        # Text feature engineering
        df_processed = self.metadata_processor.engineer_text_features(df_processed)
        
        # Credibility features
        df_processed = self.metadata_processor.create_credibility_features(df_processed)
        
        # Categorical encoding
        categorical_cols = ['speaker', 'party_affiliation', 'subject', 'state_info', 'venue']
        df_processed = self.metadata_processor.encode_categorical(
            df_processed, categorical_cols
        )
        
        # Numerical scaling
        numerical_cols = [
            'text_length', 'word_count', 'avg_word_length', 
            'credibility_score', 'false_ratio'
        ]
        df_processed = self.metadata_processor.scale_numerical(
            df_processed, numerical_cols
        )
        
        return df_processed