from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

class MetadataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.onehot_encoders = {}
        
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                # Create label encoder
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(
                    df[col].fillna('unknown')
                )
                self.label_encoders[col] = le
                
        return df_encoded
    
    def create_credibility_features(self, df):
        """Create credibility-based features"""
        credibility_cols = [
            'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_fire_counts'
        ]
        
        # Total statements by speaker
        df['total_statements'] = df[credibility_cols].sum(axis=1)
        
        # Credibility ratio
        df['credibility_score'] = (
            df['mostly_true_counts'] + df['half_true_counts']
        ) / (df['total_statements'] + 1)  # +1 to avoid division by zero
        
        # False statement ratio
        df['false_ratio'] = (
            df['false_counts'] + df['pants_fire_counts']
        ) / (df['total_statements'] + 1)
        
        return df
    
    def scale_numerical(self, df, numerical_cols):
        """Scale numerical variables"""
        df_scaled = df.copy()
        if numerical_cols:
            df_scaled[numerical_cols] = self.scaler.fit_transform(
                df[numerical_cols]
            )
        return df_scaled
    
    def engineer_text_features(self, df):
        """Create text-based features"""
        df['text_length'] = df['statement'].str.len()
        df['word_count'] = df['statement'].str.split().str.len()
        df['avg_word_length'] = df['statement'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x else 0
        )
        df['exclamation_count'] = df['statement'].str.count('!')
        df['question_count'] = df['statement'].str.count('?')
        
        return df