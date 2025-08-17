import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import os

class MetadataProcessor:
    """
    Comprehensive metadata preprocessing pipeline for fake news detection.
    Handles categorical encoding, numerical scaling, and feature engineering.
    """
    
    def __init__(self, encoding_strategy='label', scaling_method='standard', handle_unknown='ignore'):
        """
        Initialize the metadata processor.
        
        Args:
            encoding_strategy (str): 'label', 'onehot', or 'target'
            scaling_method (str): 'standard', 'minmax', or 'none'
            handle_unknown (str): How to handle unknown categories ('ignore' or 'error')
        """
        self.encoding_strategy = encoding_strategy
        self.scaling_method = scaling_method
        self.handle_unknown = handle_unknown
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.scaler = None
        self.imputer = SimpleImputer(strategy='most_frequent')
        
        # Initialize scaler based on method
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        
        # Track processed columns
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_names = []
        
    def identify_column_types(self, df, target_column=None):
        """
        Identify categorical and numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column to exclude
            
        Returns:
            tuple: (categorical_columns, numerical_columns)
        """
        # Exclude target column
        columns_to_process = [col for col in df.columns if col != target_column]
        
        categorical_cols = []
        numerical_cols = []
        
        for col in columns_to_process:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_cols.append(col)
        
        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols
        
        return categorical_cols, numerical_cols
    
    def engineer_credibility_features(self, df):
        """
        Engineer credibility-based features from speaker history.
        
        Args:
            df (pd.DataFrame): Input dataframe with credibility counts
            
        Returns:
            pd.DataFrame: DataFrame with additional credibility features
        """
        df_engineered = df.copy()
        
        # Define credibility columns (based on LIAR dataset)
        credibility_cols = [
            'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_fire_counts'
        ]
        
        # Check if credibility columns exist
        available_cred_cols = [col for col in credibility_cols if col in df.columns]
        
        if available_cred_cols:
            # Calculate total statements
            df_engineered['total_statements'] = df_engineered[available_cred_cols].sum(axis=1)
            
            # Calculate credibility score (ratio of mostly true + half true to total)
            df_engineered['credibility_score'] = (
                (df_engineered.get('mostly_true_counts', 0) + 
                 df_engineered.get('half_true_counts', 0)) / 
                (df_engineered['total_statements'] + 1)  # Add 1 to avoid division by zero
            )
            
            # Calculate deception score (ratio of false + pants fire to total)
            df_engineered['deception_score'] = (
                (df_engineered.get('false_counts', 0) + 
                 df_engineered.get('pants_fire_counts', 0)) / 
                (df_engineered['total_statements'] + 1)
            )
            
            # Speaker experience (total statements)
            df_engineered['speaker_experience'] = np.log1p(df_engineered['total_statements'])
            
            # Reliability categories
            df_engineered['reliability_category'] = pd.cut(
                df_engineered['credibility_score'],
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High']
            )
            
        return df_engineered
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using the specified strategy.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoders
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            # Handle missing values
            df_encoded[col] = df_encoded[col].fillna('unknown')
            
            if self.encoding_strategy == 'label':
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Transform only
                    if col in self.label_encoders:
                        # Handle unknown categories
                        known_classes = set(self.label_encoders[col].classes_)
                        df_encoded[col + '_mapped'] = df_encoded[col].apply(
                            lambda x: x if x in known_classes else 'unknown'
                        )
                        df_encoded[col + '_encoded'] = self.label_encoders[col].transform(
                            df_encoded[col + '_mapped'].astype(str)
                        )
                        df_encoded.drop(col + '_mapped', axis=1, inplace=True)
                
            elif self.encoding_strategy == 'onehot':
                if fit:
                    # Fit and transform
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_cols = ohe.fit_transform(df_encoded[[col]])
                    feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    
                    # Add encoded columns
                    for i, fname in enumerate(feature_names):
                        df_encoded[fname] = encoded_cols[:, i]
                    
                    self.onehot_encoders[col] = ohe
                else:
                    # Transform only
                    if col in self.onehot_encoders:
                        ohe = self.onehot_encoders[col]
                        encoded_cols = ohe.transform(df_encoded[[col]])
                        feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                        
                        # Add encoded columns
                        for i, fname in enumerate(feature_names):
                            df_encoded[fname] = encoded_cols[:, i]
        
        # Drop original categorical columns after encoding
        df_encoded.drop(columns=[col for col in self.categorical_columns if col in df_encoded.columns], 
                       inplace=True)
        
        return df_encoded
    
    def scale_numerical_features(self, df, fit=True):
        """
        Scale numerical features using the specified method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler
            
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        """
        if self.scaling_method == 'none' or not self.numerical_columns:
            return df
        
        df_scaled = df.copy()
        
        # Get numerical columns that exist in dataframe
        available_num_cols = [col for col in self.numerical_columns if col in df.columns]
        
        if available_num_cols and self.scaler:
            if fit:
                df_scaled[available_num_cols] = self.scaler.fit_transform(df[available_num_cols])
            else:
                df_scaled[available_num_cols] = self.scaler.transform(df[available_num_cols])
        
        return df_scaled
    
    def process_metadata(self, df, target_column=None, fit=True):
        """
        Complete metadata processing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Target column to exclude from processing
            fit (bool): Whether to fit the preprocessors
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Make a copy
        df_processed = df.copy()
        
        # Identify column types
        if fit:
            self.identify_column_types(df_processed, target_column)
        
        # Engineer credibility features
        df_processed = self.engineer_credibility_features(df_processed)
        
        # Update column types after feature engineering
        if fit:
            # Re-identify after feature engineering
            new_numerical = [col for col in df_processed.columns 
                           if col not in df.columns and df_processed[col].dtype in ['int64', 'float64']]
            self.numerical_columns.extend(new_numerical)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        # Scale numerical features
        df_processed = self.scale_numerical_features(df_processed, fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = [col for col in df_processed.columns if col != target_column]
        
        return df_processed
    
    def get_feature_names(self):
        """Get the names of processed features."""
        return self.feature_names
    
    def save_processor(self, filepath):
        """Save the metadata processor."""
        processor_data = {
            'encoding_strategy': self.encoding_strategy,
            'scaling_method': self.scaling_method,
            'handle_unknown': self.handle_unknown,
            'label_encoders': self.label_encoders,
            'onehot_encoders': self.onehot_encoders,
            'scaler': self.scaler,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_names': self.feature_names
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
    
    @classmethod
    def load_processor(cls, filepath):
        """Load a saved metadata processor."""
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        processor = cls(
            encoding_strategy=processor_data['encoding_strategy'],
            scaling_method=processor_data['scaling_method'],
            handle_unknown=processor_data['handle_unknown']
        )
        
        processor.label_encoders = processor_data['label_encoders']
        processor.onehot_encoders = processor_data['onehot_encoders']
        processor.scaler = processor_data['scaler']
        processor.categorical_columns = processor_data['categorical_columns']
        processor.numerical_columns = processor_data['numerical_columns']
        processor.feature_names = processor_data['feature_names']
        
        return processor

# Example usage and testing
if __name__ == "__main__":
    # Create sample data similar to LIAR dataset
    sample_data = pd.DataFrame({
        'speaker': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'John Doe'],
        'party_affiliation': ['Democrat', 'Republican', 'Independent', 'Democrat', 'Democrat'],
        'subject': ['healthcare', 'economy', 'education', 'healthcare', 'economy'],
        'barely_true_counts': [2, 5, 1, 0, 2],
        'false_counts': [1, 8, 2, 1, 1],
        'half_true_counts': [5, 2, 3, 4, 5],
        'mostly_true_counts': [3, 1, 4, 6, 3],
        'pants_fire_counts': [0, 3, 0, 0, 0],
        'label': ['half-true', 'false', 'mostly-true', 'true', 'half-true']
    })
    
    print("Original Data:")
    print(sample_data)
    print("\nData Types:")
    print(sample_data.dtypes)
    
    # Test metadata processor
    processor = MetadataProcessor(encoding_strategy='label', scaling_method='standard')
    
    # Process the data
    processed_data = processor.process_metadata(sample_data, target_column='label', fit=True)
    
    print("\nProcessed Data:")
    print(processed_data)
    print("\nProcessed Data Types:")
    print(processed_data.dtypes)
    print("\nFeature Names:")
    print(processor.get_feature_names())