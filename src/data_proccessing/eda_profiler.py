import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class DataProfiler:
    """Comprehensive data profiling for the LIAR dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile_results = {}
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        profile = {
            'basic_info': self._basic_info(),
            'data_quality': self._data_quality_assessment(),
            'feature_analysis': self._feature_analysis(),
            'label_analysis': self._label_analysis(),
            'text_analysis': self._text_analysis(),
            'metadata_analysis': self._metadata_analysis()
        }
        
        self.profile_results = profile
        return profile
    
    def _basic_info(self) -> Dict[str, Any]:
        """Basic dataset information"""
        return {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'dtypes': self.df.dtypes.to_dict(),
            'columns': list(self.df.columns)
        }
    
    def _data_quality_assessment(self) -> Dict[str, Any]:
        """Assess data quality issues"""
        return {
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'unique_values_per_column': self.df.nunique().to_dict()
        }
    
    def _feature_analysis(self) -> Dict[str, Any]:
        """Analyze individual features"""
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns
        
        numeric_analysis = {}
        for col in numeric_features:
            numeric_analysis[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'quartiles': self.df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        categorical_analysis = {}
        for col in categorical_features:
            top_values = self.df[col].value_counts().head(10)
            categorical_analysis[col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': top_values.to_dict(),
                'most_frequent': top_values.index[0] if len(top_values) > 0 else None
            }
        
        return {
            'numeric_features': numeric_analysis,
            'categorical_features': categorical_analysis
        }
    
    def _label_analysis(self) -> Dict[str, Any]:
        """Analyze label distribution and patterns"""
        if 'label' not in self.df.columns:
            return {}
        
        label_counts = self.df['label'].value_counts()
        label_percentages = self.df['label'].value_counts(normalize=True) * 100
        
        return {
            'distribution': label_counts.to_dict(),
            'percentages': label_percentages.to_dict(),
            'balance_ratio': label_counts.min() / label_counts.max(),
            'entropy': -np.sum(label_percentages/100 * np.log2(label_percentages/100))
        }
    
    def _text_analysis(self) -> Dict[str, Any]:
        """Analyze text features"""
        if 'statement' not in self.df.columns:
            return {}
        
        # Calculate text statistics
        text_lengths = self.df['statement'].str.len()
        word_counts = self.df['statement'].str.split().str.len()
        
        return {
            'text_length_stats': {
                'mean': text_lengths.mean(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'quartiles': text_lengths.quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'word_count_stats': {
                'mean': word_counts.mean(),
                'std': word_counts.std(),
                'min': word_counts.min(),
                'max': word_counts.max(),
                'quartiles': word_counts.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        }
    
    def _metadata_analysis(self) -> Dict[str, Any]:
        """Analyze metadata features"""
        metadata_features = ['speaker', 'subject', 'party_affiliation', 'speaker_job']
        analysis = {}
        
        for feature in metadata_features:
            if feature in self.df.columns:
                analysis[feature] = {
                    'unique_count': self.df[feature].nunique(),
                    'top_10': self.df[feature].value_counts().head(10).to_dict(),
                    'coverage': (self.df[feature].notna().sum() / len(self.df)) * 100
                }
        
        return analysis
    
    def save_profile_report(self, output_path: str = 'results/reports/data_profile.json'):
        """Save profile results to JSON"""
        import json
        
        if not self.profile_results:
            self.generate_profile()
        
        with open(output_path, 'w') as f:
            json.dump(self.profile_results, f, indent=2, default=str)

