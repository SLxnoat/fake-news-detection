import pandas as pd
from .text_preprocessor_0148 import TextPreprocessor
from .metadata_processor_0148 import MetadataProcessor

class FeaturePipeline:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.metadata_processor = MetadataProcessor()
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline using available processors."""
        df_processed = df.copy()
        
        # Text preprocessing
        if 'statement' in df_processed.columns:
            df_processed['statement_clean'] = self.text_processor.process_batch(
                df_processed['statement'].fillna('')
            )
        
        # Delegate metadata and feature engineering to MetadataProcessor
        try:
            df_processed = self.metadata_processor.process_metadata(
                df_processed, target_column=None, fit=True
            )
        except Exception:
            # If processing fails, return at least the cleaned text
            pass
        
        return df_processed