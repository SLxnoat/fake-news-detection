#!/usr/bin/env python3
"""
Unified Fake News Detection Application
Consolidates all functionality into one comprehensive app
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add src to path (robust to current working directory)
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from preprocessing.text_preprocessor_0148 import TextPreprocessor
    from preprocessing.metadata_processor_0148 import MetadataProcessor
    from models.baseline_models_0149 import BaselineModels
    print("✅ Core modules imported")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")

class UnifiedApp:
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.metadata_processor = MetadataProcessor()
        self.baseline_models = BaselineModels()
        self.models = {}
        self.prediction_history = []
        # Canonical label set (LIAR schema)
        self.canonical_labels = [
            'pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true'
        ]
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        try:
            baseline_models = self.baseline_models.load_models()
            if baseline_models:
                self.models.update(baseline_models)
                st.success(f"✅ Loaded {len(self.models)} models")
            else:
                st.warning("⚠️ No baseline models were loaded. This may be because:")
                st.warning("1. Models haven't been trained yet")
                st.warning("2. Model files are missing")
                st.warning("3. Model files are corrupted")
                st.info("💡 You can train models using: python scripts/train_baseline_models.py")
                
                # Try to create a simple fallback model
                self.create_fallback_model()
                
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("💡 This error suggests the BaselineModels class may be missing methods")
            st.info("💡 Try running: python scripts/train_baseline_models.py")
            
            # Try to create a simple fallback model
            self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create a simple fallback model for basic functionality"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            import pickle
            
            st.info("🔄 Creating fallback model for basic functionality...")
            
            # Create a simple fallback model
            fallback_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100)),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            # Fit on simple dummy data with more variety
            dummy_texts = [
                "This is a true statement about facts",
                "This is completely false and wrong", 
                "This is partially true but misleading",
                "This is mostly accurate information",
                "This is barely true with many errors",
                "This is completely made up nonsense"
            ]
            dummy_labels = [5, 1, 3, 4, 2, 0]  # true, false, half-true, mostly-true, barely-true, pants-fire
            
            fallback_pipeline.fit(dummy_texts, dummy_labels)
            
            # Add to models
            self.models['fallback_model'] = fallback_pipeline
            st.success("✅ Fallback model created successfully!")
            st.info("💡 Fallback model uses: 5=true, 4=mostly-true, 3=half-true, 2=barely-true, 1=false, 0=pants-fire")
            
        except Exception as e:
            st.error(f"❌ Could not create fallback model: {e}")
            st.warning("⚠️ Some features may not work without models")
    
    def make_prediction(self, text, metadata=None):
        """Make prediction using available models"""
        try:
            # Check if we have any models
            if not self.models:
                st.error("❌ No models available for prediction")
                st.info("💡 Please train models first using: python scripts/train_baseline_models.py")
                return {'error': 'No models available'}
            
            # Preprocess text
            processed_text = self.text_processor.process_single_text(text)
            
            # Process metadata (keep raw for now; processing expects a DataFrame)
            processed_metadata = metadata or {}
            
            # Get predictions from models
            predictions = {}
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict([processed_text])
                        pred_val = pred[0] if hasattr(pred, '__len__') else pred
                        
                        # Map numeric predictions to label strings if applicable
                        try:
                            import numpy as _np
                            if isinstance(pred_val, (int, _np.integer)):
                                mapped = self.baseline_models.reverse_label_mapping.get(int(pred_val), str(pred_val))
                            else:
                                mapped = str(pred_val)
                        except Exception:
                            mapped = str(pred_val)
                        predictions[name] = self.normalize_label(mapped)
                    else:
                        predictions[name] = 'error'
                except Exception as e:
                    st.warning(f"⚠️ Model {name} failed: {e}")
                    predictions[name] = 'error'
            
            # Debug: show what we got
            st.write("🔍 Debug: Model predictions:", predictions)
            
            # Ensemble prediction
            ensemble_pred = self.compute_ensemble(predictions)
            st.write("🔍 Debug: Final ensemble:", ensemble_pred)
            
            result = {
                'text': text,
                'processed_text': processed_text,
                'metadata': metadata or {},
                'model_predictions': predictions,
                'ensemble_prediction': ensemble_pred,
                'timestamp': datetime.now()
            }
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return {'error': str(e)}
    
    def normalize_label(self, label):
        """Normalize various label formats to canonical hyphenated strings."""
        if label is None:
            return 'unknown'
        try:
            # If numeric as string, map via baseline mapping if possible
            if isinstance(label, (int,)):
                return self.baseline_models.reverse_label_mapping.get(int(label), 'unknown')
            label_str = str(label).strip().lower().replace('_', '-').replace(' ', '-')
            # Common aliases
            aliases = {
                'pantsfire': 'pants-fire',
                'pants-fire': 'pants-fire',
                'pants_fire': 'pants-fire',
                'mostly_true': 'mostly-true',
                'mostlytrue': 'mostly-true',
                'half_true': 'half-true',
                'halftrue': 'half-true',
                'barely_true': 'barely-true',
                'barelytrue': 'barely-true'
            }
            if label_str in aliases:
                return aliases[label_str]
            if label_str in self.canonical_labels:
                return label_str
            return label_str
        except Exception:
            return 'unknown'

    def compute_ensemble(self, predictions):
        """Compute ensemble prediction"""
        valid_preds = [p for p in predictions.values() if p != 'error']
        if not valid_preds:
            return 'unknown'
        
        # Simple majority voting on normalized labels
        from collections import Counter
        normalized = [self.normalize_label(p) for p in valid_preds]
        most_common = Counter(normalized).most_common()
        if not most_common:
            return 'unknown'
        return most_common[0][0]
    
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="Unified Fake News Detection",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Unified Fake News Detection System")
        st.markdown("---")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Home", "🔬 Prediction", "📊 Analytics", "📁 Data Explorer", "ℹ️ About"]
        )
        
        if page == "🏠 Home":
            self.home_page()
        elif page == "🔬 Prediction":
            self.prediction_page()
        elif page == "📊 Analytics":
            self.analytics_page()
        elif page == "📁 Data Explorer":
            self.data_explorer_page()
        elif page == "ℹ️ About":
            self.about_page()
    
    def test_models(self):
        """Test models with sample inputs to see what they predict"""
        st.subheader("🧪 Model Testing")
        
        test_texts = [
            "This is a completely true statement about verified facts.",
            "This is completely false and made up information.",
            "This statement is partially true but has some inaccuracies."
        ]
        
        for i, test_text in enumerate(test_texts):
            st.write(f"**Test {i+1}**: {test_text}")
            
            processed = self.text_processor.process_single_text(test_text)
            st.write(f"Processed: {processed}")
            
            for name, model in self.models.items():
                try:
                    pred = model.predict([processed])
                    pred_val = pred[0] if hasattr(pred, '__len__') else pred
                    mapped = self.baseline_models.reverse_label_mapping.get(int(pred_val), str(pred_val))
                    st.write(f"  {name}: {pred_val} → '{mapped}'")
                except Exception as e:
                    st.write(f"  {name}: Error - {e}")
            st.write("---")

    def home_page(self):
        """Home page"""
        st.header("🏠 Welcome to Unified Fake News Detection")
        
        st.markdown("""
        A streamlined system for analyzing statements with classic ML models and
        robust preprocessing. Use the sidebar to run predictions, view analytics, and explore data.
        """)
        
        # System status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Loaded", len(self.models))
        with col2:
            st.metric("Total Predictions", len(self.prediction_history))
        with col3:
            st.metric("System Status", "🟢 Online")

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Quick Tips")
            st.markdown("""
            - Use clear, single statements for best results
            - Include optional metadata (speaker, party, subject) when available
            - Train baseline models if none are loaded: `python scripts/train_baseline_models.py`
            """)
        with colB:
            st.subheader("Capabilities")
            st.markdown("""
            - Text preprocessing (NLTK)
            - TF-IDF based baseline models (LogReg, RandomForest)
            - Simple ensemble over available models
            - Basic analytics of recent predictions
            """)
        
        # Add model testing section
        if st.checkbox("🧪 Test Models"):
            self.test_models()
    
    def prediction_page(self):
        """Prediction interface"""
        st.header("🔬 Statement Analysis")
        
        with st.form("prediction_form"):
            text = st.text_area("Enter statement to analyze:", height=150)
            speaker = st.text_input("Speaker (optional):")
            party = st.selectbox("Party (optional):", ["", "Democrat", "Republican", "Independent"])
            subject = st.text_input("Subject (optional):")
            
            submit = st.form_submit_button("🔍 Analyze")
        
        if submit and text.strip():
            with st.spinner("Analyzing..."):
                metadata = {'speaker': speaker, 'party': party, 'subject': subject}
                result = self.make_prediction(text, metadata)
                
                if 'error' not in result:
                    self.display_results(result)
                else:
                    st.error(f"Error: {result['error']}")
    
    def display_results(self, result):
        """Display prediction results"""
        ensemble = result['ensemble_prediction']
        
        # Normalize label for display and color logic
        ensemble_norm = (ensemble or 'unknown')
        if not isinstance(ensemble_norm, str):
            ensemble_norm = str(ensemble_norm)
        ensemble_norm = ensemble_norm.strip().lower()

        # Color coding (labels use hyphens)
        if ensemble_norm in ['true', 'mostly-true']:
            color = "green"
            icon = "✅"
        elif ensemble_norm in ['false', 'pants-fire']:
            color = "red"
            icon = "❌"
        else:
            color = "orange"
            icon = "⚠️"
        
        display_label = ensemble_norm.replace('-', ' ').upper()
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: {color}20; border: 2px solid {color}; border-radius: 10px;">
            <h2>{icon} Prediction: {display_label}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model predictions
        st.subheader("Model Predictions:")
        for name, pred in result['model_predictions'].items():
            st.write(f"**{name}**: {pred}")
        
        # Details
        with st.expander("Analysis Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.info(result['text'])
            with col2:
                st.write("**Processed Text:**")
                st.code(result['processed_text'])
    
    def analytics_page(self):
        """Analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        if not self.prediction_history:
            st.info("No predictions yet. Make some predictions to see analytics!")
            return
        
        # Prediction distribution
        predictions = [r['ensemble_prediction'] for r in self.prediction_history]
        pred_counts = pd.Series(predictions).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, title="Prediction Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=pred_counts.index, y=pred_counts.values, title="Prediction Counts")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions
        st.subheader("Recent Predictions")
        recent_df = pd.DataFrame([
            {
                'Text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                'Prediction': r['ensemble_prediction'],
                'Timestamp': r['timestamp']
            }
            for r in self.prediction_history[-10:]
        ])
        st.dataframe(recent_df, use_container_width=True)
    
    def data_explorer_page(self):
        """Data exploration"""
        st.header("📁 Data Explorer")
        
        # Load sample data
        data_path = "data/raw/train.tsv"
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path, sep='\t', header=0)
                st.write(f"Dataset shape: {df.shape}")
                st.write(f"Columns: {list(df.columns)}")
                
                st.subheader("Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column analysis
                if st.checkbox("Show column analysis"):
                    selected_col = st.selectbox("Select column:", df.columns)
                    if selected_col:
                        col_data = df[selected_col]
                        st.write(f"**{selected_col}** analysis:")
                        st.write(f"Unique values: {col_data.nunique()}")
                        if col_data.dtype in ['object', 'string']:
                            st.write("Top values:")
                            st.write(col_data.value_counts().head(10))
                        else:
                            st.write(f"Mean: {col_data.mean():.2f}")
                            st.write(f"Std: {col_data.std():.2f}")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.warning("Sample data not found")
    
    def about_page(self):
        """About page"""
        st.header("ℹ️ About")
        st.markdown("""
        ## About  Unified Fake News Detection System

        **Team Members:**
        - ITBIN-2211-0149: Baseline Models
        - ITBIN-2211-0169: Cross-Validation
        - ITBIN-2211-0173: BERT Integration
        - ITBIN-2211-0184: EDA & Visualizations

        **Features:**
        - Text preprocessing and analysis
        - Metadata processing
        - Multiple ML models
        - Ensemble predictions
        - Real-time analytics

        **Technology:**
        - Python, Streamlit, scikit-learn
        - NLTK, BERT, TF-IDF
        - Plotly, Pandas, NumPy
        """)

if __name__ == "__main__":
    app = UnifiedApp()
    app.run()
