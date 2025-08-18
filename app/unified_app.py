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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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
            
            # Fit on simple dummy data
            dummy_texts = [
                "This is a true statement",
                "This is a false statement", 
                "This is a half true statement"
            ]
            dummy_labels = [5, 1, 3]  # true, false, half-true
            
            fallback_pipeline.fit(dummy_texts, dummy_labels)
            
            # Add to models
            self.models['fallback_model'] = fallback_pipeline
            st.success("✅ Fallback model created successfully!")
            
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
            
            # Process metadata
            if metadata:
                processed_metadata = self.metadata_processor.process_metadata(metadata)
            else:
                processed_metadata = {}
            
            # Get predictions from models
            predictions = {}
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict([processed_text])
                        predictions[name] = pred[0] if hasattr(pred, '__len__') else pred
                    else:
                        predictions[name] = 'error'
                except Exception as e:
                    st.warning(f"⚠️ Model {name} failed: {e}")
                    predictions[name] = 'error'
            
            # Ensemble prediction
            ensemble_pred = self.compute_ensemble(predictions)
            
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
    
    def compute_ensemble(self, predictions):
        """Compute ensemble prediction"""
        valid_preds = [p for p in predictions.values() if p != 'error']
        if not valid_preds:
            return 'unknown'
        
        # Simple majority voting
        from collections import Counter
        most_common = Counter(valid_preds).most_common(1)[0][0]
        return most_common
    
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
    
    def home_page(self):
        """Home page"""
        st.header("🏠 Welcome to Unified Fake News Detection")
        
        st.markdown("""
        This system combines multiple approaches to detect fake news:
        - **Text Analysis**: TF-IDF and advanced NLP
        - **Metadata Analysis**: Speaker credibility, party affiliation
        - **Multiple Models**: Ensemble learning for better accuracy
        """)
        
        # System status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Loaded", len(self.models))
        with col2:
            st.metric("Total Predictions", len(self.prediction_history))
        with col3:
            st.metric("System Status", "🟢 Online")
    
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
        
        # Color coding
        if ensemble in ['true', 'mostly_true']:
            color = "green"
            icon = "✅"
        elif ensemble in ['false', 'pants_fire']:
            color = "red"
            icon = "❌"
        else:
            color = "orange"
            icon = "⚠️"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background-color: {color}20; border: 2px solid {color}; border-radius: 10px;">
            <h2>{icon} Prediction: {ensemble.upper()}</h2>
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
        ## Unified Fake News Detection System
        
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
