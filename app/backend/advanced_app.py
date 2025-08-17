import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import hashlib
from typing import Dict, List, Tuple, Any
import requests
import io
import base64

# Add src to path
sys.path.append('../../src')
from preprocessing.text_processor import TextProcessor
from preprocessing.metadata_processor import MetadataProcessor

class FakeNewsDetectionApp:
    def __init__(self):
        self.text_processor = None
        self.metadata_processor = None
        self.models = {}
        self.prediction_history = []
        self.session_state = st.session_state
        
        # Initialize session state
        if 'authenticated' not in self.session_state:
            self.session_state.authenticated = False
        if 'username' not in self.session_state:
            self.session_state.username = ""
        if 'prediction_count' not in self.session_state:
            self.session_state.prediction_count = 0
        if 'batch_results' not in self.session_state:
            self.session_state.batch_results = []

    def load_models(self):
        """Load all available models"""
        model_dir = "../../models"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pkl'):
                    model_name = file.replace('.pkl', '')
                    try:
                        with open(os.path.join(model_dir, file), 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        st.success(f"Loaded model: {model_name}")
                    except Exception as e:
                        st.warning(f"Failed to load {model_name}: {str(e)}")

    def authenticate_user(self, username: str, password: str) -> bool:
        """Simple authentication (replace with proper auth in production)"""
        # Demo credentials - replace with proper authentication
        demo_users = {
            "admin": "password123",
            "analyst": "analyst123",
            "demo": "demo123"
        }
        return demo_users.get(username) == password

    def login_page(self):
        """User authentication page"""
        st.title("🔐 Fake News Detection System - Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown("### Please log in to continue")
                
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                col_login, col_demo = st.columns(2)
                
                with col_login:
                    if st.button("Login", type="primary", use_container_width=True):
                        if self.authenticate_user(username, password):
                            self.session_state.authenticated = True
                            self.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                
                with col_demo:
                    if st.button("Demo Login", use_container_width=True):
                        self.session_state.authenticated = True
                        self.session_state.username = "demo_user"
                        st.rerun()
                
                st.markdown("---")
                st.info("Demo credentials: admin/password123, analyst/analyst123, demo/demo123")

    def sidebar_navigation(self):
        """Enhanced sidebar with navigation and user info"""
        with st.sidebar:
            st.markdown(f"### Welcome, {self.session_state.username}! 👋")
            st.markdown(f"**Predictions made:** {self.session_state.prediction_count}")
            
            if st.button("Logout", type="secondary"):
                self.session_state.authenticated = False
                self.session_state.username = ""
                st.rerun()
            
            st.markdown("---")
            
            # Navigation menu
            page = st.selectbox(
                "Navigation",
                ["Single Prediction", "Batch Processing", "Analytics Dashboard", 
                 "Model Comparison", "Data Explorer", "System Status"]
            )
            
            # Quick stats
            st.markdown("### Quick Stats")
            if len(self.prediction_history) > 0:
                recent_predictions = self.prediction_history[-10:]
                fake_count = sum(1 for p in recent_predictions if p.get('prediction') == 'fake')
                st.metric("Recent Fake Rate", f"{fake_count/len(recent_predictions)*100:.1f}%")
            else:
                st.metric("Recent Fake Rate", "0%")
            
            # Model status
            st.markdown("### Model Status")
            if self.models:
                st.success(f"✅ {len(self.models)} models loaded")
            else:
                st.warning("⚠️ No models loaded")
        
        return page

    def single_prediction_page(self):
        """Enhanced single prediction interface"""
        st.title("🔍 Single News Prediction")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Text Input", "URL Analysis", "File Upload"])
        
        if input_method == "Text Input":
            self.text_input_interface()
        elif input_method == "URL Analysis":
            self.url_analysis_interface()
        else:
            self.file_upload_interface()

    def text_input_interface(self):
        """Text input interface with real-time analysis"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            statement = st.text_area(
                "Enter news statement to analyze:",
                height=200,
                placeholder="Type or paste the news statement here..."
            )
            
            # Real-time word count and basic analysis
            if statement:
                word_count = len(statement.split())
                char_count = len(statement)
                st.caption(f"Words: {word_count} | Characters: {char_count}")
                
                # Basic sentiment preview
                if word_count > 10:
                    st.info("💡 Sufficient text length for analysis")
                else:
                    st.warning("⚠️ Consider adding more text for better accuracy")
        
        with col2:
            st.markdown("### Metadata (Optional)")
            speaker = st.text_input("Speaker", placeholder="e.g., John Doe")
            job = st.text_input("Job", placeholder="e.g., Politician")
            state = st.selectbox("State", ["", "California", "Texas", "New York", "Florida", "Other"])
            party = st.selectbox("Party", ["", "Republican", "Democrat", "Independent", "Other"])
            context = st.selectbox("Context", ["", "debate", "interview", "rally", "statement", "other"])
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8, 0.05)
            show_explanation = st.checkbox("Show prediction explanation", True)
            enable_uncertainty = st.checkbox("Show prediction uncertainty", True)
        
        # Prediction button
        if st.button("🎯 Analyze News", type="primary", use_container_width=True):
            if statement.strip():
                self.make_prediction(
                    statement, speaker, job, state, party, context,
                    confidence_threshold, show_explanation, enable_uncertainty
                )
            else:
                st.error("Please enter a news statement to analyze")

    def url_analysis_interface(self):
        """URL analysis interface"""
        st.markdown("### 🌐 URL Analysis")
        
        url = st.text_input("Enter news article URL:", placeholder="https://example.com/news-article")
        
        if st.button("Extract and Analyze", type="primary"):
            if url:
                try:
                    # Simulate URL content extraction (implement with requests/BeautifulSoup)
                    st.info("🔄 Extracting content from URL...")
                    time.sleep(2)  # Simulate processing time
                    
                    # Mock extracted content
                    extracted_text = "This is a simulated extracted article text for demonstration purposes..."
                    st.success("✅ Content extracted successfully!")
                    
                    # Display extracted content
                    with st.expander("📄 Extracted Content"):
                        st.text_area("Article Text", extracted_text, height=150, disabled=True)
                    
                    # Proceed with analysis
                    self.make_prediction(extracted_text, "", "", "", "", "")
                    
                except Exception as e:
                    st.error(f"Failed to extract content: {str(e)}")
            else:
                st.error("Please enter a valid URL")

    def file_upload_interface(self):
        """File upload interface"""
        st.markdown("### 📁 File Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload file(s) for analysis",
            type=['txt', 'csv', 'xlsx', 'json'],
            accept_multiple_files=True
        )
        
        if uploaded_file:
            for file in uploaded_file:
                st.markdown(f"**Processing:** {file.name}")
                
                try:
                    if file.type == "text/plain":
                        content = str(file.read(), "utf-8")
                        st.text_area(f"Content of {file.name}", content[:500] + "..." if len(content) > 500 else content)
                        
                    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                        df = pd.read_excel(file)
                        st.dataframe(df.head())
                        
                    elif file.type == "text/csv":
                        df = pd.read_csv(file)
                        st.dataframe(df.head())
                        
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

    def batch_processing_page(self):
        """Batch processing interface"""
        st.title("📊 Batch Processing")
        
        # File upload for batch processing
        uploaded_file = st.file_uploader(
            "Upload CSV file with news statements",
            type=['csv'],
            help="CSV should have columns: statement, speaker (optional), job (optional)"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                if st.button("🚀 Process Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, row in df.iterrows():
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1}/{len(df)}: {row['statement'][:50]}...")
                        
                        # Simulate prediction (replace with actual model)
                        result = {
                            'statement': row['statement'],
                            'prediction': np.random.choice(['real', 'fake']),
                            'confidence': np.random.uniform(0.6, 0.95),
                            'processing_time': np.random.uniform(0.1, 0.5)
                        }
                        results.append(result)
                        time.sleep(0.1)  # Simulate processing time
                    
                    self.session_state.batch_results = results
                    st.success(f"✅ Processed {len(results)} statements!")
                    self.display_batch_results(results)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def display_batch_results(self, results: List[Dict]):
        """Display batch processing results"""
        if not results:
            return
            
        df_results = pd.DataFrame(results)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", len(results))
        with col2:
            fake_count = len(df_results[df_results['prediction'] == 'fake'])
            st.metric("Predicted Fake", fake_count)
        with col3:
            avg_confidence = df_results['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        with col4:
            avg_time = df_results['processing_time'].mean()
            st.metric("Avg Time (s)", f"{avg_time:.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            pred_counts = df_results['prediction'].value_counts()
            fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index, 
                           title="Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_hist = px.histogram(df_results, x='confidence', nbins=20,
                                  title="Confidence Score Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detailed results
        st.markdown("### Detailed Results")
        st.dataframe(
            df_results[['statement', 'prediction', 'confidence']],
            use_container_width=True
        )
        
        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    def analytics_dashboard_page(self):
        """Analytics and monitoring dashboard"""
        st.title("📈 Analytics Dashboard")
        
        if not self.prediction_history:
            st.info("No predictions made yet. Make some predictions to see analytics!")
            return
        
        # Time-based metrics
        df_history = pd.DataFrame(self.prediction_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(df_history)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            fake_rate = len(df_history[df_history['prediction'] == 'fake']) / len(df_history) * 100
            st.metric("Fake News Rate", f"{fake_rate:.1f}%")
        
        with col3:
            avg_confidence = df_history['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            if 'processing_time' in df_history.columns:
                avg_time = df_history['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        # Time series analysis
        st.markdown("### Prediction Timeline")
        
        # Group by hour/day based on data volume
        time_grouping = 'H' if len(df_history) > 100 else 'D'
        time_series = df_history.set_index('timestamp').groupby(pd.Grouper(freq=time_grouping)).agg({
            'prediction': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_timeline.add_trace(
            go.Scatter(x=time_series['timestamp'], y=time_series['prediction'],
                      mode='lines+markers', name='Predictions Count'),
            secondary_y=False
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=time_series['timestamp'], y=time_series['confidence'],
                      mode='lines+markers', name='Avg Confidence', line=dict(color='orange')),
            secondary_y=True
        )
        
        fig_timeline.update_layout(title="Predictions Over Time")
        fig_timeline.update_xaxes(title_text="Time")
        fig_timeline.update_yaxes(title_text="Number of Predictions", secondary_y=False)
        fig_timeline.update_yaxes(title_text="Average Confidence", secondary_y=True)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution by prediction
            fig_conf = px.box(df_history, x='prediction', y='confidence',
                            title="Confidence Distribution by Prediction")
            st.plotly_chart(fig_conf, use_container_width=True)
        
        with col2:
            # Statement length analysis
            if 'statement_length' in df_history.columns:
                fig_length = px.scatter(df_history, x='statement_length', y='confidence',
                                      color='prediction', title="Confidence vs Statement Length")
                st.plotly_chart(fig_length, use_container_width=True)

    def make_prediction(self, statement: str, speaker: str = "", job: str = "", 
                       state: str = "", party: str = "", context: str = "",
                       confidence_threshold: float = 0.8, show_explanation: bool = True,
                       enable_uncertainty: bool = True):
        """Make prediction with enhanced features"""
        
        # Show processing indicator
        with st.spinner("🔄 Analyzing statement..."):
            start_time = time.time()
            
            # Simulate model prediction (replace with actual model)
            prediction = np.random.choice(['real', 'fake'], p=[0.6, 0.4])
            confidence = np.random.uniform(0.55, 0.95)
            processing_time = time.time() - start_time
            
            # Store prediction
            prediction_data = {
                'statement': statement,
                'speaker': speaker,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'statement_length': len(statement.split())
            }
            
            self.prediction_history.append(prediction_data)
            self.session_state.prediction_count += 1
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        # Main result
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if prediction == 'fake':
                st.error(f"🚨 **FAKE NEWS DETECTED**")
                st.markdown(f"**Confidence:** {confidence:.2%}")
            else:
                st.success(f"✅ **LIKELY REAL NEWS**")
                st.markdown(f"**Confidence:** {confidence:.2%}")
        
        with col2:
            # Confidence meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence"},
                gauge={'axis': {'range': [None, 1]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 0.6], 'color': "lightgray"},
                               {'range': [0.6, 0.8], 'color': "yellow"},
                               {'range': [0.8, 1], 'color': "green"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': confidence_threshold}}
            ))
            fig_gauge.update_layout(height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Words Analyzed", len(statement.split()))
        
        # Show explanation if requested
        if show_explanation:
            with st.expander("🔍 Prediction Explanation"):
                st.markdown("**Key factors influencing this prediction:**")
                
                # Simulate explanation features
                factors = [
                    ("Language complexity", np.random.uniform(0.3, 0.9)),
                    ("Emotional tone", np.random.uniform(0.2, 0.8)),
                    ("Source credibility", np.random.uniform(0.4, 0.9)),
                    ("Fact consistency", np.random.uniform(0.3, 0.9)),
                    ("Writing style", np.random.uniform(0.2, 0.7))
                ]
                
                for factor, score in factors:
                    st.markdown(f"- **{factor}:** {score:.2f}")
                
                # Word importance (simulated)
                st.markdown("**Important words/phrases:**")
                words = statement.split()[:10]  # Take first 10 words
                importance_scores = np.random.uniform(0.1, 1.0, len(words))
                
                for word, importance in zip(words, importance_scores):
                    color_intensity = int(255 * (1 - importance))
                    st.markdown(
                        f'<span style="background-color: rgba(255, {color_intensity}, {color_intensity}, 0.3); '
                        f'padding: 2px 4px; margin: 2px; border-radius: 3px;">{word}</span>',
                        unsafe_allow_html=True
                    )
        
        # Uncertainty analysis
        if enable_uncertainty:
            with st.expander("📊 Uncertainty Analysis"):
                st.markdown("**Prediction Uncertainty Metrics:**")
                
                # Simulate uncertainty metrics
                epistemic_uncertainty = np.random.uniform(0.05, 0.25)
                aleatoric_uncertainty = np.random.uniform(0.02, 0.15)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Uncertainty", f"{epistemic_uncertainty:.3f}")
                with col2:
                    st.metric("Data Uncertainty", f"{aleatoric_uncertainty:.3f}")
                
                if epistemic_uncertainty > 0.15:
                    st.warning("⚠️ High model uncertainty - consider getting additional opinions")
                if aleatoric_uncertainty > 0.10:
                    st.info("ℹ️ High data uncertainty - statement may be ambiguous")

    def run(self):
        """Main application runner"""
        # Configure page
        st.set_page_config(
            page_title="Fake News Detection System",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Authentication check
        if not self.session_state.authenticated:
            self.login_page()
            return
        
        # Load models
        if not self.models:
            self.load_models()
        
        # Navigation
        page = self.sidebar_navigation()
        
        # Page routing
        if page == "Single Prediction":
            self.single_prediction_page()
        elif page == "Batch Processing":
            self.batch_processing_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()
        elif page == "Model Comparison":
            st.title("🔄 Model Comparison")
            st.info("Model comparison features coming soon...")
        elif page == "Data Explorer":
            st.title("🔍 Data Explorer")
            st.info("Data exploration features coming soon...")
        elif page == "System Status":
            st.title("🖥️ System Status")
            st.info("System monitoring features coming soon...")

# Run the application
if __name__ == "__main__":
    app = FakeNewsDetectionApp()
    app.run()