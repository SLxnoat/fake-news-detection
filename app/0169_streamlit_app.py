import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from deployment.inference_pipeline import InferencePipeline

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialized = False

# Sidebar for navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Home", "📝 Single Prediction", "📊 Batch Analysis", "📈 Performance Monitor", "ℹ️ Model Info"]
)

def initialize_pipeline():
    """Initialize the inference pipeline"""
    if not st.session_state.initialized:
        with st.spinner("Initializing models..."):
            try:
                pipeline = InferencePipeline()
                init_result = pipeline.initialize()
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                return init_result
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {str(e)}")
                return None
    return {"status": "already_initialized"}

def home_page():
    """Home page content"""
    st.title("📰 Fake News Detection System")
    st.markdown("---")
    
    st.markdown("""
    ## Welcome to the Fake News Detection System
    
    This system uses advanced Natural Language Processing techniques to detect fake news by analyzing:
    - **Article Content**: Using TF-IDF and BERT embeddings
    - **Metadata Features**: Speaker credibility, party affiliation, subject category
    - **Hybrid Models**: Combining traditional and deep learning approaches
    
    ### 🎯 Truth Categories
    - **True**: The statement is accurate
    - **Mostly True**: The statement is accurate but needs clarification
    - **Half True**: The statement is partially accurate
    - **Barely True**: The statement contains some truth but ignores important facts
    - **False**: The statement is not accurate
    - **Pants Fire**: The statement is not accurate and makes a ridiculous claim
    """)
    
    # Initialize pipeline
    init_result = initialize_pipeline()
    
    if init_result:
        if init_result.get('models_loaded', 0) > 0:
            st.success(f"✅ System ready! {init_result['models_loaded']} models loaded")
        else:
            st.warning("⚠️ No models loaded. Some features may not work.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Loaded", init_result.get('models_loaded', 0))
        with col2:
            st.metric("Preprocessors", init_result.get('preprocessors_loaded', 0))
        with col3:
            health_status = init_result.get('health_status', 'unknown')
            st.metric("Health Status", health_status)

def single_prediction_page():
    """Single prediction page"""
    st.title("📝 Single Statement Analysis")
    st.markdown("---")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the Home page first.")
        return
    
    # Input form
    with st.form("prediction_form"):
        statement = st.text_area(
            "Enter the statement to analyze:",
            placeholder="e.g., The unemployment rate has decreased by 10% this year.",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            speaker = st.text_input("Speaker (optional)", placeholder="e.g., John Doe")
            subject = st.text_input("Subject (optional)", placeholder="e.g., economy")
        
        with col2:
            party = st.selectbox("Party Affiliation (optional)", 
                               ["", "democrat", "republican", "independent", "none"])
            use_ensemble = st.checkbox("Use ensemble prediction", value=True)
        
        submitted = st.form_submit_button("🔍 Analyze Statement")
    
    if submitted and statement.strip():
        with st.spinner("Analyzing statement..."):
            try:
                result = st.session_state.pipeline.predict(
                    statement, speaker, party, subject, use_ensemble
                )
                
                if 'error' in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    # Display results
                    if use_ensemble and 'ensemble_prediction' in result:
                        prediction = result['ensemble_prediction']
                        confidence = result['ensemble_confidence']
                        st.subheader("🎯 Ensemble Prediction")
                    else:
                        prediction = result['prediction']
                        confidence = result['confidence']
                        st.subheader("🎯 Model Prediction")
                    
                    # Main result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Truth Classification", prediction.title())
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.3f}")
                    
                    # Confidence bar
                    fig = go.Figure(go.Bar(
                        x=[confidence],
                        y=['Confidence'],
                        orientation='h',
                        marker_color='green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
                    ))
                    fig.update_layout(
                        xaxis_range=[0, 1],
                        height=150,
                        title="Confidence Level"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Individual model results (if ensemble)
                    if use_ensemble and 'individual_predictions' in result:
                        st.subheader("📊 Individual Model Results")
                        
                        individual_data = []
                        for model_name, model_result in result['individual_predictions'].items():
                            individual_data.append({
                                'Model': model_name,
                                'Prediction': model_result['prediction'],
                                'Confidence': model_result['confidence']
                            })
                        
                        df = pd.DataFrame(individual_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualization
                        fig = px.bar(df, x='Model', y='Confidence', 
                                   color='Prediction', title='Model Predictions Comparison')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance info
                    if 'performance' in result:
                        st.info(f"Response time: {result['performance']['response_time']:.3f} seconds")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def batch_analysis_page():
    """Batch analysis page"""
    st.title("📊 Batch Statement Analysis")
    st.markdown("---")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the Home page first.")
        return
    
    st.markdown("Upload a CSV file with statements for batch analysis.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Column mapping
            st.subheader("🔧 Column Mapping")
            columns = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                statement_col = st.selectbox("Statement column", columns)
                speaker_col = st.selectbox("Speaker column (optional)", ["None"] + columns)
            
            with col2:
                party_col = st.selectbox("Party column (optional)", ["None"] + columns)
                subject_col = st.selectbox("Subject column (optional)", ["None"] + columns)
            
            use_ensemble = st.checkbox("Use ensemble prediction", value=True, key="batch_ensemble")
            
            if st.button("🚀 Start Batch Analysis"):
                # Prepare data
                statements = []
                for _, row in df.iterrows():
                    item = {
                        'statement': str(row[statement_col]) if statement_col in columns else '',
                        'speaker': str(row[speaker_col]) if speaker_col != "None" and speaker_col in columns else '',
                        'party': str(row[party_col]) if party_col != "None" and party_col in columns else '',
                        'subject': str(row[subject_col]) if subject_col != "None" and subject_col in columns else ''
                    }
                    statements.append(item)
                
                # Perform batch prediction
                with st.spinner(f"Analyzing {len(statements)} statements..."):
                    results = st.session_state.pipeline.batch_predict(statements, use_ensemble)
                
                # Process results
                predictions = []
                confidences = []
                errors = []
                
                for result in results:
                    if 'error' in result:
                        predictions.append('error')
                        confidences.append(0.0)
                        errors.append(result['error'])
                    else:
                        if use_ensemble and 'ensemble_prediction' in result:
                            predictions.append(result['ensemble_prediction'])
                            confidences.append(result['ensemble_confidence'])
                        else:
                            predictions.append(result['prediction'])
                            confidences.append(result['confidence'])
                        errors.append('')
                
                # Add results to dataframe
                df['prediction'] = predictions
                df['confidence'] = confidences
                df['error'] = errors
                
                # Display results
                st.subheader("📈 Analysis Results")
                st.dataframe(df)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Statements", len(df))
                with col2:
                    successful = len(df[df['prediction'] != 'error'])
                    st.metric("Successful", successful)
                with col3:
                    failed = len(df[df['prediction'] == 'error'])
                    st.metric("Failed", failed)
                with col4:
                    avg_confidence = df[df['prediction'] != 'error']['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                # Visualizations
                valid_df = df[df['prediction'] != 'error']
                
                if len(valid_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        pred_counts = valid_df['prediction'].value_counts()
                        fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                   title="Prediction Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Confidence distribution
                        fig = px.histogram(valid_df, x='confidence', bins=20,
                                         title="Confidence Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def performance_monitor_page():
    """Performance monitoring page"""
    st.title("📈 Performance Monitor")
    st.markdown("---")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the Home page first.")
        return
    
    # Get performance stats
    stats = st.session_state.pipeline.get_performance_stats()
    
    # Display metrics
    st.subheader("📊 Current Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    with col2:
        st.metric("Success Rate", f"{stats['success_rate']:.3f}")
    with col3:
        st.metric("Average Response Time", f"{stats['average_response_time']:.3f}s")
    with col4:
        st.metric("Response Times Recorded", stats['response_times_count'])
    
    if stats['total_predictions'] > 0:
        # Response time percentiles
        st.subheader("⚡ Response Time Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("P50 (Median)", f"{stats.get('response_time_p50', 0):.3f}s")
        with col2:
            st.metric("P95", f"{stats.get('response_time_p95', 0):.3f}s")
        with col3:
            st.metric("P99", f"{stats.get('response_time_p99', 0):.3f}s")
        
        # Success/Failure breakdown
        col1, col2 = st.columns(2)
        with col1:
            success_data = pd.DataFrame({
                'Status': ['Success', 'Failure'],
                'Count': [stats['successful_predictions'], stats['failed_predictions']]
            })
            fig = px.pie(success_data, values='Count', names='Status',
                        title="Success/Failure Rate")
            st.plotly_chart(fig, use_container_width=True)
    
    # System health
    st.subheader("🏥 System Health")
    health = st.session_state.pipeline.deployment.health_check()
    
    if health['status'] == 'healthy':
        st.success("✅ All systems operational")
    elif health['status'] == 'degraded':
        st.warning("⚠️ System degraded - some models may be unavailable")
    else:
        st.error("❌ System unhealthy")
    
    # Model health details
    for model_name, model_status in health['models_status'].items():
        if model_status == 'healthy':
            st.success(f"✅ {model_name}: Healthy")
        else:
            st.error(f"❌ {model_name}: {model_status}")
    
    # Reset stats button
    if st.button("🔄 Reset Performance Statistics"):
        st.session_state.pipeline.reset_stats()
        st.success("Performance statistics reset!")
        st.experimental_rerun()

def model_info_page():
    """Model information page"""
    st.title("ℹ️ Model Information")
    st.markdown("---")
    
    if not st.session_state.initialized:
        st.warning("Please initialize the system from the Home page first.")
        return
    
    # Get model info
    model_info = st.session_state.pipeline.deployment.get_model_info()
    
    st.subheader("📋 System Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Loaded Models", model_info['loaded_models'])
    with col2:
        st.metric("Loaded Preprocessors", model_info['loaded_preprocessors'])
    
    if model_info['models']:
        st.subheader("🤖 Model Details")
        
        for model_name, details in model_info['models'].items():
            with st.expander(f"📊 {model_name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {details['type']}")
                    st.write(f"**Loaded At:** {details['loaded_at']}")
                with col2:
                    st.write(f"**Path:** {details['path']}")
                
                # Test prediction button
                if st.button(f"Test {model_name}", key=f"test_{model_name}"):
                    test_statement = "This is a test statement for model verification."
                    with st.spinner(f"Testing {model_name}..."):
                        result = st.session_state.pipeline.deployment.predict_single(
                            model_name, test_statement
                        )
                        if 'error' in result:
                            st.error(f"Test failed: {result['error']}")
                        else:
                            st.success(f"✅ Test successful!")
                            st.json(result)
    
    # Technical specifications
    st.subheader("🔧 Technical Specifications")
    st.markdown("""
    **Model Architecture:**
    - **Baseline Models**: TF-IDF + Logistic Regression, TF-IDF + Random Forest
    - **Hybrid Model**: BERT embeddings + TF-IDF features + Metadata features
    - **Ensemble Method**: Majority voting with confidence weighting
    
    **Features:**
    - Text content analysis using TF-IDF vectorization
    - Semantic understanding through BERT embeddings
    - Metadata integration (speaker, party, subject)
    - Real-time inference with performance monitoring
    
    **Performance Targets:**
    - Accuracy: ≥ 80%
    - Response Time: < 2 seconds
    - Availability: > 99%
    """)

# Main app logic
if __name__ == "__main__":
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📝 Single Prediction":
        single_prediction_page()
    elif page == "📊 Batch Analysis":
        batch_analysis_page()
    elif page == "📈 Performance Monitor":
        performance_monitor_page()
    elif page == "ℹ️ Model Info":
        model_info_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fake News Detection System v1.0**")
    st.sidebar.markdown("Member 0169 - Model Evaluation & Deployment")