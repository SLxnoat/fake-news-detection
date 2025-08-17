import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class AdvancedUIComponents:
    """Advanced UI components for the fake news detection system"""
    
    @staticmethod
    def create_prediction_card(prediction_result):
        """Create an advanced prediction result card"""
        
        # Determine styling based on prediction
        if prediction_result['prediction'] == 'fake':
            border_color = "#ff4444"
            bg_color = "#fff5f5"
            icon = "🚨"
            title = "FAKE NEWS DETECTED"
        else:
            border_color = "#44ff44"
            bg_color = "#f5fff5"
            icon = "✅"
            title = "LIKELY REAL NEWS"
        
        # Custom CSS for the card
        card_css = f"""
        <style>
        .prediction-card {{
            border: 3px solid {border_color};
            border-radius: 15px;
            padding: 20px;
            background-color: {bg_color};
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .prediction-title {{
            font-size: 24px;
            font-weight: bold;
            color: {border_color};
            text-align: center;
            margin-bottom: 15px;
        }}
        .confidence-bar {{
            width: 100%;
            height: 25px;
            background-color: #f0f0f0;
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff4444 0%, #ffaa44 50%, #44ff44 100%);
            transition: width 0.5s ease;
        }}
        </style>
        """
        
        st.markdown(card_css, unsafe_allow_html=True)
        
        # Card content
        confidence_pct = prediction_result['confidence'] * 100
        
        card_html = f"""
        <div class="prediction-card">
            <div class="prediction-title">{icon} {title}</div>
            <div style="text-align: center;">
                <h3>Confidence: {confidence_pct:.1f}%</h3>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_pct}%;"></div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        return True
    
    @staticmethod
    def create_model_comparison_chart(model_results):
        """Create interactive model comparison chart"""
        
        models = list(model_results.keys())
        predictions = [model_results[model]['prediction'] for model in models]
        confidences = [model_results[model]['confidence'] for model in models]
        
        # Convert predictions to numeric for plotting
        pred_numeric = [1 if p == 'fake' else 0 for p in predictions]
        
        fig = go.Figure()
        
        # Add confidence bars
        fig.add_trace(go.Bar(
            x=models,
            y=confidences,
            name='Confidence',
            marker_color=['red' if p == 1 else 'green' for p in pred_numeric],
            text=[f'{c:.2f}' for c in confidences],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Model Comparison - Confidence Scores",
            xaxis_title="Models",
            yaxis_title="Confidence",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_plot(feature_data):
        """Create feature importance visualization"""
        
        # Sample feature importance data
        features = ['Word Count', 'Sentiment', 'Readability', 'Entity Count', 
                   'Capital Ratio', 'Question Marks', 'Exclamations']
        importance = np.random.uniform(0.1, 1.0, len(features))
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='skyblue',
            text=[f'{imp:.3f}' for imp in importance],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Importance Score",
            height=400,
            margin=dict(l=150)
        )
        
        return fig
    
    @staticmethod
    def create_confidence_gauge(confidence_score):
        """Create a confidence gauge meter"""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def create_timeline_analysis(prediction_history):
        """Create timeline analysis of predictions"""
        
        if not prediction_history:
            return None
        
        df = pd.DataFrame(prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by hour
        hourly_data = df.set_index('timestamp').resample('H').agg({
            'prediction': lambda x: (x == 'fake').sum(),
            'confidence': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=hourly_data['timestamp'],
                y=hourly_data['prediction'],
                mode='lines+markers',
                name='Fake Predictions',
                line=dict(color='red')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=hourly_data['timestamp'],
                y=hourly_data['confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='blue')
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Fake Predictions Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Confidence", secondary_y=True)
        
        fig.update_layout(
            title_text="Prediction Timeline Analysis",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_batch_results_summary(batch_results):
        """Create comprehensive batch results visualization"""
        
        df = pd.DataFrame(batch_results)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Distribution', 'Confidence Histogram',
                          'Processing Time', 'Confidence vs Time'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Pie chart for predictions
        pred_counts = df['prediction'].value_counts()
        fig.add_trace(
            go.Pie(labels=pred_counts.index, values=pred_counts.values),
            row=1, col=1
        )
        
        # Confidence histogram
        fig.add_trace(
            go.Histogram(x=df['confidence'], nbinsx=20),
            row=1, col=2
        )
        
        # Processing time scatter
        fig.add_trace(
            go.Scatter(x=df.index, y=df['processing_time'],
                      mode='markers', name='Processing Time'),
            row=2, col=1
        )
        
        # Confidence vs time
        fig.add_trace(
            go.Scatter(x=df.index, y=df['confidence'],
                      mode='markers', name='Confidence'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Batch Processing Results")
        return fig
    
    @staticmethod
    def create_system_status_dashboard():
        """Create system status dashboard"""
        
        # Simulate system metrics
        metrics = {
            'cpu_usage': np.random.uniform(20, 80),
            'memory_usage': np.random.uniform(30, 70),
            'disk_usage': np.random.uniform(10, 50),
            'network_io': np.random.uniform(5, 95),
            'api_requests': np.random.randint(50, 200),
            'cache_hit_rate': np.random.uniform(0.7, 0.95)
        }
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage',
                          'Network I/O', 'API Requests', 'Cache Hit Rate'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # CPU Usage
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics['cpu_usage'],
            title = {'text': "CPU %"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 85], 'color': "yellow"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ), row=1, col=1)
        
        # Memory Usage
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics['memory_usage'],
            title = {'text': "Memory %"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"}}
        ), row=1, col=2)
        
        # Disk Usage
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics['disk_usage'],
            title = {'text': "Disk %"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"}}
        ), row=1, col=3)
        
        # Network I/O
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = metrics['network_io'],
            title = {'text': "Network I/O MB/s"},
            delta = {'reference': 50}
        ), row=2, col=1)
        
        # API Requests
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = metrics['api_requests'],
            title = {'text': "API Requests/hr"},
            delta = {'reference': 100}
        ), row=2, col=2)
        
        # Cache Hit Rate
        fig.add_trace(go.Indicator(
            mode = "number+gauge",
            value = metrics['cache_hit_rate'],
            title = {'text': "Cache Hit Rate"},
            number = {'suffix': "%", 'valueformat': ".1%"},
            gauge = {'axis': {'range': [None, 1]},
                    'bar': {'color': "purple"}}
        ), row=2, col=3)
        
        fig.update_layout(height=600, title_text="System Status Dashboard")
        return fig

# Example usage in Streamlit app
def demo_advanced_ui():
    """Demonstrate advanced UI components"""
    
    st.title("🎨 Advanced UI Components Demo")
    
    ui = AdvancedUIComponents()
    
    # Demo prediction card
    st.subheader("1. Enhanced Prediction Card")
    sample_result = {
        'prediction': 'fake',
        'confidence': 0.87
    }
    ui.create_prediction_card(sample_result)
    
    # Demo model comparison
    st.subheader("2. Model Comparison Chart")
    model_results = {
        'Logistic Regression': {'prediction': 'fake', 'confidence': 0.85},
        'Random Forest': {'prediction': 'fake', 'confidence': 0.78},
        'Neural Network': {'prediction': 'real', 'confidence': 0.65},
        'Ensemble': {'prediction': 'fake', 'confidence': 0.82}
    }
    comparison_chart = ui.create_model_comparison_chart(model_results)
    st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Demo confidence gauge
    st.subheader("3. Confidence Gauge")
    col1, col2 = st.columns(2)
    with col1:
        gauge = ui.create_confidence_gauge(0.87)
        st.plotly_chart(gauge, use_container_width=True)
    
    with col2:
        # Demo feature importance
        st.subheader("4. Feature Importance")
        feature_plot = ui.create_feature_importance_plot({})
        st.plotly_chart(feature_plot, use_container_width=True)
    
    # Demo system status
    st.subheader("5. System Status Dashboard")
    status_dashboard = ui.create_system_status_dashboard()
    st.plotly_chart(status_dashboard, use_container_width=True)

if __name__ == "__main__":
    demo_advanced_ui()