import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.baseline_models_0149 import BaselineModels
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fake News Detection - Member 0149",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitApp:
    def __init__(self):
        self.models = {}
        self.label_mapping = {
            0: 'pants-fire', 1: 'false', 2: 'barely-true',
            3: 'half-true', 4: 'mostly-true', 5: 'true'
        }
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            with open('models/baseline/tfidf_logistic_0149.pkl', 'rb') as f:
                self.models['TF-IDF + Logistic Regression'] = pickle.load(f)
            
            with open('models/baseline/tfidf_rf_0149.pkl', 'rb') as f:
                self.models['TF-IDF + Random Forest'] = pickle.load(f)
                
            st.sidebar.success("✅ Models loaded successfully!")
        except FileNotFoundError:
            st.sidebar.error("❌ Models not found! Please train the models first.")
            st.sidebar.info("Run the baseline training notebook to generate models.")
    
    def predict_statement(self, statement, model_name):
        """Make prediction for a statement"""
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict([statement])[0]
        probability = model.predict_proba([statement])[0]
        
        # Get label and confidence
        predicted_label = self.label_mapping[prediction]
        confidence = max(probability)
        
        return predicted_label, confidence, probability
    
    def create_probability_chart(self, probabilities):
        """Create probability distribution chart"""
        labels = list(self.label_mapping.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=probabilities,
                marker_color=['red' if p == max(probabilities) else 'lightblue' for p in probabilities],
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Prediction Confidence Distribution",
            xaxis_title="Truth Labels",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def main(self):
        """Main Streamlit application"""
        
        # Title and header
        st.title("🔍 Fake News Detection System")
        st.subheader("Member 0149 - Baseline Models Demo")
        st.markdown("**ITBIN-2211-0149** | TF-IDF + Machine Learning Approach")
        
        # Sidebar
        st.sidebar.title("📊 Model Settings")
        
        # Model selection
        available_models = list(self.models.keys())
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Choose Model:",
                available_models,
                help="Select which baseline model to use for prediction"
            )
        else:
            st.error("No models available. Please train the models first.")
            return
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter Statement to Analyze")
            
            # Text input
            statement = st.text_area(
                "Statement:",
                height=150,
                placeholder="Enter a news statement or claim to analyze...",
                help="Enter the text you want to classify for truthfulness"
            )
            
            # Example statements
            st.markdown("**📋 Example Statements:**")
            example_statements = [
                "The economy has grown by 15% this quarter",
                "Climate change is a hoax created by scientists",
                "The new healthcare bill will reduce costs by half",
                "Vaccines contain microchips for tracking people",
                "The unemployment rate dropped to 3.5% last month"
            ]
            
            for i, example in enumerate(example_statements, 1):
                if st.button(f"Example {i}: {example[:50]}...", key=f"example_{i}"):
                    statement = example
                    st.rerun()
            
            # Predict button
            if st.button("🔍 Analyze Statement", type="primary"):
                if statement.strip():
                    with st.spinner("Analyzing statement..."):
                        result = self.predict_statement(statement, selected_model)
                        
                        if result[0] is not None:
                            predicted_label, confidence, probabilities = result
                            
                            # Results display
                            st.markdown("### 📊 Analysis Results")
                            
                            # Create columns for results
                            result_col1, result_col2, result_col3 = st.columns(3)
                            
                            with result_col1:
                                st.metric("Predicted Label", predicted_label.title())
                            
                            with result_col2:
                                st.metric("Confidence", f"{confidence:.1%}")
                            
                            with result_col3:
                                # Color code based on truthfulness
                                if predicted_label in ['true', 'mostly-true']:
                                    color = "🟢"
                                elif predicted_label in ['half-true']:
                                    color = "🟡"
                                else:
                                    color = "🔴"
                                st.metric("Reliability", f"{color} {predicted_label}")
                            
                            # Probability chart
                            st.plotly_chart(
                                self.create_probability_chart(probabilities),
                                use_container_width=True
                            )
                            
                            # Interpretation
                            st.markdown("### 💡 Interpretation")
                            if confidence > 0.7:
                                st.success(f"High confidence prediction: The statement is likely **{predicted_label}**")
                            elif confidence > 0.5:
                                st.warning(f"Moderate confidence: The statement appears to be **{predicted_label}**")
                            else:
                                st.info("Low confidence prediction. Manual verification recommended.")
                        
                        else:
                            st.error("Error making prediction. Please try again.")
                else:
                    st.warning("Please enter a statement to analyze.")
        
        with col2:
            st.markdown("### ℹ️ About This Demo")
            st.info(
                """
                This demo showcases baseline models developed by **Member 0149** 
                for the Fake News Detection project.
                
                **Models Used:**
                - TF-IDF + Logistic Regression
                - TF-IDF + Random Forest
                
                **Features:**
                - Text preprocessing
                - N-gram analysis (1-2 grams)
                - Class balancing
                - Hyperparameter optimization
                """
            )
            
            # Model performance (if available)
            try:
                performance_df = pd.read_csv('results/reports/model_comparison_0149.csv')
                st.markdown("### 📈 Model Performance")
                st.dataframe(performance_df.round(3))
            except FileNotFoundError:
                st.markdown("### 📈 Model Performance")
                st.info("Performance data will be available after training completion.")
            
            # Classification explanation
            st.markdown("### 🎯 Classification Scale")
            scale_data = {
                "Label": ["True", "Mostly-True", "Half-True", "Barely-True", "False", "Pants-Fire"],
                "Description": [
                    "Factually accurate",
                    "Largely accurate with minor issues",
                    "Partially accurate",
                    "Contains some truth but misleading", 
                    "Factually incorrect",
                    "Ridiculously false"
                ],
                "Color": ["🟢", "🟢", "🟡", "🟠", "🔴", "🔴"]
            }
            scale_df = pd.DataFrame(scale_data)
            st.dataframe(scale_df, hide_index=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p><strong>Developed by Member 0149 (ITBIN-2211-0149)</strong></p>
                <p>Fake News Detection Using Hybrid NLP Approach - Baseline Models Component</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    app = StreamlitApp()
    app.main()