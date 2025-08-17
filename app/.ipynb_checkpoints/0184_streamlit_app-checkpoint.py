import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page configuration
st.set_page_config(
    page_title="Fake News Detection - EDA Dashboard", 
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Fake News Detection: Exploratory Data Analysis Dashboard")
st.markdown("**By Member 0184 (ITBIN-2211-0184) - EDA & Documentation Specialist**")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose Analysis", 
    ["Dataset Overview", "Text Analysis", "Speaker Credibility", "Political Bias", "Key Insights"])

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('results/processed_liar_dataset.csv')

@st.cache_data
def load_profile():
    with open('results/reports/data_profile.json', 'r') as f:
        return json.load(f)

df = load_data()
profile = load_profile()

if page == "Dataset Overview":
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Statements", f"{len(df):,}")
    with col2:
        st.metric("Unique Speakers", f"{df['speaker'].nunique():,}")
    with col3:
        st.metric("Political Parties", f"{df['party_affiliation'].nunique()}")
    with col4:
        st.metric("Subject Categories", f"{df['subject'].nunique()}")
    
    # Label distribution
    st.subheader("Truth Label Distribution")
    label_counts = df['label'].value_counts()
    
    fig = px.bar(x=label_counts.index, y=label_counts.values, 
                 color=label_counts.values,
                 color_continuous_scale="viridis")
    fig.update_layout(title="Distribution of Truth Labels")
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset splits
    st.subheader("Dataset Composition")
    split_counts = df['dataset'].value_counts()
    
    fig = px.pie(values=split_counts.values, names=split_counts.index,
                 title="Train/Test/Validation Split")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Text Analysis":
    st.header("📝 Text Complexity Analysis")
    
    # Text metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Count by Truth Label")
        fig = px.box(df, x='label', y='word_count', 
                     color='label', title="Word Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Text Length by Truth Label")
        fig = px.box(df, x='label', y='text_length',
                     color='label', title="Character Count Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced metrics
    st.subheader("Advanced Linguistic Features")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x='label', y='avg_word_length',
                     title="Average Word Length by Label")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='label', y='capital_ratio',
                     title="Capital Letters Ratio by Label")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Speaker Credibility":
    st.header("🎤 Speaker Credibility Analysis")
    
    # Top speakers
    top_speakers = df.groupby('speaker').agg({
        'total_statements': 'first',
        'true_ratio': 'first',
        'statement': 'count'
    }).sort_values('statement', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Active Speakers")
        fig = px.bar(x=top_speakers.index[:10], y=top_speakers['statement'][:10],
                     title="Top 10 Speakers by Statement Count")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Credibility Score Distribution")
        fig = px.histogram(df, x='true_ratio', nbins=30,
                          title="Distribution of Speaker True Ratio")
        st.plotly_chart(fig, use_container_width=True)
    
    # Credibility vs Activity
    st.subheader("Speaker Credibility vs Activity")
    speaker_stats = df.groupby('speaker').agg({
        'true_ratio': 'first',
        'statement': 'count'
    }).reset_index()
    
    fig = px.scatter(speaker_stats, x='statement', y='true_ratio',
                     hover_data=['speaker'], title="Credibility vs Statement Count",
                     labels={'statement': 'Total Statements', 'true_ratio': 'True Ratio'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Political Bias":
    st.header("🏛️ Political Bias Analysis")
    
    # Party-label correlation
    party_label_crosstab = pd.crosstab(df['party_affiliation'], df['label'], normalize='index')
    
    # Filter major parties
    party_counts = df['party_affiliation'].value_counts()
    major_parties = party_counts[party_counts >= 50].index
    filtered_crosstab = party_label_crosstab.loc[major_parties]
    
    st.subheader("Truth Label Distribution by Political Party")
    
    fig = px.imshow(filtered_crosstab.values,
                    labels=dict(x="Truth Label", y="Political Party", color="Proportion"),
                    x=filtered_crosstab.columns,
                    y=filtered_crosstab.index,
                    color_continuous_scale="RdYlBu_r",
                    title="Party Bias Heatmap (Normalized)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Subject analysis
    st.subheader("Statement Topics by Party")
    subject_party = pd.crosstab(df['subject'], df['party_affiliation'])
    top_subjects = df['subject'].value_counts().head(8).index
    subject_party_filtered = subject_party.loc[top_subjects, major_parties]
    
    fig = px.imshow(subject_party_filtered.values,
                    labels=dict(x="Political Party", y="Subject", color="Count"),
                    x=subject_party_filtered.columns,
                    y=subject_party_filtered.index,
                    color_continuous_scale="Blues",
                    title="Subject Distribution by Party")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Key Insights":
    st.header("💡 Key Insights & Recommendations")
    
    st.subheader("🔍 Major Findings")
    
    insights = [
        {
            "title": "Class Imbalance Challenge",
            "description": "The dataset shows significant class imbalance with 'false' and 'barely-true' statements dominating.",
            "recommendation": "Implement class weighting or SMOTE for balanced model training."
        },
        {
            "title": "Linguistic Patterns",
            "description": "False statements tend to be shorter and use simpler language patterns.",
            "recommendation": "Include text complexity features in the model pipeline."
        },
        {
            "title": "Speaker Reliability",
            "description": "Strong correlation between speaker history and statement veracity.",
            "recommendation": "Incorporate speaker credibility scores as key features."
        },
        {
            "title": "Political Bias",
            "description": "Clear partisan patterns in statement accuracy across different topics.",
            "recommendation": "Consider political context in model architecture and evaluation."
        }
    ]
    
    for insight in insights:
        with st.expander(f"📊 {insight['title']}"):
            st.write(f"**Finding:** {insight['description']}")
            st.write(f"**Recommendation:** {insight['recommendation']}")
    
    st.subheader("📈 Model Development Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Baseline Models:**
        - Use TF-IDF with n-grams (1,2,3)
        - Include speaker credibility metrics
        - Apply balanced class weights
        - Cross-validate with stratified splits
        """)
    
    with col2:
        st.markdown("""
        **For Advanced Models:**
        - Fine-tune BERT on political text
        - Implement ensemble methods
        - Add attention mechanisms
        - Include uncertainty quantification
        """)
    
    st.subheader("📋 Deliverables Completed")
    
    deliverables = {
        "Data Analysis": ["✅ Comprehensive EDA", "✅ Statistical validation", "✅ Data profiling"],
        "Visualizations": ["✅ Interactive dashboards", "✅ Statistical plots", "✅ Correlation matrices"],
        "Documentation": ["✅ Technical reports", "✅ Code documentation", "✅ Methodology guide"],
        "Data Processing": ["✅ Feature engineering", "✅ Data cleaning", "✅ Quality assurance"]
    }
    
    for category, items in deliverables.items():
        st.write(f"**{category}:**")
        for item in items:
            st.write(f"  {item}")
        st.write("")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Contact:** ITBIN-2211-0184")
    st.sidebar.markdown("**Role:** EDA & Documentation")
    st.sidebar.markdown(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")