# Advanced Visualizations and Data Profiling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedEDAAnalyzer:
    def __init__(self, data_path):
        # Verify file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
            
        self.df = pd.read_csv(data_path, sep='\t', header=0)
        self.setup_data()
    
    def setup_data(self):
        """Prepare data for analysis"""
        # Define column names based on LIAR dataset structure
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
                  'state_info', 'party_affiliation', 'barely_true_counts', 
                  'false_counts', 'half_true_counts', 'mostly_true_counts', 
                  'pants_fire_counts', 'context']
        
        # Rename columns if they don't match expected names
        if len(self.df.columns) >= len(columns):
            self.df.columns = columns + list(self.df.columns[len(columns):])
        
        # Create derived features
        self.df['text_length'] = self.df['statement'].str.len()
        self.df['word_count'] = self.df['statement'].str.split().str.len()
        
        # Calculate credibility scores
        credibility_cols = ['barely_true_counts', 'false_counts', 'half_true_counts',
                          'mostly_true_counts', 'pants_fire_counts']
        self.df['total_statements'] = self.df[credibility_cols].sum(axis=1)
        self.df['credibility_score'] = (
            (self.df['mostly_true_counts'] * 1.0 + 
             self.df['half_true_counts'] * 0.5 + 
             self.df['barely_true_counts'] * 0.25) / 
            (self.df['total_statements'] + 1e-5)  # Avoid division by zero
        )
    
    def create_interactive_dashboard(self):
        """Create comprehensive interactive dashboard"""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Label Distribution', 'Text Length by Label',
                          'Credibility Score Distribution', 'Party vs Label Analysis',
                          'Subject Category Analysis', 'Credibility vs Text Length'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Label Distribution
        label_counts = self.df['label'].value_counts()
        fig.add_trace(
            go.Bar(x=label_counts.index, y=label_counts.values, 
                   name='Label Count', showlegend=False),
            row=1, col=1
        )
        
        # 2. Text Length by Label - Box Plot
        for label in self.df['label'].unique():
            data = self.df[self.df['label'] == label]['text_length']
            fig.add_trace(
                go.Box(y=data, name=label, showlegend=False),
                row=1, col=2
            )
        
        # 3. Credibility Score Distribution
        fig.add_trace(
            go.Histogram(x=self.df['credibility_score'], name='Credibility', 
                        showlegend=False, nbinsx=30),
            row=2, col=1
        )
        
        # 4. Party vs Label Heatmap
        party_label_crosstab = pd.crosstab(
            self.df['party_affiliation'], self.df['label'], normalize='index'
        )
        fig.add_trace(
            go.Heatmap(z=party_label_crosstab.values,
                      x=party_label_crosstab.columns,
                      y=party_label_crosstab.index,
                      colorscale='Blues', showscale=True,
                      colorbar=dict(title='Proportion')),
            row=2, col=2
        )
        
        # 5. Subject Category Analysis
        subject_counts = self.df['subject'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=subject_counts.values, y=subject_counts.index,
                   orientation='h', name='Subject Count', showlegend=False),
            row=3, col=1
        )
        
        # 6. Credibility vs Text Length
        fig.add_trace(
            go.Scatter(x=self.df['text_length'], y=self.df['credibility_score'],
                      mode='markers', name='Correlation', showlegend=False,
                      marker=dict(opacity=0.6, size=5)),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200, width=1400,
            title_text="Fake News Dataset - Comprehensive Analysis Dashboard",
            title_x=0.5
        )
        
        # Save and show
        os.makedirs("results/figures", exist_ok=True)
        fig.write_html("results/figures/interactive_dashboard.html")
        fig.show()
        
        return fig
    
    def create_word_clouds(self):
        """Generate word clouds for different truth labels"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        labels = self.df['label'].unique()[:6]
        
        for i, label in enumerate(labels):
            # Get statements for this label
            statements = self.df[self.df['label'] == label]['statement']
            text = ' '.join(statements.dropna().astype(str))
            
            if text.strip():
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    max_words=100).generate(text)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{label} Statements')
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'No data for {label}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{label} Statements')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/figures/word_clouds_by_label.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_text_complexity(self):
        """Analyze text complexity patterns by label"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Text length distribution by label
        for label in self.df['label'].unique():
            data = self.df[self.df['label'] == label]['text_length']
            axes[0,0].hist(data, alpha=0.6, label=label, bins=30)
        axes[0,0].set_xlabel('Text Length (characters)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Text Length Distribution by Label')
        axes[0,0].legend()
        
        # 2. Word count distribution by label
        for label in self.df['label'].unique():
            data = self.df[self.df['label'] == label]['word_count']
            axes[0,1].hist(data, alpha=0.6, label=label, bins=30)
        axes[0,1].set_xlabel('Word Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Word Count Distribution by Label')
        axes[0,1].legend()
        
        # 3. Credibility score by label
        for label in self.df['label'].unique():
            data = self.df[self.df['label'] == label]['credibility_score']
            axes[1,0].hist(data, alpha=0.6, label=label, bins=30)
        axes[1,0].set_xlabel('Credibility Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Credibility Score Distribution by Label')
        axes[1,0].legend()
        
        # 4. Text length vs credibility scatter
        axes[1,1].scatter(self.df['text_length'], self.df['credibility_score'], 
                          alpha=0.5, c=pd.Categorical(self.df['label']).codes, cmap='viridis')
        axes[1,1].set_xlabel('Text Length')
        axes[1,1].set_ylabel('Credibility Score')
        axes[1,1].set_title('Text Length vs Credibility Score')
        
        plt.tight_layout()
        plt.savefig('results/figures/text_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_political_bias(self):
        """Analyze political bias patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Party affiliation distribution
        party_counts = self.df['party_affiliation'].value_counts()
        axes[0,0].pie(party_counts.values, labels=party_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Party Affiliation Distribution')
        
        # 2. Label distribution by party
        party_label_crosstab = pd.crosstab(self.df['party_affiliation'], self.df['label'])
        party_label_crosstab.plot(kind='bar', ax=axes[0,1], stacked=True)
        axes[0,1].set_title('Label Distribution by Party')
        axes[0,1].set_xlabel('Party')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Average credibility by party
        party_credibility = self.df.groupby('party_affiliation')['credibility_score'].mean().sort_values(ascending=False)
        axes[1,0].bar(party_credibility.index, party_credibility.values)
        axes[1,0].set_title('Average Credibility Score by Party')
        axes[1,0].set_xlabel('Party')
        axes[1,0].set_ylabel('Average Credibility Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Subject distribution by party
        subject_party_crosstab = pd.crosstab(self.df['subject'], self.df['party_affiliation'])
        subject_party_crosstab.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Subject Distribution by Party')
        axes[1,1].set_xlabel('Subject')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/figures/political_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_day3_results(self):
        """Save all day 3 analysis results"""
        os.makedirs('results', exist_ok=True)
        
        # Save processed dataset
        self.df.to_csv('results/processed_liar_dataset.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_statements': len(self.df),
            'label_distribution': self.df['label'].value_counts().to_dict(),
            'party_distribution': self.df['party_affiliation'].value_counts().to_dict(),
            'subject_distribution': self.df['subject'].value_counts().head(10).to_dict(),
            'text_length_stats': {
                'mean': self.df['text_length'].mean(),
                'std': self.df['text_length'].std(),
                'min': self.df['text_length'].min(),
                'max': self.df['text_length'].max()
            },
            'credibility_score_stats': {
                'mean': self.df['credibility_score'].mean(),
                'std': self.df['credibility_score'].std(),
                'min': self.df['credibility_score'].min(),
                'max': self.df['credibility_score'].max()
            }
        }
        
        with open('results/day3_analysis_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print("Day 3 results saved:")
        print(f"- results/processed_liar_dataset.csv (processed data)")
        print(f"- results/day3_analysis_summary.json (summary statistics)")
        print(f"- results/figures/ (all visualizations)")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize analyzer with your data path
        analyzer = AdvancedEDAAnalyzer('data/raw/train.tsv')
        
        # Create all visualizations
        analyzer.create_interactive_dashboard()
        analyzer.create_word_clouds()
        analyzer.analyze_text_complexity()
        analyzer.analyze_political_bias()
        
        # Save results
        analyzer.save_day3_results()
        
        print("✅ Advanced EDA completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your data path and ensure all required packages are installed.")
