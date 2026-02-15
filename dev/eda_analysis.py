"""
Social Media Sentiment Analysis - Exploratory Data Analysis (EDA)
Generate word clouds and frequency distribution plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class SentimentEDA:
    """
    Exploratory Data Analysis for sentiment analysis data
    """
    
    def __init__(self, df):
        self.df = df
        self.word_frequencies = None
        
    def calculate_word_frequencies(self):
        """Calculate word frequencies from tokens"""
        print("Calculating word frequencies...")
        
        # Flatten all tokens into a single list
        all_tokens = [token for tokens in self.df['tokens'] for token in tokens]
        
        # Count frequencies
        self.word_frequencies = Counter(all_tokens)
        
        print(f"✓ Found {len(self.word_frequencies)} unique words")
        return self.word_frequencies
    
    def generate_wordcloud(self, max_words=100, width=1600, height=800, 
                          background_color='white', colormap='viridis',
                          save_path='wordcloud.png'):
        """
        Generate and save word cloud visualization
        
        Args:
            max_words: Maximum number of words to display
            width: Width of the image
            height: Height of the image
            background_color: Background color
            colormap: Matplotlib colormap
            save_path: Path to save the image
        """
        print(f"Generating word cloud with top {max_words} words...")
        
        if self.word_frequencies is None:
            self.calculate_word_frequencies()
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            relative_scaling=0.5,
            min_font_size=10,
            collocations=False  # Don't count word pairs
        ).generate_from_frequencies(self.word_frequencies)
        
        # Create figure
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Most Frequent Words', fontsize=20, pad=20)
        plt.tight_layout(pad=0)
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Word cloud saved to {save_path}")
        
        return wordcloud
    
    def plot_frequency_distribution(self, top_n=20, save_path='frequency_distribution.png'):
        """
        Generate and save frequency distribution bar plot
        
        Args:
            top_n: Number of top words to display
            save_path: Path to save the image
        """
        print(f"Generating frequency distribution for top {top_n} words...")
        
        if self.word_frequencies is None:
            self.calculate_word_frequencies()
        
        # Get top N words
        top_words = self.word_frequencies.most_common(top_n)
        words, counts = zip(*top_words)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create bar plot
        bars = plt.bar(range(len(words)), counts, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Customize
        plt.xlabel('Words', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        plt.title(f'Top {top_n} Most Frequent Words', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Frequency distribution saved to {save_path}")
        
        return top_words
    
    def plot_horizontal_frequency(self, top_n=20, save_path='frequency_horizontal.png'):
        """
        Generate horizontal bar chart for better readability
        """
        print(f"Generating horizontal frequency chart for top {top_n} words...")
        
        if self.word_frequencies is None:
            self.calculate_word_frequencies()
        
        # Get top N words
        top_words = self.word_frequencies.most_common(top_n)
        words, counts = zip(*top_words)
        
        # Reverse for better display (highest at top)
        words = words[::-1]
        counts = counts[::-1]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(words)), counts, color='coral', alpha=0.8, edgecolor='black')
        
        # Customize
        plt.xlabel('Frequency', fontsize=14, fontweight='bold')
        plt.ylabel('Words', fontsize=14, fontweight='bold')
        plt.title(f'Top {top_n} Most Frequent Words (Horizontal View)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.yticks(range(len(words)), words)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{count}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Horizontal frequency chart saved to {save_path}")
    
    def plot_token_length_distribution(self, save_path='token_length_distribution.png'):
        """Plot distribution of token counts per record"""
        print("Generating token length distribution...")
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.df['token_count'], bins=30, color='skyblue', 
                edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Tokens', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Token Counts', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(self.df['token_count'], vert=True)
        plt.ylabel('Number of Tokens', fontsize=12)
        plt.title('Token Count Box Plot', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Token length distribution saved to {save_path}")
    
    def plot_word_length_distribution(self, save_path='word_length_distribution.png'):
        """Plot distribution of word lengths"""
        print("Generating word length distribution...")
        
        # Get all word lengths
        word_lengths = [len(word) for word in self.word_frequencies.keys()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(word_lengths, bins=range(3, max(word_lengths)+2), 
                color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Word Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Word Lengths', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Word length distribution saved to {save_path}")
    
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        if self.word_frequencies is None:
            self.calculate_word_frequencies()
        
        print(f"\nDataset Overview:")
        print(f"  Total records: {len(self.df)}")
        print(f"  Total unique words: {len(self.word_frequencies)}")
        print(f"  Total word occurrences: {sum(self.word_frequencies.values())}")
        
        print(f"\nToken Statistics:")
        print(f"  Mean tokens per record: {self.df['token_count'].mean():.2f}")
        print(f"  Median tokens per record: {self.df['token_count'].median():.0f}")
        print(f"  Std deviation: {self.df['token_count'].std():.2f}")
        print(f"  Min tokens: {self.df['token_count'].min()}")
        print(f"  Max tokens: {self.df['token_count'].max()}")
        
        print(f"\nTop 20 Most Frequent Words:")
        for i, (word, count) in enumerate(self.word_frequencies.most_common(20), 1):
            print(f"  {i:2d}. {word:15s} : {count:4d} occurrences")
    
    def generate_all_visualizations(self):
        """Generate all EDA visualizations"""
        print("\n" + "=" * 60)
        print("GENERATING ALL VISUALIZATIONS")
        print("=" * 60 + "\n")
        
        # Calculate frequencies first
        self.calculate_word_frequencies()
        
        # Generate all plots
        self.generate_wordcloud()
        self.plot_frequency_distribution(top_n=20)
        self.plot_horizontal_frequency(top_n=20)
        self.plot_token_length_distribution()
        self.plot_word_length_distribution()
        
        # Display summary
        self.generate_summary_statistics()
        
        print("\n" + "=" * 60)
        print("✓ ALL VISUALIZATIONS COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - wordcloud.png")
        print("  - frequency_distribution.png")
        print("  - frequency_horizontal.png")
        print("  - token_length_distribution.png")
        print("  - word_length_distribution.png")


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # Load processed data
    try:
        df = pd.read_csv('processed_data.csv')
        print(f"✓ Loaded {len(df)} records from processed_data.csv")
    except FileNotFoundError:
        print("Error: processed_data.csv not found!")
        print("Please run preprocessing.py first")
        return
    
    # Convert tokens from string back to list if needed
    if isinstance(df['tokens'].iloc[0], str):
        import ast
        df['tokens'] = df['tokens'].apply(ast.literal_eval)
    
    # Initialize EDA analyzer
    eda = SentimentEDA(df)
    
    # Generate all visualizations
    eda.generate_all_visualizations()
    
    print("\n✓ EDA complete!")


if __name__ == "__main__":
    main()
