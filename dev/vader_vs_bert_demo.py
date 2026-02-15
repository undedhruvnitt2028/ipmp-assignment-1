"""
Practical VADER vs BERT Comparison Script
Demonstrates real sentiment analysis on sample texts
"""

import time
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try to import transformers for BERT
try:
    from transformers import pipeline
    BERT_AVAILABLE = True
except ImportError:
    print("⚠ transformers not installed. Install with: pip install transformers torch --break-system-packages")
    BERT_AVAILABLE = False


class SentimentComparator:
    """Compare VADER and BERT sentiment analysis"""
    
    def __init__(self):
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize BERT (if available)
        if BERT_AVAILABLE:
            try:
                print("Loading BERT model (this may take a minute on first run)...")
                self.bert = pipeline(
                    'sentiment-analysis',
                    model='distilbert-base-uncased-finetuned-sst-2-english'
                )
                self.bert_loaded = True
                print("✓ BERT model loaded successfully\n")
            except Exception as e:
                print(f"⚠ Could not load BERT: {e}\n")
                self.bert_loaded = False
        else:
            self.bert_loaded = False
    
    def analyze_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        
        # Classify based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'POSITIVE'
        elif scores['compound'] <= -0.05:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'scores': scores
        }
    
    def analyze_bert(self, text):
        """Analyze sentiment using BERT"""
        if not self.bert_loaded:
            return None
        
        result = self.bert(text[:512])[0]  # BERT has 512 token limit
        
        return {
            'sentiment': result['label'],
            'confidence': result['score']
        }
    
    def compare_on_text(self, text):
        """Compare both models on a single text"""
        print(f"Text: \"{text}\"\n")
        
        # VADER analysis
        start = time.time()
        vader_result = self.analyze_vader(text)
        vader_time = time.time() - start
        
        print(f"VADER:")
        print(f"  Sentiment: {vader_result['sentiment']}")
        print(f"  Confidence: {vader_result['confidence']:.3f}")
        print(f"  Detailed scores: {vader_result['scores']}")
        print(f"  Processing time: {vader_time*1000:.2f}ms")
        
        # BERT analysis
        if self.bert_loaded:
            start = time.time()
            bert_result = self.analyze_bert(text)
            bert_time = time.time() - start
            
            print(f"\nBERT:")
            print(f"  Sentiment: {bert_result['sentiment']}")
            print(f"  Confidence: {bert_result['confidence']:.3f}")
            print(f"  Processing time: {bert_time*1000:.2f}ms")
            
            # Agreement check
            agreement = vader_result['sentiment'] == bert_result['sentiment']
            print(f"\n{'✓' if agreement else '✗'} Models {'agree' if agreement else 'disagree'}")
        
        print("\n" + "─" * 70 + "\n")


def demonstrate_differences():
    """Demonstrate cases where VADER and BERT differ"""
    
    print("=" * 70)
    print("VADER vs BERT: Practical Comparison")
    print("=" * 70 + "\n")
    
    comparator = SentimentComparator()
    
    # Test cases demonstrating different scenarios
    test_cases = [
        # Case 1: Simple positive sentiment
        {
            'text': "I love this product! It's amazing!",
            'description': "Simple positive sentiment"
        },
        # Case 2: Simple negative sentiment
        {
            'text': "This is terrible. I hate it.",
            'description': "Simple negative sentiment"
        },
        # Case 3: Sarcasm (BERT should handle better)
        {
            'text': "Oh great, another delay. Just what I needed today! 🙄",
            'description': "Sarcasm (BERT typically better)"
        },
        # Case 4: Mixed sentiment
        {
            'text': "The food was good but the service was awful.",
            'description': "Mixed sentiment"
        },
        # Case 5: Context-dependent negation
        {
            'text': "I didn't think it could get worse, but I was wrong - it's actually good!",
            'description': "Complex negation and context"
        },
        # Case 6: Emoji-heavy (VADER handles well)
        {
            'text': "Best day ever!!! 😍🎉🔥",
            'description': "Emoji-heavy positive"
        },
        # Case 7: Subtle negative
        {
            'text': "It's... okay, I guess. Not what I expected.",
            'description': "Subtle negative sentiment"
        },
        # Case 8: Social media slang
        {
            'text': "ngl this is fire fr fr no cap 💯",
            'description': "Social media slang"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Case {i}: {case['description']}")
        print("─" * 70)
        comparator.compare_on_text(case['text'])


def batch_performance_test():
    """Test performance on batch of texts"""
    print("=" * 70)
    print("BATCH PERFORMANCE TEST")
    print("=" * 70 + "\n")
    
    comparator = SentimentComparator()
    
    # Generate sample texts
    sample_texts = [
        "I love this!",
        "This is terrible.",
        "Meh, it's okay.",
        "Best purchase ever!",
        "Complete waste of money.",
        "Not bad, could be better.",
        "Absolutely fantastic! Highly recommend!",
        "Disappointed and frustrated.",
        "It works as expected.",
        "Exceeded my expectations!"
    ] * 10  # 100 texts total
    
    print(f"Testing on {len(sample_texts)} texts...\n")
    
    # VADER batch test
    start = time.time()
    vader_results = [comparator.analyze_vader(text) for text in sample_texts]
    vader_total_time = time.time() - start
    
    print(f"VADER Performance:")
    print(f"  Total time: {vader_total_time:.3f}s")
    print(f"  Average per text: {vader_total_time/len(sample_texts)*1000:.2f}ms")
    print(f"  Throughput: {len(sample_texts)/vader_total_time:.1f} texts/second")
    
    # BERT batch test
    if comparator.bert_loaded:
        start = time.time()
        bert_results = [comparator.analyze_bert(text) for text in sample_texts]
        bert_total_time = time.time() - start
        
        print(f"\nBERT Performance:")
        print(f"  Total time: {bert_total_time:.3f}s")
        print(f"  Average per text: {bert_total_time/len(sample_texts)*1000:.2f}ms")
        print(f"  Throughput: {len(sample_texts)/bert_total_time:.1f} texts/second")
        
        print(f"\nSpeed Comparison:")
        print(f"  VADER is {bert_total_time/vader_total_time:.1f}x faster than BERT")
    
    print("\n" + "=" * 70)


def analyze_your_data():
    """Analyze sentiment on your processed data"""
    print("=" * 70)
    print("ANALYZING YOUR PROCESSED DATA")
    print("=" * 70 + "\n")
    
    try:
        df = pd.read_csv('processed_data.csv')
        print(f"✓ Loaded {len(df)} records from processed_data.csv\n")
    except FileNotFoundError:
        print("✗ processed_data.csv not found!")
        print("Please run preprocessing.py first\n")
        return
    
    comparator = SentimentComparator()
    
    # Analyze first 10 records with VADER
    print("Analyzing first 10 records with VADER...\n")
    
    sentiments = []
    for idx, row in df.head(10).iterrows():
        text = row['text']
        result = comparator.analyze_vader(text)
        sentiments.append(result['sentiment'])
        
        print(f"{idx+1}. {text[:60]}...")
        print(f"   → {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print()
    
    # Summary
    from collections import Counter
    sentiment_counts = Counter(sentiments)
    
    print("─" * 70)
    print("Summary of first 10 records:")
    print(f"  Positive: {sentiment_counts.get('POSITIVE', 0)}")
    print(f"  Negative: {sentiment_counts.get('NEGATIVE', 0)}")
    print(f"  Neutral: {sentiment_counts.get('NEUTRAL', 0)}")
    print("\n" + "=" * 70)


def main():
    """Main demonstration"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "VADER vs BERT Comparison Demo" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nThis script demonstrates the practical differences between")
    print("VADER and BERT sentiment analysis models.\n")
    
    # Menu
    while True:
        print("\nWhat would you like to do?")
        print("  1. See side-by-side comparisons on test cases")
        print("  2. Run performance benchmark")
        print("  3. Analyze your processed data")
        print("  4. All of the above")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            demonstrate_differences()
        elif choice == '2':
            batch_performance_test()
        elif choice == '3':
            analyze_your_data()
        elif choice == '4':
            demonstrate_differences()
            batch_performance_test()
            analyze_your_data()
            break
        elif choice == '5':
            print("\nGoodbye! 👋\n")
            break
        else:
            print("\n⚠ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
