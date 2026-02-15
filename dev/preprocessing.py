"""
Social Media Sentiment Analysis - Text Preprocessing Pipeline
Using Regex, NLTK, and SpaCy
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK averaged_perceptron_tagger...")
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for social media data
    """
    
    def __init__(self, use_lemmatization=True, remove_stopwords=True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Additional social media stop words
        self.social_stopwords = {
            'rt', 'via', 'amp', 'gt', 'lt', 'http', 'https',
            'like', 'follow', 'retweet', 'dm', 'u', 'ur', 'pls'
        }
        self.stop_words.update(self.social_stopwords)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        # Pattern for http/https URLs
        text = re.sub(r'http\S+|https\S+', '', text)
        # Pattern for www URLs
        text = re.sub(r'www\.\S+', '', text)
        return text
    
    def remove_mentions(self, text):
        """Remove @mentions from text"""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text):
        """Remove # but keep the hashtag text"""
        return re.sub(r'#(\w+)', r'\1', text)
    
    def remove_special_characters(self, text):
        """Remove special characters and punctuation"""
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def remove_numbers(self, text):
        """Remove standalone numbers"""
        return re.sub(r'\b\d+\b', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace and normalize"""
        return ' '.join(text.split())
    
    def remove_emojis(self, text):
        """Remove emojis from text"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def expand_contractions(self, text):
        """Expand common English contractions"""
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have",
            "couldn't": "could not", "didn't": "did not",
            "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would",
            "he'll": "he will", "he's": "he is",
            "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would",
            "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not",
            "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have",
            "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are",
            "we've": "we have", "weren't": "were not",
            "what'll": "what will", "what're": "what are",
            "what's": "what is", "what've": "what have",
            "where's": "where is", "who'd": "who would",
            "who'll": "who will", "who're": "who are",
            "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are",
            "you've": "you have"
        }
        
        words = text.split()
        expanded = [contractions.get(word.lower(), word) for word in words]
        return ' '.join(expanded)
    
    def tokenize_text(self, text):
        """Tokenize text into words"""
        return word_tokenize(text.lower())
    
    def remove_stopwords_from_tokens(self, tokens):
        """Remove stop words from token list"""
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens to their base form"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def stem_tokens(self, tokens):
        """Stem tokens using Porter Stemmer"""
        return [self.stemmer.stem(word) for word in tokens]
    
    def clean_text(self, text, keep_hashtags=False):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            keep_hashtags: Whether to keep hashtag text (default: False)
        
        Returns:
            cleaned_text: Cleaned text string
            tokens: List of processed tokens
        """
        if not isinstance(text, str):
            return "", []
        
        # Step 1: Convert to lowercase
        text = text.lower()
        
        # Step 2: Expand contractions
        text = self.expand_contractions(text)
        
        # Step 3: Remove URLs
        text = self.remove_urls(text)
        
        # Step 4: Remove mentions
        text = self.remove_mentions(text)
        
        # Step 5: Handle hashtags
        if keep_hashtags:
            text = self.remove_hashtags(text)
        else:
            text = re.sub(r'#\w+', '', text)
        
        # Step 6: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 7: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 8: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 9: Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Step 10: Tokenize
        tokens = self.tokenize_text(text)
        
        # Step 11: Remove stop words
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 12: Lemmatize or stem
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)
        
        # Step 13: Remove short tokens (length < 3)
        tokens = [word for word in tokens if len(word) >= 3]
        
        # Reconstruct cleaned text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text, tokens
    
    def process_dataframe(self, df, text_column='text'):
        """
        Process entire dataframe
        
        Args:
            df: Pandas DataFrame
            text_column: Name of column containing text
        
        Returns:
            df: DataFrame with additional columns for cleaned text and tokens
        """
        print("Processing text data...")
        
        # Apply cleaning to each row
        results = df[text_column].apply(self.clean_text)
        
        # Extract cleaned text and tokens
        df['cleaned_text'] = results.apply(lambda x: x[0])
        df['tokens'] = results.apply(lambda x: x[1])
        df['token_count'] = df['tokens'].apply(len)
        
        # Remove rows with empty cleaned text
        initial_count = len(df)
        df = df[df['cleaned_text'].str.strip() != '']
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} records with no meaningful content")
        
        print(f"✓ Processing complete! {len(df)} records processed")
        return df


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("TEXT PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load raw data
    try:
        df = pd.read_csv('raw_data.csv')
        print(f"✓ Loaded {len(df)} records from raw_data.csv")
    except FileNotFoundError:
        print("Error: raw_data.csv not found!")
        print("Please run data_acquisition.py first")
        return
    
    # Display sample raw text
    print("\n[Sample Raw Text]")
    print("-" * 60)
    for i in range(min(3, len(df))):
        print(f"{i+1}. {df.iloc[i]['text'][:100]}...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        use_lemmatization=True,
        remove_stopwords=True
    )
    
    # Process data
    print("\n[Preprocessing]")
    print("-" * 60)
    df_processed = preprocessor.process_dataframe(df, text_column='text')
    
    # Display sample cleaned text
    print("\n[Sample Cleaned Text]")
    print("-" * 60)
    for i in range(min(3, len(df_processed))):
        print(f"{i+1}. Original: {df_processed.iloc[i]['text'][:80]}...")
        print(f"   Cleaned:  {df_processed.iloc[i]['cleaned_text']}")
        print(f"   Tokens:   {df_processed.iloc[i]['tokens'][:10]}")
        print()
    
    # Save processed data
    df_processed.to_csv('processed_data.csv', index=False)
    print("\n✓ Processed data saved to processed_data.csv")
    
    # Display statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {len(df_processed)}")
    print(f"Average tokens per record: {df_processed['token_count'].mean():.1f}")
    print(f"Min tokens: {df_processed['token_count'].min()}")
    print(f"Max tokens: {df_processed['token_count'].max()}")
    
    # Show most common words preview
    from collections import Counter
    all_tokens = [token for tokens in df_processed['tokens'] for token in tokens]
    word_freq = Counter(all_tokens)
    print(f"\nTop 10 most common words:")
    for word, count in word_freq.most_common(10):
        print(f"  {word}: {count}")
    
    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
