"""
Social Media Sentiment Analysis - Data Acquisition & Preprocessing
Assignment 1: Foundation Module
"""

import re
import pandas as pd
import tweepy
from datetime import datetime
import json
import os

# Alternative: Using snscrape for Twitter data (no API key needed)
try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    SNSCRAPE_AVAILABLE = False

class SocialMediaDataCollector:
    """
    Collects social media data from Twitter/X or simulated data for testing
    """
    
    def __init__(self, keyword, num_records=500):
        self.keyword = keyword
        self.num_records = num_records
        self.data = []
    
    def fetch_twitter_data_snscrape(self):
        """
        Fetch tweets using snscrape (no API key required)
        """
        if not SNSCRAPE_AVAILABLE:
            print("snscrape not installed. Install with: pip install snscrape --break-system-packages")
            return None
        
        print(f"Fetching {self.num_records} tweets for keyword: '{self.keyword}'")
        tweets_list = []
        
        # Query format: keyword since:YYYY-MM-DD until:YYYY-MM-DD
        query = f"{self.keyword} lang:en -filter:retweets"
        
        try:
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= self.num_records:
                    break
                
                tweets_list.append({
                    'id': tweet.id,
                    'date': tweet.date,
                    'text': tweet.rawContent,
                    'username': tweet.user.username,
                    'likes': tweet.likeCount,
                    'retweets': tweet.retweetCount,
                    'replies': tweet.replyCount,
                    'source': 'twitter'
                })
                
                if (i + 1) % 50 == 0:
                    print(f"Collected {i + 1} tweets...")
            
            self.data = tweets_list
            print(f"✓ Successfully collected {len(tweets_list)} tweets")
            return tweets_list
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return None
    
    def fetch_twitter_data_tweepy(self, bearer_token=None):
        """
        Fetch tweets using official Twitter API v2 (requires API key)
        """
        if not bearer_token:
            print("No bearer token provided. Please set TWITTER_BEARER_TOKEN environment variable")
            return None
        
        try:
            client = tweepy.Client(bearer_token=bearer_token)
            
            print(f"Fetching {self.num_records} tweets for keyword: '{self.keyword}'")
            tweets = client.search_recent_tweets(
                query=f"{self.keyword} -is:retweet lang:en",
                max_results=min(100, self.num_records),
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            tweets_list = []
            for tweet in tweets.data:
                tweets_list.append({
                    'id': tweet.id,
                    'date': tweet.created_at,
                    'text': tweet.text,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count'],
                    'replies': tweet.public_metrics['reply_count'],
                    'source': 'twitter'
                })
            
            self.data = tweets_list
            print(f"✓ Successfully collected {len(tweets_list)} tweets")
            return tweets_list
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return None
    
    def generate_sample_data(self):
        """
        Generate sample social media data for testing/demonstration
        """
        print(f"Generating {self.num_records} sample records for keyword: '{self.keyword}'")
        
        # Sample templates with varying sentiments
        positive_templates = [
            f"I absolutely love {self.keyword}! Best decision ever 😊",
            f"{self.keyword} is amazing! Highly recommend it to everyone",
            f"Just got {self.keyword} and I'm so happy! Worth every penny",
            f"Can't stop using {self.keyword}! Game changer for me",
            f"{self.keyword} exceeded all my expectations! Fantastic!",
            f"Obsessed with {self.keyword}! Life is so much better now",
            f"{self.keyword} is incredible! Everyone should try it",
            f"Best purchase ever! {self.keyword} is phenomenal 🎉",
        ]
        
        negative_templates = [
            f"{self.keyword} is so disappointing. Not worth the hype",
            f"Terrible experience with {self.keyword}. Don't recommend it",
            f"Wasted my money on {self.keyword}. Very frustrated 😞",
            f"{self.keyword} has too many issues. Not happy at all",
            f"Regret buying {self.keyword}. Should have read reviews first",
            f"{self.keyword} is overrated. Expected much better quality",
            f"Very disappointed with {self.keyword}. Customer service is awful",
            f"Don't waste your time on {self.keyword}. Terrible product",
        ]
        
        neutral_templates = [
            f"Checking out {self.keyword} today. Anyone have experience?",
            f"What do people think about {self.keyword}?",
            f"Considering {self.keyword}. Need more information first",
            f"Has anyone used {self.keyword} recently?",
            f"Looking into {self.keyword} options. Any recommendations?",
            f"Thinking about trying {self.keyword}. Mixed reviews though",
            f"{self.keyword} seems interesting. Need to research more",
            f"Comparing {self.keyword} with alternatives. Thoughts?",
        ]
        
        import random
        from datetime import timedelta
        
        tweets_list = []
        base_date = datetime.now()
        
        for i in range(self.num_records):
            # Random sentiment distribution: 40% positive, 30% negative, 30% neutral
            rand = random.random()
            if rand < 0.4:
                text = random.choice(positive_templates)
            elif rand < 0.7:
                text = random.choice(negative_templates)
            else:
                text = random.choice(neutral_templates)
            
            # Add some variation
            if random.random() > 0.5:
                text += f" #{self.keyword.replace(' ', '')}"
            
            tweets_list.append({
                'id': f"sample_{i}",
                'date': base_date - timedelta(hours=random.randint(1, 720)),
                'text': text,
                'username': f"user{random.randint(1000, 9999)}",
                'likes': random.randint(0, 500),
                'retweets': random.randint(0, 100),
                'replies': random.randint(0, 50),
                'source': 'sample_data'
            })
        
        self.data = tweets_list
        print(f"✓ Successfully generated {len(tweets_list)} sample records")
        return tweets_list
    
    def save_to_csv(self, filename='raw_data.csv'):
        """Save collected data to CSV"""
        if not self.data:
            print("No data to save!")
            return None
        
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"✓ Data saved to {filename}")
        return df
    
    def save_to_json(self, filename='raw_data.json'):
        """Save collected data to JSON"""
        if not self.data:
            print("No data to save!")
            return None
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"✓ Data saved to {filename}")
        return self.data


def main():
    """
    Main execution function
    """
    # Configuration
    KEYWORD = "Python programming"  # Change this to your target keyword
    NUM_RECORDS = 500
    
    print("=" * 60)
    print("SOCIAL MEDIA SENTIMENT ANALYSIS - DATA ACQUISITION")
    print("=" * 60)
    
    # Initialize collector
    collector = SocialMediaDataCollector(keyword=KEYWORD, num_records=NUM_RECORDS)
    
    # Try different data collection methods
    print("\n[1] Attempting to collect real Twitter data...")
    
    # Method 1: Try snscrape (no API key needed)
    if SNSCRAPE_AVAILABLE:
        data = collector.fetch_twitter_data_snscrape()
    else:
        # Method 2: Try Tweepy with API key
        bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        if bearer_token:
            data = collector.fetch_twitter_data_tweepy(bearer_token)
        else:
            data = None
    
    # Method 3: Generate sample data if real data unavailable
    if not data:
        print("\n[2] Generating sample data for demonstration...")
        data = collector.generate_sample_data()
    
    # Save data
    print("\n[3] Saving collected data...")
    df = collector.save_to_csv('raw_data.csv')
    collector.save_to_json('raw_data.json')
    
    # Display summary
    if df is not None:
        print("\n" + "=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total records: {len(df)}")
        print(f"Keyword: {KEYWORD}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nFirst 3 records:")
        print(df[['date', 'text', 'likes']].head(3))
        print("\n✓ Data acquisition complete!")


if __name__ == "__main__":
    main()
