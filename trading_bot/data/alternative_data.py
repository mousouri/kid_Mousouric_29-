import tweepy
import praw
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from logger import Logger

class AlternativeData:
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger
        
        # Initialize Twitter API
        auth = tweepy.OAuthHandler(
            config['twitter_api_key'],
            config['twitter_api_secret']
        )
        auth.set_access_token(
            config['twitter_access_token'],
            config['twitter_access_token_secret']
        )
        self.twitter_api = tweepy.API(auth)
        
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=config['reddit_client_id'],
            client_secret=config['reddit_client_secret'],
            user_agent=config['reddit_user_agent']
        )
        
    def get_twitter_sentiment(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment from Twitter"""
        try:
            # Search for tweets
            query = f"${symbol} OR #{symbol}"
            tweets = self.twitter_api.search_tweets(
                q=query,
                lang="en",
                count=100,
                tweet_mode="extended"
            )
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets:
                # Simple sentiment analysis (you might want to use a proper NLP model)
                text = tweet.full_text.lower()
                positive_words = sum(1 for word in self.config['positive_words'] if word in text)
                negative_words = sum(1 for word in self.config['negative_words'] if word in text)
                
                sentiment = (positive_words - negative_words) / (positive_words + negative_words + 1)
                sentiments.append(sentiment)
                
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'symbol': symbol,
                'sentiment': avg_sentiment,
                'tweet_count': len(tweets),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment: {e}")
            return {
                'symbol': symbol,
                'sentiment': 0,
                'tweet_count': 0,
                'timestamp': datetime.now()
            }
            
    def get_reddit_sentiment(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment from Reddit"""
        try:
            # Search in relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            posts = []
            
            for subreddit in subreddits:
                sub = self.reddit.subreddit(subreddit)
                for post in sub.search(symbol, limit=50):
                    if post.created_utc > (datetime.now() - timedelta(hours=hours)).timestamp():
                        posts.append(post)
                        
            # Analyze sentiment
            sentiments = []
            for post in posts:
                # Simple sentiment analysis
                text = (post.title + " " + post.selftext).lower()
                positive_words = sum(1 for word in self.config['positive_words'] if word in text)
                negative_words = sum(1 for word in self.config['negative_words'] if word in text)
                
                sentiment = (positive_words - negative_words) / (positive_words + negative_words + 1)
                sentiments.append(sentiment)
                
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'symbol': symbol,
                'sentiment': avg_sentiment,
                'post_count': len(posts),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment: {e}")
            return {
                'symbol': symbol,
                'sentiment': 0,
                'post_count': 0,
                'timestamp': datetime.now()
            }
            
    def analyze_order_flow(self, order_book: Dict) -> Dict:
        """Analyze order flow from Level 2 data"""
        try:
            # Get bid and ask orders
            bids = np.array(order_book['bids'], dtype=float)
            asks = np.array(order_book['asks'], dtype=float)
            
            # Calculate order flow metrics
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            
            bid_ask_ratio = bid_volume / (ask_volume + 1e-10)
            
            # Calculate price levels
            bid_price_levels = len(bids)
            ask_price_levels = len(asks)
            
            # Calculate spread
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            spread = best_ask - best_bid
            
            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'bid_ask_ratio': bid_ask_ratio,
                'bid_price_levels': bid_price_levels,
                'ask_price_levels': ask_price_levels,
                'spread': spread,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow: {e}")
            return {
                'bid_volume': 0,
                'ask_volume': 0,
                'bid_ask_ratio': 0,
                'bid_price_levels': 0,
                'ask_price_levels': 0,
                'spread': 0,
                'timestamp': datetime.now()
            }
            
    def get_combined_sentiment(self, symbol: str) -> Dict:
        """Get combined sentiment from all sources"""
        try:
            twitter_sentiment = self.get_twitter_sentiment(symbol)
            reddit_sentiment = self.get_reddit_sentiment(symbol)
            
            # Combine sentiments (you might want to weight them differently)
            combined_sentiment = (
                twitter_sentiment['sentiment'] * 0.6 +
                reddit_sentiment['sentiment'] * 0.4
            )
            
            return {
                'symbol': symbol,
                'combined_sentiment': combined_sentiment,
                'twitter_sentiment': twitter_sentiment['sentiment'],
                'reddit_sentiment': reddit_sentiment['sentiment'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting combined sentiment: {e}")
            return {
                'symbol': symbol,
                'combined_sentiment': 0,
                'twitter_sentiment': 0,
                'reddit_sentiment': 0,
                'timestamp': datetime.now()
            } 