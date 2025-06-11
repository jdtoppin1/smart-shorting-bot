"""
News fetcher that collects financial news from multiple sources
"""

import yfinance as yf
import feedparser
import praw
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
import logging
from utils import setup_logging, clean_text, rate_limiter
from config import NEWS_SOURCES, REDDIT_CONFIG

logger = setup_logging()

class NewsFetcher:
    """Fetches news from multiple sources"""
    
    def __init__(self):
        self.reddit = None
        self.setup_reddit()
    
    def setup_reddit(self):
        """Setup Reddit API connection"""
        try:
            if (REDDIT_CONFIG["client_id"] != "YOUR_REDDIT_CLIENT_ID" and 
                NEWS_SOURCES["reddit"]["enabled"]):
                
                self.reddit = praw.Reddit(
                    client_id=REDDIT_CONFIG["client_id"],
                    client_secret=REDDIT_CONFIG["client_secret"],
                    user_agent=REDDIT_CONFIG["user_agent"]
                )
                logger.info("Reddit API connected successfully")
            else:
                logger.warning("Reddit API not configured - skipping Reddit news")
        except Exception as e:
            logger.error(f"Failed to setup Reddit API: {e}")
    
    def fetch_reddit_sentiment(self) -> List[Dict[str, Any]]:
        """Fetch posts and comments from financial subreddits"""
        if not self.reddit or not NEWS_SOURCES["reddit"]["enabled"]:
            return []
        
        news_items = []
        config = NEWS_SOURCES["reddit"]
        
        try:
            for subreddit_name in config["subreddits"]:
                if not rate_limiter.can_call("reddit", 30):  # 30 calls per minute limit
                    logger.warning("Reddit rate limit reached, skipping")
                    break
                
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=config["post_limit"]):
                    # Look for stock tickers in title
                    tickers = self.extract_tickers(post.title + " " + post.selftext)
                    
                    if tickers:  # Only include if it mentions stocks
                        news_items.append({
                            "source": f"reddit_{subreddit_name}",
                            "title": post.title,
                            "text": clean_text(post.selftext),
                            "url": f"https://reddit.com{post.permalink}",
                            "timestamp": datetime.fromtimestamp(post.created_utc),
                            "score": post.score,
                            "tickers": tickers,
                            "type": "post"
                        })
                        
                        # Get top comments for additional sentiment
                        post.comments.replace_more(limit=0)
                        for comment in post.comments[:5]:  # Top 5 comments
                            if len(comment.body) > 50:  # Skip short comments
                                news_items.append({
                                    "source": f"reddit_{subreddit_name}_comment",
                                    "title": f"Comment on: {post.title}",
                                    "text": clean_text(comment.body),
                                    "url": f"https://reddit.com{comment.permalink}",
                                    "timestamp": datetime.fromtimestamp(comment.created_utc),
                                    "score": comment.score,
                                    "tickers": tickers,
                                    "type": "comment"
                                })
                
                time.sleep(1)  # Be nice to Reddit's servers
                
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
        
        logger.info(f"Fetched {len(news_items)} items from Reddit")
        return news_items
    
    def fetch_rss_news(self) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds"""
        if not NEWS_SOURCES["rss_feeds"]["enabled"]:
            return []
        
        news_items = []
        
        for feed_url in NEWS_SOURCES["rss_feeds"]["feeds"]:
            try:
                if not rate_limiter.can_call("rss", 20):  # 20 calls per minute
                    break
                
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit to recent items
                    # Parse timestamp
                    timestamp = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        timestamp = datetime(*entry.published_parsed[:6])
                    
                    # Only include recent news (last 24 hours)
                    if timestamp > datetime.now() - timedelta(hours=24):
                        tickers = self.extract_tickers(entry.title + " " + entry.get('summary', ''))
                        
                        news_items.append({
                            "source": "rss_" + feed_url.split('/')[2],  # Domain name
                            "title": entry.title,
                            "text": clean_text(entry.get('summary', '')),
                            "url": entry.link,
                            "timestamp": timestamp,
                            "tickers": tickers,
                            "type": "news"
                        })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        logger.info(f"Fetched {len(news_items)} items from RSS feeds")
        return news_items
    
    def fetch_yahoo_finance_news(self) -> List[Dict[str, Any]]:
        """Fetch news for specific tickers from Yahoo Finance"""
        if not NEWS_SOURCES["yahoo_finance"]["enabled"]:
            return []
        
        news_items = []
        
        for ticker in NEWS_SOURCES["yahoo_finance"]["tickers"]:
            try:
                if not rate_limiter.can_call("yahoo", 10):  # Conservative limit
                    break
                
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news:
                    timestamp = datetime.fromtimestamp(article.get('providerPublishTime', time.time()))
                    
                    # Only recent news
                    if timestamp > datetime.now() - timedelta(hours=24):
                        news_items.append({
                            "source": "yahoo_finance",
                            "title": article.get('title', ''),
                            "text": clean_text(article.get('summary', '')),
                            "url": article.get('link', ''),
                            "timestamp": timestamp,
                            "tickers": [ticker],
                            "type": "financial_news"
                        })
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching Yahoo Finance news for {ticker}: {e}")
        
        logger.info(f"Fetched {len(news_items)} items from Yahoo Finance")
        return news_items
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text using regex"""
        import re
        
        # Common ticker patterns
        ticker_pattern = r'\$([A-Z]{1,5})|([A-Z]{2,5})(?=\s|$|[^A-Z])'
        matches = re.findall(ticker_pattern, text.upper())
        
        # Flatten and filter
        tickers = []
        for match in matches:
            ticker = match[0] or match[1]
            if ticker and len(ticker) >= 2 and ticker not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'USE', 'MAN', 'NEW', 'NOW', 'WAY', 'MAY', 'SAY']:
                tickers.append(ticker)
        
        return list(set(tickers))  # Remove duplicates
    
    def fetch_all_headlines(self) -> List[Dict[str, Any]]:
        """Main function to fetch all news from all sources"""
        logger.info("Starting news collection from all sources...")
        
        all_news = []
        
        # Fetch from all sources
        reddit_news = self.fetch_reddit_sentiment()
        rss_news = self.fetch_rss_news()
        yahoo_news = self.fetch_yahoo_finance_news()
        
        all_news.extend(reddit_news)
        all_news.extend(rss_news)
        all_news.extend(yahoo_news)
        
        # Sort by timestamp (newest first)
        all_news.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove duplicates based on title similarity
        unique_news = self.deduplicate_news(all_news)
        
        logger.info(f"Collected {len(unique_news)} unique news items total")
        
        return unique_news
    
    def deduplicate_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate news items based on title similarity"""
        unique_items = []
        seen_titles = set()
        
        for item in news_items:
            # Simple deduplication - normalize title
            normalized_title = ''.join(item['title'].lower().split())
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_items.append(item)
        
        return unique_items

# Create global instance
news_fetcher = NewsFetcher()

# For backward compatibility
def fetch_all_headlines():
    """Legacy function for compatibility"""
    return news_fetcher.fetch_all_headlines()