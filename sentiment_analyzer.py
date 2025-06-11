"""
Sentiment analysis using FinBERT and other NLP models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging
from utils import setup_logging, clean_text
from config import SENTIMENT_CONFIG

logger = setup_logging()

class SentimentAnalyzer:
    """Analyzes sentiment of financial news using AI models"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_model()
    
    def setup_model(self):
        """Initialize the FinBERT model for financial sentiment analysis"""
        try:
            model_name = SENTIMENT_CONFIG["model"]
            logger.info(f"Loading sentiment model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Sentiment model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            logger.info("Falling back to simple keyword-based sentiment")
            self.setup_fallback_sentiment()
    
    def setup_fallback_sentiment(self):
        """Setup simple keyword-based sentiment as fallback"""
        self.positive_words = {
            'surge', 'soar', 'rally', 'gain', 'rise', 'jump', 'climb', 'boost', 
            'strong', 'bullish', 'optimistic', 'beat', 'exceed', 'outperform',
            'growth', 'profit', 'revenue', 'earnings', 'upgrade', 'buy'
        }
        
        self.negative_words = {
            'plunge', 'crash', 'fall', 'drop', 'decline', 'sink', 'tumble',
            'weak', 'bearish', 'pessimistic', 'miss', 'disappoint', 'underperform',
            'loss', 'deficit', 'concern', 'worry', 'risk', 'downgrade', 'sell',
            'layoff', 'bankruptcy', 'debt', 'recession', 'inflation'
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text"""
        if not text or len(text.strip()) < 5:
            return {"sentiment": "neutral", "confidence": 0.0, "score": 0.0}
        
        text = clean_text(text)
        
        try:
            if self.sentiment_pipeline:
                return self._analyze_with_model(text)
            else:
                return self._analyze_with_keywords(text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "score": 0.0}
    
    def _analyze_with_model(self, text: str) -> Dict[str, float]:
        """Use FinBERT model for sentiment analysis"""
        # Truncate text to model's max length
        max_length = SENTIMENT_CONFIG["max_text_length"]
        if len(text) > max_length:
            text = text[:max_length]
        
        result = self.sentiment_pipeline(text)[0]
        
        # Convert to standardized format
        label = result['label'].lower()
        confidence = result['score']
        
        # Map FinBERT labels to sentiment scores
        if 'positive' in label:
            score = confidence
            sentiment = "positive"
        elif 'negative' in label:
            score = -confidence
            sentiment = "negative"
        else:
            score = 0.0
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "score": score
        }
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based sentiment analysis"""
        words = text.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {"sentiment": "neutral", "confidence": 0.0, "score": 0.0}
        
        # Calculate sentiment score
        score = (positive_count - negative_count) / len(words)
        confidence = total_sentiment_words / len(words)
        
        if score > 0.01:
            sentiment = "positive"
        elif score < -0.01:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence, 1.0),
            "score": max(-1.0, min(1.0, score * 10))  # Scale to -1 to 1
        }
    
    def analyze_news_batch(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of news items"""
        logger.info(f"Analyzing sentiment for {len(news_items)} news items...")
        
        enriched_news = []
        
        for item in news_items:
            # Combine title and text for analysis
            full_text = f"{item.get('title', '')} {item.get('text', '')}"
            
            # Analyze sentiment
            sentiment_result = self.analyze_text_sentiment(full_text)
            
            # Add sentiment to item
            item_copy = item.copy()
            item_copy.update({
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["score"],
                "sentiment_confidence": sentiment_result["confidence"]
            })
            
            enriched_news.append(item_copy)
        
        logger.info("Sentiment analysis completed")
        return enriched_news
    
    def calculate_news_velocity(self, news_items: List[Dict[str, Any]], ticker: str) -> Dict[str, Any]:
        """
        Track how fast negative news is accelerating
        Sudden acceleration often indicates real problems
        """
        ticker_news = [item for item in news_items if ticker in item.get('tickers', [])]
        
        if len(ticker_news) < 2:
            return {"velocity": 0, "acceleration": 0, "signal": "neutral"}
        
        # Sort by timestamp
        ticker_news.sort(key=lambda x: x.get('timestamp', datetime.now()))
        
        # Count articles per hour over last 24 hours
        now = datetime.now()
        hourly_counts = {}
        
        for item in ticker_news:
            timestamp = item.get('timestamp', now)
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    continue
            
            hours_ago = int((now - timestamp).total_seconds() / 3600)
            if 0 <= hours_ago <= 24:
                hourly_counts[hours_ago] = hourly_counts.get(hours_ago, 0) + 1
        
        if len(hourly_counts) < 3:
            return {"velocity": 0, "acceleration": 0, "signal": "neutral"}
        
        # Calculate velocity (articles per hour)
        recent_velocity = sum(hourly_counts.get(i, 0) for i in range(0, 3)) / 3  # Last 3 hours
        baseline_velocity = sum(hourly_counts.get(i, 0) for i in range(12, 24)) / 12  # 12-24 hours ago
        
        # Calculate acceleration
        if baseline_velocity == 0:
            acceleration = recent_velocity * 2  # Arbitrary high value
        else:
            acceleration = recent_velocity / baseline_velocity
        
        # Generate signal
        signal = "accelerating" if acceleration > 2.0 else "neutral"
        
        return {
            "velocity": recent_velocity,
            "baseline_velocity": baseline_velocity,
            "acceleration": acceleration,
            "signal": signal,
            "total_articles_24h": sum(hourly_counts.values())
        }
        """Get sentiment summary for each ticker mentioned in news"""
        ticker_sentiment = {}
        
        for item in news_items:
            for ticker in item.get('tickers', []):
                if ticker not in ticker_sentiment:
                    ticker_sentiment[ticker] = {
                        "total_articles": 0,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "avg_sentiment_score": 0.0,
                        "sentiment_scores": [],
                        "recent_headlines": []
                    }
                
                ticker_data = ticker_sentiment[ticker]
                ticker_data["total_articles"] += 1
                
                sentiment = item.get("sentiment", "neutral")
                score = item.get("sentiment_score", 0.0)
                
                ticker_data["sentiment_scores"].append(score)
                
                if sentiment == "positive":
                    ticker_data["positive_count"] += 1
                elif sentiment == "negative":
                    ticker_data["negative_count"] += 1
                else:
                    ticker_data["neutral_count"] += 1
                
                # Keep track of recent headlines
                if len(ticker_data["recent_headlines"]) < 5:
                    ticker_data["recent_headlines"].append({
                        "title": item.get("title", ""),
                        "sentiment": sentiment,
                        "score": score,
                        "source": item.get("source", "")
                    })
        
        # Calculate averages
        for ticker, data in ticker_sentiment.items():
            if data["sentiment_scores"]:
                data["avg_sentiment_score"] = np.mean(data["sentiment_scores"])
                data["sentiment_volatility"] = np.std(data["sentiment_scores"])
            
            # Calculate sentiment ratio
            total = data["total_articles"]
            data["negative_ratio"] = data["negative_count"] / total if total > 0 else 0
            data["positive_ratio"] = data["positive_count"] / total if total > 0 else 0
        
        return ticker_sentiment

# Global instance
sentiment_analyzer = SentimentAnalyzer()

def analyze_sentiment(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Legacy function for compatibility"""
    return sentiment_analyzer.analyze_news_batch(news_items)