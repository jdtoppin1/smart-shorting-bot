"""
Trading signal generator that combines sentiment analysis with technical indicators
"""

import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from utils import setup_logging
from config import SIGNAL_CONFIG, RISK_CONFIG
from sentiment_analyzer import sentiment_analyzer

logger = setup_logging()

class SignalGenerator:
    """Generates trading signals based on sentiment and technical analysis"""
    
    def __init__(self):
        self.cache = {}  # Cache price data to avoid repeated API calls
    
    def get_stock_data(self, ticker: str, period: str = "5d") -> Optional[Dict[str, Any]]:
        """Get stock price and volume data"""
        cache_key = f"{ticker}_{period}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate metrics
            price_change = (current_price - prev_close) / prev_close
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() if len(returns) > 1 else 0
            
            # Support/Resistance levels (simple)
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            
            data = {
                "ticker": ticker,
                "current_price": current_price,
                "prev_close": prev_close,
                "price_change": price_change,
                "price_change_percent": price_change * 100,
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "volatility": volatility,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "price_near_low": (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
            }
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_options_flow_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze options flow for unusual put activity
        High put volume often precedes stock drops
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expiry dates
            expiry_dates = stock.options
            if not expiry_dates:
                return {"signal": "neutral", "confidence": 0}
            
            # Get nearest expiry options (most liquid)
            nearest_expiry = expiry_dates[0]
            options_chain = stock.option_chain(nearest_expiry)
            
            puts = options_chain.puts
            calls = options_chain.calls
            
            if puts.empty or calls.empty:
                return {"signal": "neutral", "confidence": 0}
            
            # Calculate put/call volume ratio
            total_put_volume = puts['volume'].fillna(0).sum()
            total_call_volume = calls['volume'].fillna(0).sum()
            
            if total_call_volume == 0:
                pc_ratio = 10  # Very high if no call volume
            else:
                pc_ratio = total_put_volume / total_call_volume
            
            # Get unusual activity (high volume puts)
            puts['volume_rank'] = puts['volume'].rank(pct=True)
            unusual_puts = puts[puts['volume_rank'] > 0.9]  # Top 10% volume
            
            # Analyze signals
            signal_strength = 0
            reasons = []
            
            if pc_ratio > 1.5:  # More puts than calls
                signal_strength += 2
                reasons.append(f"High put/call ratio: {pc_ratio:.1f}")
            
            if len(unusual_puts) > 3:  # Multiple high-volume puts
                signal_strength += 1
                reasons.append(f"Unusual put activity: {len(unusual_puts)} contracts")
            
            # Check for near-the-money puts (most predictive)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            ntm_puts = puts[abs(puts['strike'] - current_price) / current_price < 0.05]
            ntm_volume = ntm_puts['volume'].fillna(0).sum()
            
            if ntm_volume > puts['volume'].quantile(0.8):
                signal_strength += 1
                reasons.append("High near-the-money put volume")
            
            return {
                "signal": "bearish" if signal_strength >= 2 else "neutral",
                "confidence": min(signal_strength / 4, 1.0),
                "pc_ratio": pc_ratio,
                "unusual_puts": len(unusual_puts),
                "ntm_put_volume": ntm_volume,
                "reasons": reasons
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options flow for {ticker}: {e}")
            return {"signal": "neutral", "confidence": 0, "error": str(e)}

    def is_near_earnings(self, ticker: str, days_buffer: int = 2) -> bool:
        """
        Check if stock has earnings in the next few days
        Avoid shorting right before earnings (too risky)
        """
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None or calendar.empty:
                return False  # No earnings data = assume safe
            
            # Get next earnings date
            earnings_date = calendar.index[0] if len(calendar) > 0 else None
            
            if earnings_date:
                days_until_earnings = (earnings_date.date() - datetime.now().date()).days
                return 0 <= days_until_earnings <= days_buffer
            
            return False
            
        except Exception as e:
            logger.debug(f"Could not get earnings data for {ticker}: {e}")
            return False  # Default to safe if can't determine

    def get_macro_market_conditions(self) -> Dict[str, Any]:
        """
        Analyze broader market conditions that affect shorting success
        """
        conditions = {}
        
        try:
            # VIX - Fear index
            vix = yf.Ticker("VIX").history(period="5d")
            if not vix.empty:
                current_vix = vix['Close'].iloc[-1]
                conditions["vix"] = {
                    "level": current_vix,
                    "signal": "high_fear" if current_vix > 25 else "low_fear" if current_vix < 15 else "normal"
                }
            
            # USD strength (affects risk assets)
            dxy = yf.Ticker("DX-Y.NYB").history(period="5d")
            if not dxy.empty:
                dxy_change = (dxy['Close'].iloc[-1] - dxy['Close'].iloc[0]) / dxy['Close'].iloc[0]
                conditions["usd"] = {
                    "5day_change": dxy_change,
                    "signal": "strengthening" if dxy_change > 0.01 else "weakening" if dxy_change < -0.01 else "stable"
                }
            
            # 10-year Treasury yield (risk-off indicator)
            tnx = yf.Ticker("^TNX").history(period="5d")
            if not tnx.empty:
                yield_change = tnx['Close'].iloc[-1] - tnx['Close'].iloc[0]
                conditions["yields"] = {
                    "10y_change": yield_change,
                    "signal": "rising" if yield_change > 0.1 else "falling" if yield_change < -0.1 else "stable"
                }
            
        except Exception as e:
            logger.error(f"Error getting macro conditions: {e}")
            conditions["error"] = str(e)
        
        # Overall market sentiment
        signals = [cond.get("signal", "neutral") for cond in conditions.values() if isinstance(cond, dict)]
        
        # Count bearish signals
        bearish_signals = sum(1 for signal in signals if signal in ["high_fear", "strengthening", "rising"])
        
        conditions["overall_sentiment"] = "bearish" if bearish_signals >= 2 else "neutral"
        
        return conditions

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals_from_sentiment(self, ticker_sentiment: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        for ticker, sentiment_data in ticker_sentiment.items():
            # Skip if not enough data
            if sentiment_data["total_articles"] < SIGNAL_CONFIG["min_news_volume"]:
                continue
            
            # Skip blacklisted tickers
            if ticker in RISK_CONFIG["blacklist_tickers"]:
                continue
            
            avg_sentiment = sentiment_data["avg_sentiment_score"]
            negative_ratio = sentiment_data["negative_ratio"]
            
            # Check if sentiment is sufficiently negative
            if (avg_sentiment < SIGNAL_CONFIG["min_sentiment_score"] and 
                negative_ratio > 0.6):  # 60% of articles are negative
                
                # Get price data
                price_data = self.get_stock_data(ticker)
                if not price_data:
                    continue
                
                # Generate signal
                signal = self.evaluate_short_signal(ticker, sentiment_data, price_data)
                if signal:
                    signals.append(signal)
        
        # Sort by signal strength
        signals.sort(key=lambda x: x["signal_strength"], reverse=True)
        
        # Limit to max daily trades
        max_trades = RISK_CONFIG["max_daily_trades"]
        return signals[:max_trades]
    
    def evaluate_short_signal(self, ticker: str, sentiment_data: Dict[str, Any], 
                            price_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if a ticker should be shorted based on multiple factors"""
        
        signal_strength = 0
        reasons = []
        
        # 🚨 EARNINGS FILTER - Skip if earnings coming up
        if self.is_near_earnings(ticker):
            logger.info(f"Skipping {ticker} - earnings within 2 days")
            return None
        
        # Sentiment factors
        avg_sentiment = sentiment_data["avg_sentiment_score"]
        negative_ratio = sentiment_data["negative_ratio"]
        
        if avg_sentiment < -0.3:
            signal_strength += 3
            reasons.append(f"Very negative sentiment ({avg_sentiment:.2f})")
        elif avg_sentiment < -0.1:
            signal_strength += 1
            reasons.append(f"Negative sentiment ({avg_sentiment:.2f})")
        
        if negative_ratio > 0.7:
            signal_strength += 2
            reasons.append(f"High negative news ratio ({negative_ratio:.1%})")
        
        # 📊 OPTIONS FLOW ANALYSIS
        options_signal = self.get_options_flow_signal(ticker)
        if options_signal["signal"] == "bearish":
            signal_strength += int(options_signal["confidence"] * 3)  # Up to +3 points
            reasons.extend(options_signal["reasons"])
        
        # Technical factors
        price_change = price_data["price_change"]
        volume_ratio = price_data["volume_ratio"]
        volatility = price_data["volatility"]
        
        # Price momentum
        if price_change < SIGNAL_CONFIG["price_drop_threshold"]:
            signal_strength += 2
            reasons.append(f"Price dropping ({price_change:.1%})")
        
        # Volume analysis
        if volume_ratio > SIGNAL_CONFIG["volume_spike_threshold"]:
            signal_strength += 1
            reasons.append(f"High volume ({volume_ratio:.1f}x average)")
        
        # Volatility (higher vol = more opportunity but more risk)
        if volatility > 0.03:  # 3% daily volatility
            signal_strength += 1
            reasons.append("High volatility")
        
        # Position near highs (good for shorting)
        if price_data["price_near_low"] > 0.8:  # Near 52-week high
            signal_strength += 1
            reasons.append("Near 52-week high")
        
        # 🌍 MACRO CONDITIONS
        macro_conditions = self.get_macro_market_conditions()
        if macro_conditions.get("overall_sentiment") == "bearish":
            signal_strength += 1
            reasons.append("Bearish macro environment")
        
        # Risk factors (reduce signal strength)
        if sentiment_data["total_articles"] < 3:
            signal_strength -= 1
            reasons.append("Limited news coverage")
        
        if volatility > 0.08:  # Very high volatility = too risky
            signal_strength -= 2
            reasons.append("Extremely high volatility - risky")
        
        # Minimum signal strength threshold (increased due to more signals)
        if signal_strength < 4:  # Raised from 3 to 4
            return None
        
        # Calculate position size
        base_position_size = RISK_CONFIG["max_position_size"]
        
        # Adjust position size based on confidence
        confidence_multiplier = min(signal_strength / 10, 1.0)  # Scale 0-1, raised denominator
        position_size = base_position_size * confidence_multiplier
        
        return {
            "ticker": ticker,
            "signal_type": "short",
            "signal_strength": signal_strength,
            "confidence": confidence_multiplier,
            "position_size": position_size,
            "entry_price": price_data["current_price"],
            "stop_loss": price_data["current_price"] * (1 + RISK_CONFIG["stop_loss_percent"]),
            "take_profit": price_data["current_price"] * (1 - RISK_CONFIG["take_profit_percent"]),
            "reasons": reasons,
            "sentiment_score": avg_sentiment,
            "negative_ratio": negative_ratio,
            "price_change": price_change,
            "volume_ratio": volume_ratio,
            "timestamp": datetime.now(),
            "news_count": sentiment_data["total_articles"],
            "recent_headlines": sentiment_data.get("recent_headlines", [])[:3],  # Top 3 headlines
            "options_analysis": options_signal,
            "macro_conditions": macro_conditions,
            "earnings_safe": True  # Passed earnings filter
        }
    
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        try:
            # Get market indices data
            spy_data = self.get_stock_data("SPY")
            vix_data = self.get_stock_data("VIX")
            
            conditions = {
                "overall_sentiment": "neutral",
                "market_trend": "sideways",
                "volatility_level": "normal",
                "recommendation": "proceed_with_caution"
            }
            
            if spy_data:
                spy_change = spy_data["price_change"]
                
                if spy_change < -0.015:  # Market down 1.5%+
                    conditions["market_trend"] = "bearish"
                    conditions["overall_sentiment"] = "negative"
                    conditions["recommendation"] = "favorable_for_shorts"
                elif spy_change > 0.015:  # Market up 1.5%+
                    conditions["market_trend"] = "bullish"
                    conditions["overall_sentiment"] = "positive"
                    conditions["recommendation"] = "avoid_shorts"
            
            if vix_data:
                vix_price = vix_data["current_price"]
                
                if vix_price > 25:
                    conditions["volatility_level"] = "high"
                elif vix_price > 20:
                    conditions["volatility_level"] = "elevated"
                else:
                    conditions["volatility_level"] = "low"
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {"recommendation": "proceed_with_caution"}
    
    def generate_trade_signals(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main function to generate all trading signals"""
        logger.info("Generating trading signals...")
        
        # Get sentiment summary by ticker
        ticker_sentiment = sentiment_analyzer.get_ticker_sentiment_summary(news_items)
        
        # Analyze market conditions
        market_conditions = self.analyze_market_conditions()
        
        # Generate signals
        signals = self.generate_signals_from_sentiment(ticker_sentiment)
        
        # Filter signals based on market conditions
        if market_conditions.get("recommendation") == "avoid_shorts":
            logger.info("Market conditions unfavorable for shorting - filtering signals")
            signals = [s for s in signals if s["signal_strength"] >= 6]  # Only very strong signals
        
        result = {
            "signals": signals,
            "market_conditions": market_conditions,
            "ticker_sentiment_summary": ticker_sentiment,
            "total_signals": len(signals),
            "timestamp": datetime.now()
        }
        
        logger.info(f"Generated {len(signals)} trading signals")
        
        return result

# Global instance
signal_generator = SignalGenerator()