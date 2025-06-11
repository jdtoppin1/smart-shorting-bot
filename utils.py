"""
Utility functions and logging setup for the Smart Shorting System
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import smtplib
from email.mime.text import MIMEText
from config import MONITORING_CONFIG, LOG_LEVEL

def setup_logging():
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Console logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # File logging if enabled
    if MONITORING_CONFIG["log_to_file"]:
        file_handler = logging.FileHandler(MONITORING_CONFIG["log_file_path"])
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger("SmartShortingBot")

def save_trade_result(trade_data: Dict[str, Any]):
    """Save trade results to JSON file for analysis"""
    filename = f"trade_results_{datetime.now().strftime('%Y%m')}.json"
    
    # Load existing data
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            trades = json.load(f)
    else:
        trades = []
    
    # Add timestamp
    trade_data['timestamp'] = datetime.now().isoformat()
    trades.append(trade_data)
    
    # Save updated data
    with open(filename, 'w') as f:
        json.dump(trades, f, indent=2)

def send_alert(subject: str, message: str, logger):
    """Send email alert if configured"""
    if not MONITORING_CONFIG["enable_email_alerts"]:
        return
    
    try:
        email_config = MONITORING_CONFIG["email_settings"]
        
        msg = MIMEText(message)
        msg['Subject'] = f"Smart Shorting Alert: {subject}"
        msg['From'] = email_config["from_email"]
        msg['To'] = email_config["to_email"]
        
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["from_email"], email_config["password"])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Alert sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

def calculate_position_size(current_portfolio_value: float, risk_percent: float = 0.05) -> float:
    """Calculate position size based on risk management rules"""
    from config import RISK_CONFIG
    max_position = current_portfolio_value * RISK_CONFIG["max_position_size"]
    risk_position = current_portfolio_value * risk_percent
    return min(max_position, risk_position)

def is_market_hours() -> bool:
    """Check if it's during market hours (9:30 AM - 4:00 PM ET)"""
    now = datetime.now()
    # Simple check - you might want to add timezone handling and holiday checking
    if now.weekday() >= 5:  # Weekend
        return False
    
    hour = now.hour
    return 9 <= hour <= 16  # Rough market hours

def clean_text(text: str) -> str:
    """Clean text for sentiment analysis"""
    if not text:
        return ""
    
    # Remove URLs, mentions, special characters
    import re
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    text = ' '.join(text.split())  # Remove extra whitespace
    
    return text[:512]  # Limit length for models

def get_trading_performance() -> Dict[str, float]:
    """Calculate basic trading performance metrics"""
    current_month = datetime.now().strftime('%Y%m')
    filename = f"trade_results_{current_month}.json"
    
    if not os.path.exists(filename):
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
    
    with open(filename, 'r') as f:
        trades = json.load(f)
    
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
    
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": total_pnl / total_trades if total_trades > 0 else 0
    }

class RateLimiter:
    """Simple rate limiter to avoid hitting API limits"""
    
    def __init__(self):
        self.calls = {}
    
    def can_call(self, api_name: str, max_calls_per_minute: int = 60) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        if api_name not in self.calls:
            self.calls[api_name] = []
        
        # Remove old calls
        self.calls[api_name] = [call_time for call_time in self.calls[api_name] if call_time > minute_ago]
        
        # Check if we can make another call
        if len(self.calls[api_name]) < max_calls_per_minute:
            self.calls[api_name].append(now)
            return True
        
        return False

# Global rate limiter instance
rate_limiter = RateLimiter()