"""
Configuration file for the Smart Shorting System
Centralized place to modify all settings
"""

# GENERAL SETTINGS
LOOP_INTERVAL_SECONDS = 3600  # Run every hour
LOG_LEVEL = "INFO"
SIMULATION_MODE = True  # Set to False for real trading (when ready)

# NEWS SOURCES CONFIGURATION
NEWS_SOURCES = {
    "reddit": {
        "enabled": True,
        "subreddits": ["wallstreetbets", "stocks", "investing", "SecurityAnalysis"],
        "post_limit": 50,
        "comment_limit": 100
    },
    "rss_feeds": {
        "enabled": True,
        "feeds": [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/topstories/"
        ]
    },
    "yahoo_finance": {
        "enabled": True,
        "tickers": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    }
}

# SENTIMENT ANALYSIS SETTINGS
SENTIMENT_CONFIG = {
    "model": "ProsusAI/finbert",  # Free FinBERT model for financial sentiment
    "negative_threshold": -0.3,   # Threshold for "very negative" sentiment
    "batch_size": 32,
    "max_text_length": 512
}

# TRADING SIGNAL SETTINGS
SIGNAL_CONFIG = {
    "min_sentiment_score": -0.5,  # Minimum negative sentiment to consider
    "min_news_volume": 5,         # Minimum number of negative articles
    "price_drop_threshold": -0.02, # 2% price drop
    "volume_spike_threshold": 1.5,  # 50% above average volume
    "lookback_hours": 24          # Hours to look back for price/volume data
}

# RISK MANAGEMENT
RISK_CONFIG = {
    "max_position_size": 0.05,    # 5% of portfolio per trade
    "stop_loss_percent": 0.03,    # 3% stop loss
    "take_profit_percent": 0.10,  # 10% take profit
    "max_daily_trades": 5,
    "blacklist_tickers": ["UVXY", "VIX"]  # Avoid these
}

# REDDIT API (you'll need to get these from reddit.com/prefs/apps)
REDDIT_CONFIG = {
    "client_id": "2cXwspkc7tvPEDvpA3QD9g",
    "client_secret": "RZPyqAnagmR0EQTPhBDJiRMd425eUQ", 
    "user_agent": "SmartShortingBot/1.0"
}

# SIMULATION SETTINGS
SIMULATION_CONFIG = {
    "starting_capital": 50000,    # $50k simulation (change this!)
    "commission_per_trade": 1.0,  # $1 per trade
    "log_all_trades": True,
    "save_results_to_file": True
}

# ENHANCED FEATURES CONFIGURATION
ENHANCED_FEATURES = {
    "use_options_flow": True,        # Analyze options put/call ratios
    "use_earnings_filter": True,     # Skip stocks near earnings
    "use_news_velocity": True,       # Track news acceleration
    "use_volatility_adjustment": True, # Adjust position size by volatility
    "use_macro_conditions": True,    # Consider VIX, USD, yields
    "earnings_buffer_days": 2,       # Days to avoid before earnings
    "options_pc_threshold": 1.5,     # Put/call ratio threshold
    "news_acceleration_threshold": 2.0 # News velocity acceleration threshold
}

# MONITORING & ALERTS
MONITORING_CONFIG = {
    "log_to_file": True,
    "log_file_path": "trading_log.txt",
    "enable_email_alerts": False,  # Set to True if you want email notifications
    "email_settings": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "from_email": "your_email@gmail.com",
        "to_email": "your_email@gmail.com",
        "password": "your_app_password"
    }
}