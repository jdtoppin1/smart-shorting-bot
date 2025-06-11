"""
Main loop for the Smart Shorting System
Orchestrates all components: news fetching, sentiment analysis, signal generation, and trading
"""

import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Import all components
from utils import setup_logging, send_alert, is_market_hours, get_trading_performance
from config import LOOP_INTERVAL_SECONDS, SIMULATION_MODE
from news_fetcher import news_fetcher
from sentiment_analyzer import sentiment_analyzer
from signal_generator import signal_generator
from simulate_trade import trade_simulator
from model_trainer import model_trainer

# Setup logging
logger = setup_logging()

class SmartShortingSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.last_model_training = None
        self.system_start_time = datetime.now()
        self.total_cycles = 0
        self.errors_count = 0
        
    def should_train_model(self) -> bool:
        """Check if we should retrain the model"""
        # Train model daily at 6 AM (before market open)
        now = datetime.now()
        if (not self.last_model_training or 
            (now.hour == 6 and now.minute < 10 and 
             self.last_model_training.date() < now.date())):
            return True
        return False
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Main pipeline execution"""
        start_time = datetime.now()
        logger.info(f"🔄 Running smart shorting pipeline at {start_time}")
        
        pipeline_results = {
            "timestamp": start_time.isoformat(),
            "cycle_number": self.total_cycles + 1,
            "components": {},
            "performance": {},
            "errors": []
        }
        
        try:
            # Step 1: Check if market is open (optional - you might want to run analysis even when closed)
            market_open = is_market_hours()
            logger.info(f"Market hours check: {'Open' if market_open else 'Closed'}")
            
            # Step 2: Train model if needed
            if self.should_train_model():
                logger.info("🧠 Training ML model...")
                training_results = model_trainer.train_model()
                pipeline_results["components"]["model_training"] = training_results
                self.last_model_training = datetime.now()
                
                # Send alert if model improved significantly
                if (training_results.get("status") == "success" and 
                    training_results.get("improvement", 0) > 0.1):
                    send_alert(
                        "Model Improvement",
                        f"Trading model improved by {training_results['improvement']:.1%}",
                        logger
                    )
            
            # Step 3: Fetch news from all sources
            logger.info("📰 Fetching news from all sources...")
            news_items = news_fetcher.fetch_all_headlines()
            pipeline_results["components"]["news_fetching"] = {
                "total_items": len(news_items),
                "sources": list(set(item.get("source", "unknown") for item in news_items))
            }
            
            if not news_items:
                logger.warning("No news items fetched - skipping analysis")
                return pipeline_results
            
            # Step 4: Analyze sentiment
            logger.info("🎭 Analyzing sentiment...")
            analyzed_news = sentiment_analyzer.analyze_news_batch(news_items)
            
            # Get sentiment summary
            sentiment_summary = sentiment_analyzer.get_ticker_sentiment_summary(analyzed_news)
            pipeline_results["components"]["sentiment_analysis"] = {
                "analyzed_items": len(analyzed_news),
                "tickers_analyzed": len(sentiment_summary),
                "avg_sentiment": sum(item.get("sentiment_score", 0) for item in analyzed_news) / len(analyzed_news) if analyzed_news else 0
            }
            
            # Step 5: Generate trading signals
            logger.info("📊 Generating trading signals...")
            signal_results = signal_generator.generate_trade_signals(analyzed_news)
            signals = signal_results["signals"]
            
            pipeline_results["components"]["signal_generation"] = {
                "total_signals": len(signals),
                "market_conditions": signal_results["market_conditions"],
                "signal_strengths": [s["signal_strength"] for s in signals]
            }
            
            # Step 6: Apply ML model filtering (if available)
            if model_trainer.model and signals:
                logger.info("🤖 Applying ML model filtering...")
                filtered_signals = []
                
                for signal in signals:
                    ml_prediction = model_trainer.predict_signal_quality(signal)
                    signal["ml_prediction"] = ml_prediction
                    
                    # Only keep signals with high ML confidence
                    if ml_prediction["recommendation"] in ["strong_buy", "buy"]:
                        filtered_signals.append(signal)
                
                logger.info(f"ML model filtered signals: {len(signals)} -> {len(filtered_signals)}")
                signals = filtered_signals
                
                pipeline_results["components"]["ml_filtering"] = {
                    "original_signals": len(signal_results["signals"]),
                    "filtered_signals": len(signals),
                    "filter_ratio": len(signals) / len(signal_results["signals"]) if signal_results["signals"] else 0
                }
            
            # Step 7: Execute trades (simulation or real)
            if signals:
                logger.info(f"💰 {'Simulating' if SIMULATION_MODE else 'Executing'} {len(signals)} trades...")
                trade_results = trade_simulator.simulate_short_trade(signals)
                pipeline_results["components"]["trade_execution"] = trade_results
                
                # Send alerts for executed trades
                if trade_results["executed_trades"] > 0:
                    send_alert(
                        "Trades Executed",
                        f"Executed {trade_results['executed_trades']} new trades",
                        logger
                    )
            else:
                logger.info("No valid signals to execute")
                pipeline_results["components"]["trade_execution"] = {"executed_trades": 0, "message": "No signals"}
            
            # Step 8: Get performance summary
            performance = get_trading_performance()
            portfolio_summary = trade_simulator.get_portfolio_summary()
            
            pipeline_results["performance"] = {
                "trading_performance": performance,
                "portfolio_summary": portfolio_summary
            }
            
            # Log key metrics
            total_value = portfolio_summary.get("total_value", 0)
            total_pnl_percent = portfolio_summary.get("total_pnl_percent", 0)
            open_positions = portfolio_summary.get("open_positions", 0)
            
            logger.info(f"📈 Portfolio: ${total_value:,.2f} ({total_pnl_percent:+.1f}%) | Open positions: {open_positions}")
            
            # Step 9: Performance alerts
            if total_pnl_percent < -10:  # 10% loss alert
                send_alert(
                    "Large Loss Alert",
                    f"Portfolio down {total_pnl_percent:.1f}% (${portfolio_summary.get('total_pnl', 0):,.2f})",
                    logger
                )
            elif total_pnl_percent > 20:  # 20% gain alert
                send_alert(
                    "Large Gain Alert",
                    f"Portfolio up {total_pnl_percent:.1f}% (${portfolio_summary.get('total_pnl', 0):,.2f})",
                    logger
                )
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            pipeline_results["errors"].append({
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            
            self.errors_count += 1
            
            # Send error alert
            send_alert("System Error", error_msg, logger)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        pipeline_results["execution_time_seconds"] = execution_time
        
        self.total_cycles += 1
        
        logger.info(f"✅ Pipeline completed in {execution_time:.1f} seconds")
        
        return pipeline_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = datetime.now() - self.system_start_time
        
        return {
            "system_start_time": self.system_start_time.isoformat(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "total_cycles": self.total_cycles,
            "errors_count": self.errors_count,
            "error_rate": self.errors_count / max(self.total_cycles, 1),
            "last_model_training": self.last_model_training.isoformat() if self.last_model_training else None,
            "simulation_mode": SIMULATION_MODE,
            "next_run": (datetime.now() + timedelta(seconds=LOOP_INTERVAL_SECONDS)).isoformat()
        }
    
    def run_forever(self):
        """Main loop that runs forever"""
        logger.info("🚀 Smart Shorting System starting...")
        logger.info(f"Mode: {'SIMULATION' if SIMULATION_MODE else 'LIVE TRADING'}")
        logger.info(f"Loop interval: {LOOP_INTERVAL_SECONDS} seconds ({LOOP_INTERVAL_SECONDS/3600:.1f} hours)")
        
        while True:
            try:
                # Run the pipeline
                results = self.run_pipeline()
                
                # Print system status every 10 cycles
                if self.total_cycles % 10 == 0:
                    status = self.get_system_status()
                    logger.info(f"📊 System Status - Uptime: {status['uptime_hours']:.1f}h | Cycles: {status['total_cycles']} | Errors: {status['errors_count']}")
                
                # Sleep until next cycle
                logger.info(f"😴 Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
                time.sleep(LOOP_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logger.info("🛑 System shutdown requested by user")
                break
            except Exception as e:
                logger.error(f"Critical system error: {e}")
                logger.error(traceback.format_exc())
                
                # Send critical error alert
                send_alert("Critical System Error", str(e), logger)
                
                # Wait a bit before retrying
                time.sleep(60)
        
        logger.info("👋 Smart Shorting System stopped")

# Create system instance
system = SmartShortingSystem()

# Main execution
if __name__ == "__main__":
    system.run_forever()