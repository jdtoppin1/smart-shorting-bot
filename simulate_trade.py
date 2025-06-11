"""
Trade simulation and portfolio management system
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
import numpy as np
import logging
from utils import setup_logging, save_trade_result, calculate_position_size
from config import SIMULATION_CONFIG, RISK_CONFIG

logger = setup_logging()

class Portfolio:
    """Manages portfolio state and positions"""
    
    def __init__(self, starting_capital: float = None):
        self.starting_capital = starting_capital or SIMULATION_CONFIG["starting_capital"]
        self.current_capital = self.starting_capital
        self.positions = {}  # ticker -> position data
        self.trade_history = []
        self.daily_pnl = []
        
        # Load existing portfolio if exists
        self.load_portfolio()
    
    def save_portfolio(self):
        """Save portfolio state to file"""
        portfolio_data = {
            "starting_capital": self.starting_capital,
            "current_capital": self.current_capital,
            "positions": self.positions,
            "trade_history": self.trade_history,
            "daily_pnl": self.daily_pnl,
            "last_updated": datetime.now().isoformat()
        }
        
        with open("portfolio_state.json", "w") as f:
            json.dump(portfolio_data, f, indent=2, default=str)
    
    def load_portfolio(self):
        """Load portfolio state from file"""
        if os.path.exists("portfolio_state.json"):
            try:
                with open("portfolio_state.json", "r") as f:
                    data = json.load(f)
                
                self.starting_capital = data.get("starting_capital", self.starting_capital)
                self.current_capital = data.get("current_capital", self.starting_capital)
                self.positions = data.get("positions", {})
                self.trade_history = data.get("trade_history", [])
                self.daily_pnl = data.get("daily_pnl", [])
                
                logger.info(f"Loaded portfolio: ${self.current_capital:,.2f} capital, {len(self.positions)} positions")
                
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
    
    def get_position_value(self, ticker: str) -> float:
        """Get current value of a position"""
        if ticker not in self.positions:
            return 0
        
        position = self.positions[ticker]
        try:
            current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            
            if position["type"] == "short":
                # For short positions, profit when price goes down
                pnl = position["shares"] * (position["entry_price"] - current_price)
            else:
                # For long positions, profit when price goes up
                pnl = position["shares"] * (current_price - position["entry_price"])
            
            return position["initial_value"] + pnl
            
        except Exception as e:
            logger.error(f"Error calculating position value for {ticker}: {e}")
            return position["initial_value"]
    
    def update_portfolio_value(self):
        """Update total portfolio value"""
        total_position_value = sum(self.get_position_value(ticker) for ticker in self.positions)
        cash = self.current_capital - sum(pos["initial_value"] for pos in self.positions.values())
        
        total_value = cash + total_position_value
        
        # Record daily P&L
        today = datetime.now().strftime("%Y-%m-%d")
        if not self.daily_pnl or self.daily_pnl[-1]["date"] != today:
            pnl_today = total_value - self.starting_capital
            self.daily_pnl.append({
                "date": today,
                "total_value": total_value,
                "pnl": pnl_today,
                "pnl_percent": (pnl_today / self.starting_capital) * 100
            })
        
        return total_value

class TradeSimulator:
    """Simulates trade execution and manages portfolio"""
    
    def __init__(self):
        self.portfolio = Portfolio()
        self.open_orders = []  # Track stop losses and take profits
    
    def calculate_volatility_adjusted_position_size(self, ticker: str, base_position_size: float) -> float:
        """
        Adjust position size based on stock volatility
        Smaller positions for volatile stocks, larger for stable stocks
        """
        try:
            data = yf.Ticker(ticker).history(period="30d")
            
            if len(data) < 10:
                return base_position_size  # Not enough data
            
            # Calculate 30-day volatility (annualized)
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Adjust position size inversely to volatility
            # High vol (>0.4) = smaller position
            # Low vol (<0.2) = larger position
            
            if volatility > 0.6:  # Very high volatility
                adjustment = 0.5
            elif volatility > 0.4:  # High volatility
                adjustment = 0.7
            elif volatility < 0.2:  # Low volatility
                adjustment = 1.3
            elif volatility < 0.15:  # Very low volatility
                adjustment = 1.5
            else:  # Normal volatility
                adjustment = 1.0
            
            adjusted_size = base_position_size * adjustment
            
            # Cap adjustments
            return max(0.01, min(adjusted_size, base_position_size * 2))
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {ticker}: {e}")
            return base_position_size

    def can_open_position(self, signal: Dict[str, Any]) -> bool:
        """Check if we can open a new position"""
        ticker = signal["ticker"]
        
        # Already have position in this ticker
        if ticker in self.portfolio.positions:
            return False
        
        # Calculate position value
        position_value = signal["position_size"] * self.portfolio.current_capital
        
        # Check if we have enough capital
        available_capital = self.portfolio.current_capital * 0.9  # Keep 10% cash buffer
        if position_value > available_capital:
            return False
        
        # Check daily trade limit
        today = datetime.now().strftime("%Y-%m-%d")
        trades_today = len([t for t in self.portfolio.trade_history if t.get("date") == today])
        if trades_today >= RISK_CONFIG["max_daily_trades"]:
            return False
        
        return True
    
    def execute_short_trade(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a short trade based on signal"""
        ticker = signal["ticker"]
        
        if not self.can_open_position(signal):
            logger.info(f"Cannot open position for {ticker} - insufficient capital or limits reached")
            return None
        
        try:
            # Get current price
            current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            
            # Calculate position size in dollars
            base_position_value = signal["position_size"] * self.portfolio.current_capital
            
            # 🎯 APPLY VOLATILITY ADJUSTMENT
            adjusted_position_size = self.calculate_volatility_adjusted_position_size(ticker, signal["position_size"])
            position_value = adjusted_position_size * self.portfolio.current_capital
            
            shares = int(position_value / current_price)
            
            if shares == 0:
                logger.info(f"Position too small for {ticker}")
                return None
            
            actual_position_value = shares * current_price
            commission = SIMULATION_CONFIG["commission_per_trade"]
            
            # Create position
            position = {
                "ticker": ticker,
                "type": "short",
                "shares": shares,
                "entry_price": current_price,
                "initial_value": actual_position_value,
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "entry_time": datetime.now().isoformat(),
                "signal_strength": signal["signal_strength"],
                "reasons": signal["reasons"]
            }
            
            # Add to portfolio
            self.portfolio.positions[ticker] = position
            
            # Record trade
            trade_record = {
                "type": "open_short",
                "ticker": ticker,
                "shares": shares,
                "price": current_price,
                "value": actual_position_value,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "signal_data": signal
            }
            
            self.portfolio.trade_history.append(trade_record)
            self.portfolio.save_portfolio()
            
            logger.info(f"OPENED SHORT: {shares} shares of {ticker} at ${current_price:.2f} (${actual_position_value:,.2f})")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing short trade for {ticker}: {e}")
            return None
    
    def check_stop_losses_and_take_profits(self):
        """Check all open positions for stop loss or take profit triggers"""
        positions_to_close = []
        
        for ticker, position in self.portfolio.positions.items():
            try:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                
                should_close = False
                close_reason = ""
                
                if position["type"] == "short":
                    # For short positions: stop loss when price goes UP, take profit when price goes DOWN
                    if current_price >= position["stop_loss"]:
                        should_close = True
                        close_reason = "stop_loss"
                    elif current_price <= position["take_profit"]:
                        should_close = True
                        close_reason = "take_profit"
                
                if should_close:
                    positions_to_close.append((ticker, close_reason, current_price))
                    
            except Exception as e:
                logger.error(f"Error checking position {ticker}: {e}")
        
        # Close triggered positions
        for ticker, reason, price in positions_to_close:
            self.close_position(ticker, reason, price)
    
    def close_position(self, ticker: str, reason: str = "manual", current_price: float = None):
        """Close a position"""
        if ticker not in self.portfolio.positions:
            logger.warning(f"No position found for {ticker}")
            return
        
        position = self.portfolio.positions[ticker]
        
        try:
            if current_price is None:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
            
            # Calculate P&L
            if position["type"] == "short":
                pnl = position["shares"] * (position["entry_price"] - current_price)
            else:
                pnl = position["shares"] * (current_price - position["entry_price"])
            
            pnl_percent = (pnl / position["initial_value"]) * 100
            commission = SIMULATION_CONFIG["commission_per_trade"]
            net_pnl = pnl - commission
            
            # Record trade
            trade_record = {
                "type": "close_position",
                "ticker": ticker,
                "shares": position["shares"],
                "entry_price": position["entry_price"],
                "exit_price": current_price,
                "pnl": net_pnl,
                "pnl_percent": pnl_percent,
                "commission": commission,
                "reason": reason,
                "hold_time_hours": (datetime.now() - datetime.fromisoformat(position["entry_time"])).total_seconds() / 3600,
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            
            self.portfolio.trade_history.append(trade_record)
            
            # Remove position
            del self.portfolio.positions[ticker]
            
            # Update capital
            self.portfolio.current_capital += net_pnl
            
            self.portfolio.save_portfolio()
            
            logger.info(f"CLOSED {position['type'].upper()}: {ticker} - P&L: ${net_pnl:.2f} ({pnl_percent:.1f}%) - Reason: {reason}")
            
            # Save detailed trade result
            save_trade_result(trade_record)
            
        except Exception as e:
            logger.error(f"Error closing position {ticker}: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self.portfolio.update_portfolio_value()
        
        # Calculate metrics
        total_pnl = total_value - self.portfolio.starting_capital
        total_pnl_percent = (total_pnl / self.portfolio.starting_capital) * 100
        
        # Position summary
        position_values = {}
        for ticker in self.portfolio.positions:
            position_values[ticker] = self.get_position_value(ticker)
        
        # Recent performance
        recent_trades = [t for t in self.portfolio.trade_history if t.get("type") == "close_position"][-10:]
        
        return {
            "total_value": total_value,
            "starting_capital": self.portfolio.starting_capital,
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_percent,
            "open_positions": len(self.portfolio.positions),
            "position_tickers": list(self.portfolio.positions.keys()),
            "position_values": position_values,
            "total_trades": len([t for t in self.portfolio.trade_history if t.get("type") in ["open_short", "open_long"]]),
            "recent_trades": recent_trades,
            "daily_pnl": self.portfolio.daily_pnl[-30:],  # Last 30 days
            "last_updated": datetime.now().isoformat()
        }
    
    def simulate_short_trade(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main function to simulate trades based on signals"""
        if not signals:
            logger.info("No signals to process")
            return {"executed_trades": 0, "message": "No signals"}
        
        logger.info(f"Processing {len(signals)} trading signals...")
        
        # Check existing positions for stop losses/take profits
        self.check_stop_losses_and_take_profits()
        
        executed_trades = []
        
        # Execute new trades
        for signal in signals:
            trade_result = self.execute_short_trade(signal)
            if trade_result:
                executed_trades.append(trade_result)
        
        # Get portfolio summary
        portfolio_summary = self.get_portfolio_summary()
        
        result = {
            "executed_trades": len(executed_trades),
            "trade_details": executed_trades,
            "portfolio_summary": portfolio_summary,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Executed {len(executed_trades)} new trades")
        logger.info(f"Portfolio value: ${portfolio_summary['total_value']:,.2f} (P&L: {portfolio_summary['total_pnl_percent']:.1f}%)")
        
        return result

# Global instance
trade_simulator = TradeSimulator()

# Legacy function for compatibility
def simulate_short_trade(signals):
    """Legacy function wrapper"""
    if isinstance(signals, dict):
        signals = signals.get("signals", [])
    return trade_simulator.simulate_short_trade(signals)