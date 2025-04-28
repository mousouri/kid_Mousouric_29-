import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

class AdvancedPremiumBot:
    def __init__(self):
        """Initialize the trading bot with default parameters"""
        # Load environment variables
        load_dotenv()
        
        # Initialize MT5
        if not mt5.initialize():
            logging.error("Failed to initialize MT5")
            raise Exception("MT5 initialization failed")
        
        # Core attributes
        self.symbols = []
        self.positions = {}
        self.websocket_connections = {}
        self.tick_data = {}
        self.quote_history = {}
        self.order_history = []
        self.performance_metrics = {}
        self.running = False
        
        # Load configuration
        self.load_configuration()
        
        # Initialize risk parameters
        self.risk_parameters = {
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', 10.0)),
            'max_daily_trades': int(os.getenv('MAX_DAILY_TRADES', 50)),
            'max_drawdown_percent': float(os.getenv('MAX_DRAWDOWN_PERCENT', 5.0)),
            'risk_per_trade_percent': float(os.getenv('RISK_PER_TRADE_PERCENT', 1.0))
        }
        
        # Initialize Telegram
        self.telegram_bot = None
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self.initialize_telegram()
        
        # State management
        self.last_state_save = datetime.now()
        self.state_save_interval = 300  # 5 minutes
        
        # Load previous state if exists
        self.load_state()

    def load_configuration(self):
        """Load bot configuration from environment variables"""
        try:
            self.symbols = os.getenv('TRADING_SYMBOLS', '').split(',')
            if not self.symbols:
                raise ValueError("No trading symbols configured")
            logging.info(f"Loaded trading symbols: {self.symbols}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def initialize_telegram(self):
        """Initialize Telegram bot for notifications"""
        try:
            from telegram import Bot
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            self.telegram_bot = Bot(token=token)
            logging.info("Telegram bot initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Telegram bot: {e}")

    def get_latest_prices(self) -> Dict[str, Dict]:
        """Get latest prices for all symbols"""
        try:
            prices = {}
            for symbol in self.symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    prices[symbol] = {
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'last': tick.last,
                        'volume': tick.volume,
                        'time': datetime.fromtimestamp(tick.time).isoformat()
                    }
            return prices
        except Exception as e:
            logging.error(f"Error getting latest prices: {e}")
            return {}

    def save_state(self):
        """Save bot state to file"""
        try:
            state = {
                'positions': self.positions,
                'order_history': self.order_history,
                'performance_metrics': self.performance_metrics,
                'risk_parameters': self.risk_parameters,
                'last_save': datetime.now().isoformat()
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, default=str)
            
            logging.info("Bot state saved successfully")
        except Exception as e:
            logging.error(f"Error saving bot state: {e}")

    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                self.positions = state.get('positions', {})
                self.order_history = state.get('order_history', [])
                self.performance_metrics = state.get('performance_metrics', {})
                self.risk_parameters.update(state.get('risk_parameters', {}))
                
                logging.info("Bot state loaded successfully")
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")

    def shutdown(self):
        """Gracefully shutdown the bot"""
        try:
            self.running = False
            
            # Close all open positions
            for symbol, position in self.positions.items():
                try:
                    self.close_position(symbol)
                except Exception as e:
                    logging.error(f"Error closing position for {symbol}: {e}")
            
            # Close all WebSocket connections
            for symbol, ws in self.websocket_connections.items():
                try:
                    ws.close()
                except Exception as e:
                    logging.error(f"Error closing WebSocket for {symbol}: {e}")
            
            # Save final state
            self.save_state()
            
            # Send shutdown notification
            if self.telegram_bot and self.telegram_chat_id:
                try:
                    self.telegram_bot.send_message(
                        self.telegram_chat_id,
                        "Bot is shutting down. Final state saved."
                    )
                except Exception as e:
                    logging.error(f"Error sending Telegram shutdown message: {e}")
            
            # Shutdown MT5
            mt5.shutdown()
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            print("\nBot shutdown complete.")

    def run(self):
        """Main bot execution loop"""
        try:
            self.running = True
            logging.info("Bot started successfully")
            
            if self.telegram_bot and self.telegram_chat_id:
                self.telegram_bot.send_message(
                    self.telegram_chat_id,
                    "Bot started successfully"
                )
            
            while self.running:
                try:
                    # Get latest market data
                    prices = self.get_latest_prices()
                    
                    # Update performance metrics
                    self.update_performance_metrics(prices)
                    
                    # Check if it's time to save state
                    if (datetime.now() - self.last_state_save).total_seconds() > self.state_save_interval:
                        self.save_state()
                        self.last_state_save = datetime.now()
                    
                    # Sleep for a short interval
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(5)  # Wait longer on error
                    
        except Exception as e:
            logging.error(f"Critical error in bot execution: {e}")
        finally:
            self.shutdown()

if __name__ == "__main__":
    try:
        bot = AdvancedPremiumBot()
        bot.run()
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
        print(f"Error: {e}") 