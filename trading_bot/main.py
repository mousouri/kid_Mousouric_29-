import asyncio
import time
import os
from typing import Dict, Optional
import MetaTrader5 as mt5
from config import Config
from logger import Logger
from data_fetcher import DataFetcher
from strategy import Strategy
from risk_manager import RiskManager
from order_manager import OrderManager
from notifier import Notifier

class TradingBot:
    def __init__(self, config_path: str):
        # Get the absolute path to the config file
        config_path = os.path.abspath(config_path)
        
        # Initialize logger first
        self.config = Config(config_path)
        self.logger = Logger(self.config)
        self.notifier = Notifier(self.config)
        
        # Initialize MT5
        self.logger.info("Initializing MetaTrader 5...")
        if not mt5.initialize():
            error = mt5.last_error()
            self.logger.error(f"Failed to initialize MT5: {error}")
            self.notifier.notify_error(f"Failed to initialize MT5: {error}")
            raise Exception(f"MT5 initialization failed: {error}")
            
        # Print MT5 version and connection status
        self.logger.info(f"MT5 version: {mt5.version()}")
        terminal_info = mt5.terminal_info()
        self.logger.info(f"Connected to: {terminal_info.name}")
        self.logger.info(f"Connection status: {'Connected' if terminal_info.connected else 'Disconnected'}")
        
        # Initialize components
        self.logger.info("Initializing components...")
        self.data_fetcher = DataFetcher(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        self.strategy = Strategy(self.config, self.logger, self.risk_manager)
        self.order_manager = OrderManager(self.config, self.logger, self.risk_manager, self.data_fetcher)
        
        # Trading state
        self.is_running = False
        self.active_trades = {}
        
    async def start(self):
        """Start the trading bot"""
        try:
            self.logger.info("Starting trading bot...")
            self.notifier.notify_warning("Trading bot is starting up...")
            
            # Verify MT5 connection
            terminal_info = mt5.terminal_info()
            if not terminal_info.connected:
                self.logger.error("MT5 is not connected to the server")
                self.notifier.notify_error("MT5 is not connected to the server")
                raise Exception("MT5 connection failed")
                
            self.is_running = True
            self.logger.info("Trading bot started successfully")
            self.notifier.notify_warning("Trading bot started successfully")
            
            # Start main trading loop
            while self.is_running:
                try:
                    await self._trading_cycle()
                    await asyncio.sleep(self.config.trading_interval)
                except asyncio.CancelledError:
                    self.logger.info("Trading bot received shutdown signal")
                    self.notifier.notify_warning("Trading bot received shutdown signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in trading cycle: {e}")
                    self.notifier.notify_error(f"Error in trading cycle: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Fatal error in trading bot: {e}")
            self.notifier.notify_error(f"Fatal error in trading bot: {e}")
            self.stop()
            
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        self.notifier.notify_warning("Trading bot is shutting down...")
        self.is_running = False
        
        try:
            # Close all open positions
            positions = mt5.positions_get()
            if positions:
                self.logger.info(f"Closing {len(positions)} open positions...")
                self.notifier.notify_warning(f"Closing {len(positions)} open positions...")
                for position in positions:
                    asyncio.run(self.order_manager.close_position(position.ticket))
                    
            # Shutdown MT5
            self.logger.info("Shutting down MT5...")
            mt5.shutdown()
            self.logger.info("Trading bot stopped successfully")
            self.notifier.notify_warning("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.notifier.notify_error(f"Error during shutdown: {e}")
            
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Fetch market data
            market_data = await self.data_fetcher.fetch_market_data()
            if not market_data:
                self.logger.warning("No market data received")
                self.notifier.notify_warning("No market data received")
                return
                
            # Analyze market and generate signals
            for symbol in self.config.symbols:
                try:
                    # Get symbol data
                    symbol_data = market_data.get(symbol)
                    if not symbol_data:
                        self.logger.warning(f"No data for symbol {symbol}")
                        continue
                        
                    # Analyze market
                    analysis = self.strategy.analyze_market(symbol_data)
                    if analysis['signal'] == 0:
                        continue
                        
                    # Check if we already have an active trade
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        self.logger.info(f"Active position exists for {symbol}, skipping")
                        continue
                        
                    # Place order based on signal
                    side = "BUY" if analysis['signal'] == 1 else "SELL"
                    lot_size = self.risk_manager.calculate_position_size(
                        symbol,
                        analysis['current_price'],
                        self.config.max_risk_per_trade
                    )
                    
                    self.logger.info(f"Placing {side} order for {symbol} with lot size {lot_size}")
                    self.notifier.notify_trade_opened(
                        symbol,
                        side,
                        analysis['current_price']
                    )
                    
                    # Place the order
                    success = await self.order_manager.place_order(
                        symbol=symbol,
                        order_type=side,
                        lot_size=lot_size,
                        stop_loss=analysis['stop_loss'],
                        take_profit=analysis['take_profit']
                    )
                    
                    if not success:
                        self.logger.error(f"Failed to place order for {symbol}")
                        self.notifier.notify_error(f"Failed to place order for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    self.notifier.notify_error(f"Error processing {symbol}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            self.notifier.notify_error(f"Error in trading cycle: {e}")
            
async def main():
    """Main entry point"""
    try:
        # Get the absolute path to the config file
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        
        # Initialize and start the bot
        bot = TradingBot(config_path)
        await bot.start()
        
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        if 'bot' in locals():
            bot.stop()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}") 