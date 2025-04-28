import MetaTrader5 as mt5
from typing import Dict, Optional
from logger import Logger
import asyncio
import time
from functools import wraps

def with_error_handling(logger=None):
    """Decorator factory for consistent error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except mt5.MT5Error as e:
                self.logger.error(f"MT5 Error in {func.__name__}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator

class DataFetcher:
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        
    @with_error_handling()
    async def fetch_market_data(self) -> Dict:
        """Fetch market data from MT5"""
        market_data = {}
        
        for symbol in self.config.symbols:
            try:
                # Get symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.logger.error(f"Symbol {symbol} not found in MT5")
                    continue
                    
                # Check if market is open
                if not symbol_info.visible:
                    self.logger.warning(f"Symbol {symbol} is not visible")
                    continue
                    
                if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                    self.logger.warning(f"Symbol {symbol} is not available for trading")
                    continue
                    
                # Get current tick with retry
                tick = None
                for _ in range(3):  # Try 3 times
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None and tick.ask > 0 and tick.bid > 0:
                        break
                    await asyncio.sleep(1)  # Wait 1 second between retries
                    
                if tick is None or tick.ask == 0 or tick.bid == 0:
                    self.logger.error(f"Failed to get valid tick data for {symbol}")
                    continue
                    
                # Get historical data
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
                if rates is None:
                    self.logger.error(f"Failed to get historical data for {symbol}")
                    continue
                    
                # Prepare market data
                market_data[symbol] = {
                    'symbol': symbol,
                    'current_price': tick.ask,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': tick.ask - tick.bid,
                    'volume': tick.volume,
                    'time': tick.time,
                    'historical_data': rates
                }
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                continue
                
        return market_data
        
    @with_error_handling()
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Fetch order book data for a symbol"""
        try:
            # Get market depth data from MT5
            depth = mt5.market_book_get(symbol)
            if depth is None:
                self.logger.error(f"Failed to get market depth for {symbol}")
                return None
                
            # Convert to order book format
            order_book = {
                'bids': [(level.price, level.volume) for level in depth.bids],
                'asks': [(level.price, level.volume) for level in depth.asks]
            }
            return order_book
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
            
    @with_error_handling()
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch ticker data for a symbol"""
        try:
            # Get tick data from MT5
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Failed to get tick data for {symbol}")
                return None
                
            # Convert to ticker format
            ticker = {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': tick.time
            }
            return ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    @with_error_handling()
    async def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid and ask prices for a symbol with retries"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check if symbol is visible
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Symbol {symbol} is not visible in Market Watch")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                # Check if market is open
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    self.logger.error(f"Failed to get symbol info for {symbol}")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                if not symbol_info.visible:
                    self.logger.error(f"Symbol {symbol} is not visible")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    self.logger.error(f"Symbol {symbol} is not available for trading")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                # Get current tick
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    self.logger.error(f"Failed to get tick data for {symbol}")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                if tick.bid == 0 or tick.ask == 0:
                    self.logger.warning(f"Invalid prices for {symbol} (bid: {tick.bid}, ask: {tick.ask})")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': tick.time
                }
                
            except Exception as e:
                self.logger.error(f"Error getting current price for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue
                
        self.logger.error(f"Failed to get valid prices for {symbol} after {max_retries} attempts")
        return None 