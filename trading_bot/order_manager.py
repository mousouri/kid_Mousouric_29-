import MetaTrader5 as mt5
from typing import Dict, Optional
from logger import Logger
from risk_manager import RiskManager
from notifier import Notifier
import time
from functools import wraps
import asyncio

def with_error_handling(logger=None, notifier=None):
    """Decorator factory for consistent error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except mt5.MT5Error as e:
                self.logger.error(f"MT5 Error in {func.__name__}: {e}")
                self.notifier.notify_error(f"MT5 Error: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error in {func.__name__}: {e}")
                self.notifier.notify_error(f"System Error: {e}")
                return None
        return wrapper
    return decorator

class OrderManager:
    def __init__(self, config, logger: Logger, risk_manager: RiskManager, data_fetcher):
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        self.notifier = Notifier(config)
        self.data_fetcher = data_fetcher
        
    @with_error_handling()
    async def place_order(self, symbol: str, order_type: str, lot_size: float, 
                   stop_loss: float, take_profit: float) -> bool:
        """Place an order with enhanced price validation and retries"""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get current price
                price_data = await self.data_fetcher.get_current_price(symbol)
                if not price_data:
                    self.logger.error(f"Failed to get valid prices for {symbol}")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                # Validate prices
                if price_data['bid'] == 0 or price_data['ask'] == 0:
                    self.logger.error(f"Invalid prices for {symbol}")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                # Calculate entry price based on order type
                if order_type == "BUY":
                    entry_price = price_data['ask']
                    sl_price = entry_price - stop_loss
                    tp_price = entry_price + take_profit
                else:  # SELL
                    entry_price = price_data['bid']
                    sl_price = entry_price + stop_loss
                    tp_price = entry_price - take_profit
                    
                # Validate stop loss and take profit levels
                if not await self._validate_levels(symbol, entry_price, sl_price, tp_price):
                    self.logger.error(f"Invalid stop loss or take profit levels for {symbol}")
                    return False
                    
                # Prepare the order request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 10,
                    "magic": 234000,
                    "comment": "python script open",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                # Send the order
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"Order failed: {result.comment}")
                    if "No prices" in result.comment:
                        await asyncio.sleep(retry_delay)
                        continue
                    return False
                    
                self.logger.info(f"Order placed successfully: {result.comment}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error placing order for {symbol}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                continue
                
        self.logger.error(f"Failed to place order for {symbol} after {max_retries} attempts")
        return False
        
    @with_error_handling()
    async def _validate_levels(self, symbol: str, entry_price: float, 
                        stop_loss: float, take_profit: float) -> bool:
        """Validate stop loss and take profit levels"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
                
            # Get minimum stop level
            min_stop_level = symbol_info.trade_stops_level * symbol_info.point
            
            # Validate stop loss distance
            sl_distance = abs(entry_price - stop_loss)
            if sl_distance < min_stop_level:
                self.logger.error(f"Stop loss too close to entry price for {symbol}")
                return False
                
            # Validate take profit distance
            tp_distance = abs(entry_price - take_profit)
            if tp_distance < min_stop_level:
                self.logger.error(f"Take profit too close to entry price for {symbol}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating levels for {symbol}: {e}")
            return False
            
    @with_error_handling()
    async def close_position(self, ticket: int) -> Dict:
        """Close a position by ticket"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                self.notifier.notify_error(f"Position {ticket} not found")
                return {"status": "error", "message": "Position not found"}
                
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "python script close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to close position {ticket}: {result.comment}")
                self.notifier.notify_error(f"Failed to close position {ticket}: {result.comment}")
                return {"status": "error", "message": result.comment}
                
            # Calculate profit
            profit = position.profit
            self.logger.info(f"Position {ticket} closed successfully. Profit: {profit}")
            self.notifier.notify_trade_closed(
                position.symbol,
                "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL",
                result.price,
                profit
            )
            return {"status": "success", "order": result, "profit": profit}
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            self.notifier.notify_error(f"Error closing position {ticket}: {e}")
            return {"status": "error", "message": str(e)}
            
    @with_error_handling()
    async def modify_position(self, ticket: int, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Dict:
        """Modify position's stop loss and take profit"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                self.logger.error(f"Position {ticket} not found")
                self.notifier.notify_error(f"Position {ticket} not found")
                return {"status": "error", "message": "Position not found"}
                
            position = position[0]
            
            # Get symbol info
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {position.symbol} not found")
                self.notifier.notify_error(f"Symbol {position.symbol} not found")
                return {"status": "error", "message": "Symbol not found"}
                
            # Validate stop loss and take profit
            min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
            if min_stop_distance == 0:
                min_stop_distance = 10 * symbol_info.point  # Default to 10 points
                
            current_price = position.price_open
            if stop_loss is not None:
                if position.type == mt5.POSITION_TYPE_BUY:
                    if current_price - stop_loss < min_stop_distance:
                        stop_loss = current_price - min_stop_distance
                        self.logger.info(f"Adjusted stop loss to minimum distance for {position.symbol}")
                else:
                    if stop_loss - current_price < min_stop_distance:
                        stop_loss = current_price + min_stop_distance
                        self.logger.info(f"Adjusted stop loss to minimum distance for {position.symbol}")
                    
            if take_profit is not None:
                if position.type == mt5.POSITION_TYPE_BUY:
                    if take_profit - current_price < min_stop_distance:
                        take_profit = current_price + min_stop_distance
                        self.logger.info(f"Adjusted take profit to minimum distance for {position.symbol}")
                else:
                    if current_price - take_profit < min_stop_distance:
                        take_profit = current_price - min_stop_distance
                        self.logger.info(f"Adjusted take profit to minimum distance for {position.symbol}")
                
            # Prepare modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "sl": stop_loss if stop_loss is not None else position.sl,
                "tp": take_profit if take_profit is not None else position.tp,
                "position": position.ticket,
            }
            
            # Send modify order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Failed to modify position {ticket}: {result.comment}")
                self.notifier.notify_error(f"Failed to modify position {ticket}: {result.comment}")
                return {"status": "error", "message": result.comment}
                
            self.logger.info(f"Position {ticket} modified successfully")
            return {"status": "success", "order": result}
            
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {e}")
            self.notifier.notify_error(f"Error modifying position {ticket}: {e}")
            return {"status": "error", "message": str(e)} 