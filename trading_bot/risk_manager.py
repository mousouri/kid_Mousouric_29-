from typing import Dict, Optional, List
from logger import Logger
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.account_info = mt5.account_info()
        self.account_balance = self.account_info.balance
        self.available_margin = self.account_info.margin_free
        self.max_drawdown = 0.0
        self.daily_loss_limit = self.account_balance * 0.05  # 5% daily loss limit
        self.daily_loss = 0.0
        self.max_positions = 5  # Increased maximum positions
        self.last_reset_time = datetime.now()
        self._cache = {}  # Cache for calculations
        self._cache_timeout = 300  # 5 minutes cache timeout
        
    def _get_cached_data(self, key: str, func: callable, *args, **kwargs):
        """Get cached data or calculate and cache it"""
        current_time = time.time()
        if key in self._cache:
            data, timestamp = self._cache[key]
            if current_time - timestamp < self._cache_timeout:
                return data
                
        data = func(*args, **kwargs)
        self._cache[key] = (data, current_time)
        return data
        
    def _fetch_rates_with_retry(self, symbol: str, timeframe: int, count: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch rates with retry mechanism"""
        for attempt in range(max_retries):
            try:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
                if rates is not None:
                    return pd.DataFrame(rates)
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error fetching rates (attempt {attempt + 1}): {e}")
                time.sleep(1)
        return None
        
    def calculate_volatility(self, symbol: str, period: int = 14) -> float:
        """Calculate market volatility using ATR"""
        cache_key = f"volatility_{symbol}_{period}"
        return self._get_cached_data(cache_key, self._calculate_volatility, symbol, period)
        
    def _calculate_volatility(self, symbol: str, period: int) -> float:
        try:
            df = self._fetch_rates_with_retry(symbol, mt5.TIMEFRAME_H1, period + 1)
            if df is None:
                return 0.0
                
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
            
    def calculate_liquidity(self, symbol: str) -> float:
        """Calculate market liquidity based on spread and volume"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return 0.0
                
            spread = tick.ask - tick.bid
            volume = tick.volume
            
            # Normalize liquidity score (higher is better)
            liquidity_score = (1 / spread) * (volume / 1000000)
            return liquidity_score
        except Exception as e:
            self.logger.error(f"Error calculating liquidity: {e}")
            return 0.0
            
    def analyze_market_state(self, symbol: str) -> str:
        """Analyze current market state"""
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
            if rates is None:
                return "unknown"
                
            df = pd.DataFrame(rates)
            
            # Calculate trend indicators
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            price_sma20 = current_price / sma20.iloc[-1]
            price_sma50 = current_price / sma50.iloc[-1]
            
            # Determine market state
            if price_sma20 > 1.02 and price_sma50 > 1.02:
                return "strong_uptrend"
            elif price_sma20 < 0.98 and price_sma50 < 0.98:
                return "strong_downtrend"
            elif abs(price_sma20 - 1) < 0.02 and abs(price_sma50 - 1) < 0.02:
                return "range"
            else:
                return "trending"
        except Exception as e:
            self.logger.error(f"Error analyzing market state: {e}")
            return "unknown"
            
    def get_risk_multiplier(self, volatility: float, liquidity: float, market_state: str) -> float:
        """Calculate risk multiplier based on market conditions"""
        try:
            # Base multiplier
            multiplier = 1.0
            
            # Adjust for volatility (higher volatility = lower risk)
            if volatility > 0:
                volatility_factor = 1 / (1 + volatility)
                multiplier *= volatility_factor
                
            # Adjust for liquidity (higher liquidity = higher risk)
            if liquidity > 0:
                liquidity_factor = min(1.5, 1 + (liquidity / 10))
                multiplier *= liquidity_factor
                
            # Adjust for market state
            if market_state == "strong_uptrend":
                multiplier *= 1.2
            elif market_state == "strong_downtrend":
                multiplier *= 1.2
            elif market_state == "range":
                multiplier *= 0.8
                
            return max(0.5, min(1.5, multiplier))
        except Exception as e:
            self.logger.error(f"Error calculating risk multiplier: {e}")
            return 1.0
            
    def calculate_kelly_fraction(self, win_rate: float, risk_reward: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if win_rate <= 0 or risk_reward <= 0:
                return 0.5  # Default to half Kelly
            
            kelly = win_rate - ((1 - win_rate) / risk_reward)
            return max(0.1, min(0.5, kelly * 0.5))  # Use half Kelly for safety
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.5
            
    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        cache_key = f"correlation_{symbol1}_{symbol2}"
        return self._get_cached_data(cache_key, self._calculate_correlation, symbol1, symbol2)
        
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        try:
            # Get recent price data with same timeframe
            df1 = self._fetch_rates_with_retry(symbol1, mt5.TIMEFRAME_H1, 100)
            df2 = self._fetch_rates_with_retry(symbol2, mt5.TIMEFRAME_H1, 100)
            
            if df1 is None or df2 is None:
                return 0.0
                
            # Align timeframes
            df1 = df1.set_index('time')
            df2 = df2.set_index('time')
            aligned_df = pd.concat([df1['close'], df2['close']], axis=1, join='inner')
            
            # Calculate returns
            returns = np.log(aligned_df).diff().dropna()
            
            # Calculate correlation
            correlation = returns.corr().iloc[0, 1]
            return correlation
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
            
    def calculate_correlation_limit(self, current_positions: List[str], new_symbol: str) -> float:
        """Calculate position size limit based on portfolio correlation"""
        try:
            if not current_positions:
                return 1.0
                
            max_correlation = 0.0
            for symbol in current_positions:
                correlation = self.calculate_correlation(symbol, new_symbol)
                max_correlation = max(max_correlation, abs(correlation))
                
            # Reduce position size based on correlation
            correlation_factor = 1 - max_correlation
            return max(0.5, correlation_factor)
        except Exception as e:
            self.logger.error(f"Error calculating correlation limit: {e}")
            return 1.0
            
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size with enhanced risk management"""
        try:
            # Get market conditions
            volatility = self.calculate_volatility(symbol)
            liquidity = self.calculate_liquidity(symbol)
            market_state = self.analyze_market_state(symbol)
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Symbol {symbol} not found")
                return self._calculate_fallback_position_size(symbol, entry_price, stop_loss)
                
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss)
            point_value = symbol_info.point
            
            # Adjust risk based on market conditions
            base_risk = self.config.risk_per_trade
            adjusted_risk = base_risk * self.get_risk_multiplier(volatility, liquidity, market_state)
            
            # Calculate risk amount
            risk_amount = adjusted_risk * self.account_balance
            
            # Get historical performance
            win_rate = self.get_historical_win_rate(symbol)
            risk_reward = self.calculate_risk_reward_ratio(entry_price, stop_loss)
            
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_fraction(win_rate, risk_reward)
            
            # Calculate base position size
            position_size = (risk_amount * kelly_fraction) / (stop_distance / point_value)
            
            # Apply position limits
            current_positions = self.get_current_positions()
            position_limit = self.max_positions - len(current_positions)
            
            if position_limit <= 0:
                self.logger.warning("Maximum number of positions reached")
                return 0.0
                
            # Apply correlation limits
            correlation_limit = self.calculate_correlation_limit(current_positions, symbol)
            position_size *= correlation_limit
            
            # Validate against symbol limits
            position_size = min(position_size, symbol_info.volume_max)
            position_size = max(position_size, symbol_info.volume_min)
            
            # Round to allowed step
            step = symbol_info.volume_step
            position_size = round(position_size / step) * step
            
            # Ensure minimum position size
            if position_size < symbol_info.volume_min:
                self.logger.warning(f"Position size too small for {symbol}, using minimum size")
                return symbol_info.volume_min
                
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self._calculate_fallback_position_size(symbol, entry_price, stop_loss)
            
    def _calculate_fallback_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate fallback position size when main calculation fails"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
                
            # Use fixed risk amount (0.5% of account balance)
            risk_amount = self.account_balance * 0.005
            
            # Calculate stop distance
            stop_distance = abs(entry_price - stop_loss)
            point_value = symbol_info.point
            
            # Calculate simple position size
            position_size = risk_amount / (stop_distance / point_value)
            
            # Validate against symbol limits
            position_size = min(position_size, symbol_info.volume_max)
            position_size = max(position_size, symbol_info.volume_min)
            
            # Round to allowed step
            step = symbol_info.volume_step
            position_size = round(position_size / step) * step
            
            return position_size
        except Exception as e:
            self.logger.error(f"Error in fallback position size calculation: {e}")
            return symbol_info.volume_min if symbol_info else 0.01  # Return minimum lot size as last resort
            
    def calculate_stop_loss(self, symbol: str, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic stop loss based on market conditions"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
                
            # Get market conditions
            atr = self.calculate_volatility(symbol)
            trend_strength = self.calculate_trend_strength(symbol)
            
            # Dynamic stop loss calculation
            base_stop = entry_price * 0.02  # 2% base stop
            volatility_adjustment = atr * 2  # 2 ATRs
            trend_adjustment = self.get_trend_adjustment(trend_strength)
            
            # Combine factors
            stop_distance = base_stop * (1 + (atr / entry_price)) * trend_adjustment
            
            # Ensure minimum stop level
            min_stop = symbol_info.trade_stops_level * symbol_info.point
            stop_distance = max(stop_distance, min_stop)
            
            # Calculate stop loss price
            if is_long:
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
                
            # Round to correct digits
            stop_loss = round(stop_loss, symbol_info.digits)
            
            return stop_loss
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0
            
    def calculate_take_profit(self, symbol: str, entry_price: float, is_long: bool) -> float:
        """Calculate dynamic take profit based on market conditions"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
                
            # Get market conditions
            volatility = self.calculate_volatility(symbol)
            trend_strength = self.calculate_trend_strength(symbol)
            
            # Dynamic take profit calculation
            base_tp = entry_price * 0.04  # 4% base take profit
            volatility_adjustment = 1 + (volatility * 2)  # Adjust for volatility
            trend_adjustment = self.get_trend_adjustment(trend_strength)
            
            # Combine factors
            tp_distance = base_tp * volatility_adjustment * trend_adjustment
            
            # Calculate take profit price
            if is_long:
                take_profit = entry_price + tp_distance
            else:
                take_profit = entry_price - tp_distance
                
            # Round to correct digits
            take_profit = round(take_profit, symbol_info.digits)
            
            return take_profit
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return 0.0
            
    def validate_trade(self, symbol: str, entry_price: float, stop_loss: float, 
                      take_profit: float, position_size: float) -> bool:
        """Validate trade with enhanced risk checks"""
        try:
            # Check daily loss limit
            if self.daily_loss >= self.daily_loss_limit:
                self.logger.warning("Daily loss limit reached")
                return False
                
            # Check maximum drawdown
            current_drawdown = (self.account_balance - self.account_info.equity) / self.account_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                if current_drawdown > 0.05:  # Reduced to 5% max drawdown
                    self.logger.warning("Maximum drawdown limit reached")
                    return False
                    
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False
                
            # Check stop loss distance
            stop_distance = abs(entry_price - stop_loss)
            min_stop = symbol_info.trade_stops_level * symbol_info.point
            if stop_distance < min_stop:
                return False
                
            # Check margin requirements
            required_margin = self.calculate_required_margin(symbol, position_size)
            if required_margin > self.available_margin * 0.5:  # Use only 50% of available margin
                return False
                
            # Check maximum positions
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.max_positions:
                return False
                
            # Check portfolio correlation
            correlation_limit = self.calculate_correlation_limit(current_positions, symbol)
            if correlation_limit < 0.5:  # Too correlated with existing positions
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False
            
    def update_daily_loss(self, profit: float):
        """Update daily loss tracking"""
        try:
            # Reset daily loss at midnight
            current_time = datetime.now()
            if current_time.date() != self.last_reset_time.date():
                self.daily_loss = 0.0
                self.last_reset_time = current_time
                
            # Update daily loss
            if profit < 0:
                self.daily_loss += abs(profit)
        except Exception as e:
            self.logger.error(f"Error updating daily loss: {e}")
            
    def get_current_positions(self) -> List[str]:
        """Get list of currently open positions"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            return [pos.symbol for pos in positions]
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
            
    def calculate_required_margin(self, symbol: str, volume: float) -> float:
        """Calculate required margin for a position"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.0
                
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return 0.0
                
            # Calculate required margin
            required_margin = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY,
                symbol,
                volume,
                tick.ask
            )
            
            return required_margin
        except Exception as e:
            self.logger.error(f"Error calculating required margin: {e}")
            return 0.0
            
    def get_historical_win_rate(self, symbol: str) -> float:
        """Get historical win rate for a symbol"""
        try:
            # This would typically come from your trading history database
            # For now, return a default value
            return 0.55  # 55% win rate
        except Exception as e:
            self.logger.error(f"Error getting historical win rate: {e}")
            return 0.5
            
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float) -> float:
        """Calculate risk-reward ratio for a trade"""
        try:
            stop_distance = abs(entry_price - stop_loss)
            tp_distance = stop_distance * 2  # 1:2 risk-reward ratio
            return tp_distance / stop_distance
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return 2.0
            
    def calculate_trend_strength(self, symbol: str) -> float:
        """Calculate trend strength using ADX"""
        cache_key = f"trend_strength_{symbol}"
        return self._get_cached_data(cache_key, self._calculate_trend_strength, symbol)
        
    def _calculate_trend_strength(self, symbol: str) -> float:
        try:
            df = self._fetch_rates_with_retry(symbol, mt5.TIMEFRAME_H1, 50)
            if df is None:
                return 0.0
                
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate +DM and -DM
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            # Calculate TR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate +DI and -DI with dynamic period
            period = min(14, len(df) // 2)  # Dynamic period based on available data
            plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
            minus_di = 100 * (minus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period).mean()
            
            return adx.iloc[-1] / 100  # Normalize to 0-1 range
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
            
    def get_trend_adjustment(self, trend_strength: float) -> float:
        """Get adjustment factor based on trend strength"""
        try:
            if trend_strength > 0.7:  # Strong trend
                return 1.2
            elif trend_strength > 0.4:  # Moderate trend
                return 1.0
            else:  # Weak trend
                return 0.8
        except Exception as e:
            self.logger.error(f"Error getting trend adjustment: {e}")
            return 1.0 