from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from config import Config
from logger import Logger
from risk_manager import RiskManager

class Strategy:
    def __init__(self, config: Config, logger: Logger, risk_manager: RiskManager):
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        
    def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data and generate trading signals"""
        try:
            # Convert MT5 data to DataFrame
            df = self._convert_to_dataframe(market_data['historical_data'])
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Generate signals
            signals = self._generate_signals(df, indicators)
            
            # Calculate entry, stop loss, and take profit levels
            levels = self._calculate_levels(df, signals)
            
            return {
                'signal': signals['signal'],
                'current_price': market_data['current_price'],
                'entry_price': levels['entry'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {e}")
            return {
                'signal': 0,
                'current_price': market_data['current_price'],
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None
            }
            
    def _convert_to_dataframe(self, rates) -> pd.DataFrame:
        """Convert MT5 rates to pandas DataFrame"""
        try:
            # Create DataFrame from rates
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match standard OHLCV format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            raise
            
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            sma = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'signal': signal,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'sma': sma
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise
            
    def _generate_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Generate trading signals based on indicators"""
        try:
            # Get the latest values
            current_rsi = indicators['rsi'].iloc[-1]
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['signal'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            current_upper = indicators['upper_band'].iloc[-1]
            current_lower = indicators['lower_band'].iloc[-1]
            
            # Initialize signal
            signal = 0  # 0 = no signal, 1 = buy, -1 = sell
            
            # RSI signals
            if current_rsi < 30:  # Oversold
                signal = 1
            elif current_rsi > 70:  # Overbought
                signal = -1
                
            # MACD signals
            if current_macd > current_signal and signal == 0:
                signal = 1
            elif current_macd < current_signal and signal == 0:
                signal = -1
                
            # Bollinger Bands signals
            if current_price < current_lower and signal == 0:
                signal = 1
            elif current_price > current_upper and signal == 0:
                signal = -1
                
            return {'signal': signal}
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            raise
            
    def _calculate_levels(self, df: pd.DataFrame, signals: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            current_price = df['Close'].iloc[-1]
            
            if signals['signal'] == 1:  # Buy signal
                entry = current_price
                # Calculate stop loss using ATR
                atr = self._calculate_atr(df)
                stop_loss = entry - (atr * 2)  # 2 ATRs for stop loss
                take_profit = entry + (atr * 6)  # 6 ATRs for take profit (3:1 risk-reward)
            elif signals['signal'] == -1:  # Sell signal
                entry = current_price
                # Calculate stop loss using ATR
                atr = self._calculate_atr(df)
                stop_loss = entry + (atr * 2)  # 2 ATRs for stop loss
                take_profit = entry - (atr * 6)  # 6 ATRs for take profit (3:1 risk-reward)
            else:
                entry = None
                stop_loss = None
                take_profit = None
                
            # Round to proper decimal places
            if entry is not None:
                entry = round(entry, 5)
            if stop_loss is not None:
                stop_loss = round(stop_loss, 5)
            if take_profit is not None:
                take_profit = round(take_profit, 5)
                
            self.logger.info(f"Calculated levels - Entry: {entry}, SL: {stop_loss}, TP: {take_profit}")
            return {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating levels: {e}")
            raise
            
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
            
    def calculate_stop_loss(self, data: pd.DataFrame, signal: int) -> float:
        """Calculate stop loss level"""
        try:
            if signal == 1:  # Long position
                return data['BB_lower'].iloc[-1]
            elif signal == -1:  # Short position
                return data['BB_upper'].iloc[-1]
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0
            
    def calculate_take_profit(self, data: pd.DataFrame, signal: int, stop_loss: float) -> float:
        """Calculate take profit level"""
        try:
            current_price = data['close'].iloc[-1]
            if signal == 1:  # Long position
                risk = current_price - stop_loss
                return current_price + (risk * self.config.risk_reward_ratio)
            elif signal == -1:  # Short position
                risk = stop_loss - current_price
                return current_price - (risk * self.config.risk_reward_ratio)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return 0.0 