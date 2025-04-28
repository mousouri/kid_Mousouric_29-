import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import json
import MetaTrader5 as mt5

class Config:
    def __init__(self, config_path: str):
        """Initialize configuration from JSON file"""
        try:
            # Load environment variables
            load_dotenv()
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Trading parameters
            self.symbols: List[str] = config_data.get('symbols', [])
            self.timeframe: int = config_data.get('timeframe', mt5.TIMEFRAME_M1)
            self.max_positions: int = config_data.get('max_positions', 5)
            self.trading_hours: Dict[str, Dict[str, str]] = config_data.get('trading_hours', {})
            self.trading_interval: int = config_data.get('trading_interval', 60)  # Default 60 seconds
            
            # Risk management parameters
            self.risk_per_trade: float = config_data.get('risk_per_trade', 0.01)
            self.max_risk_per_trade: float = config_data.get('max_risk_per_trade', 0.02)  # Maximum 2% risk per trade
            self.max_daily_loss: float = config_data.get('max_daily_loss', 0.05)
            self.max_drawdown: float = config_data.get('max_drawdown', 0.05)
            self.min_stop_distance: float = config_data.get('min_stop_distance', 10)
            self.risk_reward_ratio: float = config_data.get('risk_reward_ratio', 2.0)
            self.use_kelly: bool = config_data.get('use_kelly', True)
            self.kelly_fraction: float = config_data.get('kelly_fraction', 0.5)
            self.max_correlation: float = config_data.get('max_correlation', 0.7)
            
            # Strategy parameters
            self.rsi_period: int = config_data.get('rsi_period', 14)
            self.rsi_overbought: int = config_data.get('rsi_overbought', 70)
            self.rsi_oversold: int = config_data.get('rsi_oversold', 30)
            self.macd_fast: int = config_data.get('macd_fast', 12)
            self.macd_slow: int = config_data.get('macd_slow', 26)
            self.macd_signal: int = config_data.get('macd_signal', 9)
            self.bb_period: int = config_data.get('bb_period', 20)
            self.bb_std: float = config_data.get('bb_std', 2.0)
            
            # Logging
            self.log_level: str = config_data.get('log_level', 'INFO')
            self.log_file: str = config_data.get('log_file', 'trading_bot.log')
            
            # Telegram configuration
            self.telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
            self.telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')
            
        except FileNotFoundError:
            raise Exception(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in config file: {config_path}")
        except Exception as e:
            raise Exception(f"Error loading config: {str(e)}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'max_positions': self.max_positions,
            'trading_hours': self.trading_hours,
            'trading_interval': self.trading_interval,
            'risk_per_trade': self.risk_per_trade,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'min_stop_distance': self.min_stop_distance,
            'risk_reward_ratio': self.risk_reward_ratio,
            'use_kelly': self.use_kelly,
            'kelly_fraction': self.kelly_fraction,
            'max_correlation': self.max_correlation,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'telegram_token': self.telegram_token,
            'telegram_chat_id': self.telegram_chat_id
        }

    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.symbols:
            raise ValueError("No trading symbols configured")
        if not all(key in self.trading_hours for key in self.symbols):
            raise ValueError("Missing trading hours for some symbols")
        if not 0 < self.risk_per_trade <= self.max_risk_per_trade:
            raise ValueError(f"Risk per trade must be between 0 and {self.max_risk_per_trade}")
        if not 0 < self.max_daily_loss <= 0.1:
            raise ValueError("Max daily loss must be between 0 and 10%")
        if not 0 < self.max_drawdown <= 0.1:
            raise ValueError("Max drawdown must be between 0 and 10%")
        if not self.min_stop_distance > 0:
            raise ValueError("Min stop distance must be positive")
        if not self.risk_reward_ratio >= 1:
            raise ValueError("Risk:Reward ratio must be at least 1")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("Kelly fraction must be between 0 and 1")
        if not 0 <= self.max_correlation <= 1:
            raise ValueError("Max correlation must be between 0 and 1")
        if not self.rsi_period > 0:
            raise ValueError("RSI period must be positive")
        if not 0 < self.rsi_oversold < self.rsi_overbought < 100:
            raise ValueError("Invalid RSI levels")
        if not (self.macd_fast < self.macd_slow and 
               self.macd_signal > 0):
            raise ValueError("Invalid MACD parameters")
        if not (self.bb_period > 0 and 
               self.bb_std > 0):
            raise ValueError("Invalid Bollinger Bands parameters")
        if not self.trading_interval > 0:
            raise ValueError("Trading interval must be positive")
        return True 