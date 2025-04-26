import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
from dotenv import load_dotenv
import ta
import threading
import queue
import sys
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import websocket
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from transformers import pipeline

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_premium_bot.log'),
        logging.StreamHandler()
    ]
)

class LSTMPredictor:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 20
        self.patience = 5  # Early stopping patience
        
    def create_sequences(self, data):
        """Create sequences with multiple features"""
        X, y = [], []
        for i in range(len(data)-self.window_size-1):
            # Include multiple features: close price, volume, and technical indicators
            features = np.column_stack((
                data[i:(i+self.window_size), 0],  # Close price
                data[i:(i+self.window_size), 1],  # Volume
                data[i:(i+self.window_size), 2],  # RSI
                data[i:(i+self.window_size), 3]   # MACD
            ))
            X.append(features)
            y.append(data[i+self.window_size, 0])  # Next close price
        return np.array(X), np.array(y)
        
    def build_model(self, input_size):
        """Build enhanced LSTM model with multiple layers and dropout"""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=100, num_layers=3):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # LSTM layers
                self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, num_layers//2, batch_first=True, dropout=0.2)
                
                # Fully connected layers
                self.fc1 = nn.Linear(hidden_size//2, hidden_size//4)
                self.fc2 = nn.Linear(hidden_size//4, 1)
                
                # Dropout layers
                self.dropout = nn.Dropout(0.3)
                
                # Batch normalization
                self.bn1 = nn.BatchNorm1d(hidden_size//4)
                
            def forward(self, x):
                # First LSTM layer
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                out, _ = self.lstm1(x, (h0, c0))
                
                # Second LSTM layer
                h1 = torch.zeros(self.num_layers//2, x.size(0), self.hidden_size//2).to(self.device)
                c1 = torch.zeros(self.num_layers//2, x.size(0), self.hidden_size//2).to(self.device)
                out, _ = self.lstm2(out, (h1, c1))
                
                # Get the last time step
                out = out[:, -1, :]
                
                # Fully connected layers with dropout and batch norm
                out = self.dropout(out)
                out = self.fc1(out)
                out = self.bn1(out)
                out = torch.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                
                return out
                
        return LSTMModel(input_size)
        
    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for LSTM training"""
        try:
            # Select relevant features
            features = df[['close', 'volume', 'rsi', 'macd']].values
            
            # Scale the features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None
            
    def train(self, df: pd.DataFrame) -> bool:
        """Train the LSTM model with early stopping"""
        try:
            # Prepare data
            data = self.prepare_data(df)
            if data is None:
                return False
                
            # Create sequences
            X, y = self.create_sequences(data)
            if len(X) == 0 or len(y) == 0:
                logging.error("Failed to create sequences")
                return False
                
            # Convert to tensors
            X = torch.FloatTensor(X).to(self.device)
            y = torch.FloatTensor(y).to(self.device)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Build and initialize model
            self.model = self.build_model(X.shape[2]).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Early stopping variables
            best_loss = float('inf')
            patience_counter = 0
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                avg_epoch_loss = epoch_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")
                
                # Early stopping check
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logging.info("Early stopping triggered")
                        break
                        
            self.is_trained = True
            logging.info("LSTM model trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training LSTM model: {e}")
            return False
            
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """Make prediction using the trained LSTM model"""
        if not self.is_trained:
            return None
            
        try:
            # Prepare data
            data = self.prepare_data(df)
            if data is None:
                return None
                
            # Get the last sequence
            last_sequence = data[-self.window_size:]
            if len(last_sequence) < self.window_size:
                logging.error("Insufficient data for prediction")
                return None
                
            # Convert to tensor
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X)
                # Inverse transform the prediction
                dummy_array = np.zeros((1, 4))  # 4 features
                dummy_array[0, 0] = prediction.cpu().numpy()[0, 0]
                predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            return predicted_price
            
        except Exception as e:
            logging.error(f"Error making LSTM prediction: {e}")
            return None

class NewsSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = None
        self.last_update = None
        self.sentiment_cache = {}
        self.cache_duration = 3600  # Cache for 1 hour
        
    def initialize(self):
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_pipeline = pipeline("text-classification", model="yiyanghkust/finbert-tone")
            return True
        except Exception as e:
            logging.error(f"Error initializing sentiment analyzer: {e}")
            return False
            
    def get_news_sentiment(self, symbol: str) -> float:
        try:
            current_time = time.time()
            
            # Check cache
            if (symbol in self.sentiment_cache and 
                self.last_update and 
                current_time - self.last_update < self.cache_duration):
                return self.sentiment_cache[symbol]
                
            # For now, return neutral sentiment (0.0)
            # In the future, this can be replaced with actual news sentiment analysis
            sentiment = 0.0
            
            # Update cache
            self.sentiment_cache[symbol] = sentiment
            self.last_update = current_time
            
            return sentiment
            
        except Exception as e:
            logging.error(f"Error getting news sentiment: {e}")
            return 0.0

class EconomicCalendar:
    def __init__(self):
        self.calendar_cache = {}
        self.last_update = None
        self.cache_duration = 3600  # Cache for 1 hour
        
    def get_economic_events(self, symbol: str) -> List[Dict]:
        try:
            current_time = time.time()
            
            # Check cache
            if (symbol in self.calendar_cache and 
                self.last_update and 
                current_time - self.last_update < self.cache_duration):
                return self.calendar_cache[symbol]
                
            # For now, return empty list
            # In the future, this can be replaced with actual economic calendar data
            events = []
            
            # Update cache
            self.calendar_cache[symbol] = events
            self.last_update = current_time
            
            return events
            
        except Exception as e:
            logging.error(f"Error getting economic calendar: {e}")
            return []

class MarketCorrelationAnalyzer:
    def __init__(self):
        self.correlation_matrix = {}
        self.last_update = None
        self.cache_duration = 3600  # Cache for 1 hour
        
    def calculate_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        try:
            if df1.empty or df2.empty:
                return 0.0
                
            # Align the dataframes
            aligned_df1, aligned_df2 = df1.align(df2, join='inner')
            
            if aligned_df1.empty or aligned_df2.empty:
                return 0.0
                
            # Calculate correlation
            correlation = aligned_df1['close'].corr(aligned_df2['close'])
            return correlation if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {e}")
            return 0.0
            
    def update_correlations(self, symbols: List[str], timeframe: int):
        try:
            current_time = time.time()
            
            # Check cache
            if (self.last_update and 
                current_time - self.last_update < self.cache_duration):
                return self.correlation_matrix
                
            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
                if rates is not None and len(rates) > 0:
                    data[symbol] = pd.DataFrame(rates)
                    
            # Calculate correlations
            correlations = {}
            for sym1 in symbols:
                for sym2 in symbols:
                    if sym1 != sym2 and sym1 in data and sym2 in data:
                        corr = self.calculate_correlation(data[sym1], data[sym2])
                        correlations[f"{sym1}_{sym2}"] = corr
                        
            # Update cache
            self.correlation_matrix = correlations
            self.last_update = current_time
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error updating correlations: {e}")
            return {}

class RiskManager:
    def __init__(self):
        self.max_daily_drawdown = 0.02  # 2% max daily drawdown
        self.max_total_drawdown = 0.1   # 10% max total drawdown
        self.volatility_threshold = 0.015  # 1.5% volatility threshold
        self.correlation_threshold = 0.7
        self.time_based_risk = {
            'high_risk_hours': [(14, 16)],  # High volatility hours (UTC)
            'news_hours': [(13, 15), (19, 21)]  # Major news release hours
        }
        self.risk_multipliers = {
            'high_volatility': 0.5,
            'high_correlation': 0.7,
            'high_risk_hours': 0.6,
            'news_hours': 0.4
        }
        self.position_history = []
        self.performance_history = []
        self.initial_equity = None
        self.max_equity = None
        
    def _round_position_size(self, size: float, symbol_info: Any) -> float:
        """Round position size to valid lot size"""
        try:
            if symbol_info is None:
                return 0.0
                
            # Round to nearest valid step
            size = round(size / symbol_info.volume_step) * symbol_info.volume_step
            
            # Ensure within min/max limits
            size = max(symbol_info.volume_min, min(size, symbol_info.volume_max))
            
            return size
        except Exception as e:
            logging.error(f"Error rounding position size: {e}")
            return 0.0
            
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate normalized volatility"""
        try:
            if df is None or df.empty:
                return 0.0
                
            returns = df['close'].pct_change()
            if len(returns) < window:
                return 0.0
                
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            return volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.0
        except Exception as e:
            logging.error(f"Error calculating volatility: {e}")
            return 0.0
            
    def check_time_based_risk(self) -> float:
        """Check if current time is in high-risk periods"""
        try:
            current_hour = datetime.utcnow().hour
            risk_multiplier = 1.0
            
            # Check high-risk hours
            for start, end in self.time_based_risk['high_risk_hours']:
                if start <= current_hour <= end:
                    risk_multiplier *= self.risk_multipliers['high_risk_hours']
                    logging.info(f"High-risk hours detected: {start}:00-{end}:00 UTC")
                    
            # Check news hours
            for start, end in self.time_based_risk['news_hours']:
                if start <= current_hour <= end:
                    risk_multiplier *= self.risk_multipliers['news_hours']
                    logging.info(f"News hours detected: {start}:00-{end}:00 UTC")
                    
            return risk_multiplier
        except Exception as e:
            logging.error(f"Error checking time-based risk: {e}")
            return 1.0
            
    def calculate_position_size(self, 
                              symbol: str, 
                              current_price: float, 
                              stop_loss: float, 
                              account_balance: float,
                              volatility: float,
                              correlation_factor: float) -> float:
        """Calculate position size based on multiple risk factors"""
        try:
            if account_balance <= 0:
                logging.error("Invalid account balance")
                return 0.0
                
            # Base position size (1% of account balance)
            base_size = account_balance * 0.01
            
            # Volatility adjustment
            if volatility > self.volatility_threshold:
                base_size *= self.risk_multipliers['high_volatility']
                logging.info(f"Volatility adjustment applied: {volatility:.2%}")
                
            # Correlation adjustment
            if correlation_factor < 1.0:
                base_size *= correlation_factor
                logging.info(f"Correlation adjustment applied: {correlation_factor:.2%}")
                
            # Time-based risk adjustment
            time_risk = self.check_time_based_risk()
            if time_risk < 1.0:
                base_size *= time_risk
                logging.info(f"Time-based risk adjustment applied: {time_risk:.2%}")
                
            # Convert to lots
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            pip_value = symbol_info.trade_tick_value * (abs(current_price - stop_loss) / symbol_info.trade_tick_size)
            if pip_value <= 0:
                logging.error("Invalid pip value calculation")
                return 0.0
                
            position_size = base_size / pip_value
            
            # Ensure position size is within limits
            position_size = self._round_position_size(position_size, symbol_info)
            
            logging.info(f"Calculated position size: {position_size:.2f} lots")
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.0
            
    def check_drawdown_limits(self, current_equity: float, initial_equity: float) -> bool:
        """Check if drawdown limits are exceeded"""
        try:
            if current_equity <= 0 or initial_equity <= 0:
                logging.error("Invalid equity values")
                return False
                
            # Initialize max equity if not set
            if self.max_equity is None:
                self.max_equity = initial_equity
            else:
                self.max_equity = max(self.max_equity, current_equity)
                
            # Calculate current drawdown
            current_drawdown = (self.max_equity - current_equity) / self.max_equity
            
            # Check daily drawdown
            today = datetime.now().date()
            daily_trades = [t for t in self.position_history 
                          if datetime.fromtimestamp(t['time']).date() == today]
            
            if daily_trades:
                daily_start_equity = daily_trades[0]['balance']
                daily_drawdown = (daily_start_equity - current_equity) / daily_start_equity
                if daily_drawdown > self.max_daily_drawdown:
                    logging.warning(f"Daily drawdown limit exceeded: {daily_drawdown:.2%}")
                    return False
                    
            # Check total drawdown
            if current_drawdown > self.max_total_drawdown:
                logging.warning(f"Total drawdown limit exceeded: {current_drawdown:.2%}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking drawdown limits: {e}")
            return False
            
    def update_performance_history(self, trade_result: Dict):
        """Update performance history with new trade result"""
        try:
            if not trade_result:
                logging.error("Invalid trade result")
                return
                
            self.position_history.append(trade_result)
            
            # Calculate performance metrics
            if len(self.position_history) >= 2:
                last_trade = self.position_history[-1]
                prev_trade = self.position_history[-2]
                
                try:
                    performance = {
                        'time': last_trade['time'],
                        'profit': last_trade['profit'],
                        'drawdown': (prev_trade['balance'] - last_trade['balance']) / prev_trade['balance'],
                        'risk_reward': abs(last_trade['profit'] / (last_trade['sl'] - last_trade['price'])),
                        'volatility': self.calculate_volatility(pd.DataFrame(last_trade['price_data']))
                    }
                    
                    self.performance_history.append(performance)
                    
                    # Log performance metrics
                    logging.info(f"Trade Performance: Profit={performance['profit']:.2f}, "
                               f"Drawdown={performance['drawdown']:.2%}, "
                               f"Risk/Reward={performance['risk_reward']:.2f}")
                               
                except Exception as e:
                    logging.error(f"Error calculating performance metrics: {e}")
                    
        except Exception as e:
            logging.error(f"Error updating performance history: {e}")

class AdvancedPremiumBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # MT5 credentials
        try:
            self.account = int(os.getenv('MT5_ACCOUNT'))
            self.password = os.getenv('MT5_PASSWORD')
            self.server = os.getenv('MT5_SERVER')
            
            if not all([self.account, self.password, self.server]):
                raise ValueError("Missing MT5 credentials in .env file")
        except ValueError as e:
            logging.error(f"Error loading MT5 credentials: {e}")
            raise
            
        # Telegram credentials
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Trading parameters
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
        self.timeframes = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
        # Risk parameters
        self.lot_size = 0.01
        self.trailing_stop = 35
        self.take_profit = 750
        self.stop_loss = 25
        self.max_positions = 5
        
        # Initialize state variables
        self.initialized = False
        self.running = True
        self.data_queue = queue.Queue()
        self.trade_history = []  # Add trade history list
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'average_profit': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'risk_reward_ratio': 0.0,
            'daily_trades': 0,
            'daily_profit': 0.0,
            'last_trade_date': None
        }
        
        # Initialize ML models
        self.ml_models = {
            'trend': None,
            'regime': None,
            'volatility': None
        }
        self.ml_scalers = {
            'trend': StandardScaler(),
            'regime': StandardScaler(),
            'volatility': StandardScaler()
        }
        
        # Initialize market conditions
        self.market_conditions = {}
        self.correlation_matrix = {}
        
        # Initialize position sizing methods
        self.position_sizing_methods = {
            'fixed': self._fixed_position_size,
            'kelly': self._kelly_position_size,
            'dynamic': self._dynamic_position_size,
            'compound': self._compound_position_size
        }
        
        # Set default position sizing method
        self.position_sizing_method = "dynamic"
        
        # Initialize LSTM predictor
        self.lstm_predictor = LSTMPredictor()
        
        # Add LSTM prediction to market conditions
        self.market_conditions['lstm_prediction'] = None
        
        # Initialize news sentiment analyzer
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.sentiment_analyzer.initialize()
        
        # Add sentiment to market conditions
        self.market_conditions['news_sentiment'] = {}
        
        # Initialize new components
        self.economic_calendar = EconomicCalendar()
        self.correlation_analyzer = MarketCorrelationAnalyzer()
        
        # Add new market conditions
        self.market_conditions['economic_events'] = {}
        self.market_conditions['correlations'] = {}
        
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
    async def initialize_telegram(self):
        """Initialize Telegram bot for notifications"""
        if self.telegram_token and self.telegram_chat_id:
            try:
                self.telegram_app = Application.builder().token(self.telegram_token).build()
                await self.telegram_app.initialize()
                logging.info("Telegram bot initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Telegram bot: {e}")
        else:
            logging.warning("Telegram credentials not found. Notifications will be disabled.")
            
    async def send_telegram_message(self, message: str):
        """Send message via Telegram"""
        if hasattr(self, 'telegram_app') and self.telegram_chat_id:
            try:
                await self.telegram_app.bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML'
                )
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")
                
    def initialize(self):
        """Initialize MT5 connection and verify account"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                logging.error(f"Failed to initialize MT5: {error}")
                return False
                
            if not mt5.login(self.account, password=self.password, server=self.server):
                error = mt5.last_error()
                logging.error(f"Failed to login to MT5: {error}")
                return False
                
            account_info = mt5.account_info()
            if account_info is None:
                error = mt5.last_error()
                logging.error(f"Failed to get account info: {error}")
                return False
                
            # Verify account is enabled for trading
            if not account_info.trade_allowed:
                logging.error("Trading is not allowed for this account")
                return False
                
            # Verify account has sufficient margin
            if account_info.margin_free < 100:  # Minimum 100 USD free margin
                logging.error("Insufficient free margin for trading")
                return False
                
            logging.info(f"Successfully connected to MT5!")
            logging.info(f"Account: {self.account}")
            logging.info(f"Server: {self.server}")
            logging.info(f"Balance: {account_info.balance:.2f}")
            logging.info(f"Equity: {account_info.equity:.2f}")
            logging.info(f"Margin: {account_info.margin:.2f}")
            logging.info(f"Free Margin: {account_info.margin_free:.2f}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            return False
        
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators using ta package"""
        try:
            # Make a copy of the dataframe to avoid modifying the original
            df = df.copy()
            
            # Trend indicators
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_hist'] = ta.trend.macd_diff(df['close'])
            
            # Volatility indicators
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Volume indicators
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
                df['volume_ma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
            else:
                df['obv'] = 0
                df['mfi'] = 50
                df['volume_ratio'] = 1
                
            # Custom indicators
            df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['atr']
            df['volatility_ratio'] = df['atr'] / df['close'] * 100
            df['price_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
            
            # ADX
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            df['di_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
            df['di_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
            
            # Trend detection
            df['trend_direction'] = np.where(
                (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']), 1,
                np.where((df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']), -1, 0)
            )
            
            # Momentum detection
            df['momentum_strength'] = np.where(
                (df['rsi'] > 70) & (df['macd'] > 0), 1,
                np.where((df['rsi'] < 30) & (df['macd'] < 0), -1, 0)
            )
            
            # Fill NaN values with appropriate defaults
            df = df.fillna({
                'ema_9': df['close'],
                'ema_21': df['close'],
                'ema_50': df['close'],
                'ema_200': df['close'],
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'macd_hist': 0,
                'bb_upper': df['close'],
                'bb_middle': df['close'],
                'bb_lower': df['close'],
                'bb_width': 0,
                'atr': 0,
                'trend_strength': 0,
                'volatility_ratio': 0,
                'price_momentum': 0,
                'adx': 0,
                'di_plus': 0,
                'di_minus': 0,
                'trend_direction': 0,
                'momentum_strength': 0
            })
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return None
        
    def update_correlation_matrix(self):
        """Update correlation matrix for all currency pairs"""
        for symbol1 in self.symbols:
            for symbol2 in self.symbols:
                if symbol1 != symbol2:
                    # Get recent price data
                    rates1 = mt5.copy_rates_from_pos(symbol1, mt5.TIMEFRAME_H1, 0, self.correlation_window)
                    rates2 = mt5.copy_rates_from_pos(symbol2, mt5.TIMEFRAME_H1, 0, self.correlation_window)
                    
                    if rates1 is not None and rates2 is not None:
                        df1 = pd.DataFrame(rates1)
                        df2 = pd.DataFrame(rates2)
                        
                        # Calculate correlation
                        correlation = df1['close'].corr(df2['close'])
                        self.correlation_matrix[f"{symbol1}_{symbol2}"] = correlation
                        
    def check_news_events(self, symbol: str) -> bool:
        """Check if there are any high-impact news events for the symbol"""
        current_time = datetime.now()
        
        # Update news cache if needed
        if (self.last_news_update is None or 
            (current_time - self.last_news_update).total_seconds() > 3600):
            # Here you would implement actual news API call
            # For now, we'll use a placeholder
            self.news_cache = self._get_news_events()
            self.last_news_update = current_time
            
        # Check for news events
        for event in self.news_cache.get(symbol, []):
            event_time = event['time']
            if abs((current_time - event_time).total_seconds()) <= self.news_buffer_minutes * 60:
                if event['impact'] >= self.news_impact_threshold:
                    return True
        return False
        
    def _get_news_events(self) -> Dict:
        """Placeholder for news API integration"""
        # In a real implementation, this would call a news API
        return {}
        
    def prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for machine learning"""
        features = pd.DataFrame()
        
        # Price action features
        features['returns'] = df['close'].pct_change()
        features['volatility'] = df['returns'].rolling(window=20).std()
        
        # Technical indicator features
        for feature in self.ml_features:
            if feature in df.columns:
                features[feature] = df[feature]
                
        # Market regime features
        features['trend_regime'] = np.where(
            df['adx'] > self.strong_trend_threshold, 1,
            np.where(df['bb_width'] < df['bb_width'].mean() * self.low_volatility_threshold, 2, 0)
        )
        
        # Volatility regime features
        features['volatility_regime'] = np.where(
            df['atr'] / df['close'] > self.market_volatility_threshold, 1, 0
        )
        
        return features.dropna()
        
    def train_ml_models(self, df: pd.DataFrame):
        """Train multiple ML models for different aspects of trading"""
        features = self.prepare_ml_features(df)
        
        # Prepare targets
        trend_target = (df['close'].shift(-1) > df['close']).astype(int)
        regime_target = features['trend_regime']
        volatility_target = features['volatility_regime']
        
        # Split data
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        
        # Train trend model
        y_trend = trend_target[:train_size]
        X_trend_scaled = self.ml_scalers['trend'].fit_transform(X_train)
        self.ml_models['trend'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_models['trend'].fit(X_trend_scaled, y_trend)
        
        # Train regime model
        y_regime = regime_target[:train_size]
        X_regime_scaled = self.ml_scalers['regime'].fit_transform(X_train)
        self.ml_models['regime'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_models['regime'].fit(X_regime_scaled, y_regime)
        
        # Train volatility model
        y_volatility = volatility_target[:train_size]
        X_volatility_scaled = self.ml_scalers['volatility'].fit_transform(X_train)
        self.ml_models['volatility'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_models['volatility'].fit(X_volatility_scaled, y_volatility)
        
    def _fixed_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Fixed position sizing method"""
        return self.lot_size
        
    def _kelly_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Kelly Criterion position sizing"""
        win_rate = max(0.5, self.performance_stats['win_rate'])
        avg_win = max(1, self.performance_stats['average_profit'])
        avg_loss = max(1, abs(self.performance_stats['average_loss']))
        
        if avg_loss == 0:
            return self.lot_size
            
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.3))
        
        account_info = mt5.account_info()
        if account_info is None:
            return self.lot_size
            
        balance = account_info.balance
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return self.lot_size
            
        pip_value = symbol_info.trade_tick_value * (stop_loss_pips / symbol_info.trade_tick_size)
        position_size = balance * kelly_fraction / pip_value
        
        return self._round_position_size(position_size, symbol_info)
        
    def _dynamic_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Dynamic position sizing based on market conditions"""
        base_size = self._kelly_position_size(symbol, stop_loss_pips)
        
        # Adjust based on market conditions
        volatility_factor = min(1.5, max(0.5, self.market_conditions.get('volatility', 0.5) / self.market_volatility_threshold))
        trend_factor = min(1.3, max(0.7, self.market_conditions.get('trend_strength', 1.0)))
        adx_factor = min(1.2, max(0.8, self.market_conditions.get('adx_strength', 20) / self.strong_trend_threshold))
        
        return base_size * volatility_factor * trend_factor * adx_factor
        
    def _compound_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Compound interest based position sizing"""
        if self.initial_equity is None:
            account_info = mt5.account_info()
            if account_info is not None:
                self.initial_equity = account_info.equity
                self.max_equity = self.initial_equity
                self.equity_curve = [self.initial_equity]
        
        account_info = mt5.account_info()
        if account_info is None:
            return self.lot_size
            
        current_equity = account_info.equity
        self.equity_curve.append(current_equity)
        self.max_equity = max(self.max_equity, current_equity)
        
        # Calculate drawdown
        self.current_drawdown = (self.max_equity - current_equity) / self.max_equity
        
        # Base position size on equity growth
        equity_growth = current_equity / self.initial_equity
        base_size = self.lot_size * equity_growth
        
        # Adjust for drawdown
        if self.current_drawdown > 0.1:  # 10% drawdown
            base_size *= 0.5
        
        return self._round_position_size(base_size, mt5.symbol_info(symbol))
        
    def _round_position_size(self, size: float, symbol_info: Any) -> float:
        """Round position size to valid lot size"""
        size = round(size / symbol_info.volume_step) * symbol_info.volume_step
        return max(symbol_info.volume_min, min(size, symbol_info.volume_max))
        
    def check_trade_conditions(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, str, float, float, float]:
        """Check trading conditions using multiple strategies"""
        try:
            if len(df) < 200:  # Need enough data for indicators
                return False, "", 0, 0, 0
                
            # Get latest data
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_bb_upper = df['bb_upper'].iloc[-1]
            current_bb_lower = df['bb_lower'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            current_adx = df['adx'].iloc[-1]
            
            # Get news sentiment (now simplified)
            news_sentiment = self.sentiment_analyzer.get_news_sentiment(symbol)
            self.market_conditions['news_sentiment'][symbol] = news_sentiment
            
            # Train LSTM model if not trained
            if not self.lstm_predictor.is_trained:
                prices = df['close'].values
                if not self.lstm_predictor.train(df):
                    logging.warning(f"Failed to train LSTM model for {symbol}")
                    return False, "", 0, 0, 0
            
            # Get LSTM prediction
            lstm_prediction = self.lstm_predictor.predict(df)
            self.market_conditions['lstm_prediction'] = lstm_prediction
            
            # Check if market is too volatile
            if current_atr / current_price > 0.02:  # 2% ATR threshold
                logging.info(f"Market too volatile for {symbol}")
                return False, "", 0, 0, 0
                
            # Check if trend is strong enough
            if current_adx < 25:  # Increased threshold for stronger trend
                logging.info(f"Trend too weak for {symbol}")
                return False, "", 0, 0, 0
                
            # Check if we already have too many positions
            positions = mt5.positions_get(symbol=symbol)
            if positions is not None and len(positions) >= self.max_positions:
                logging.info(f"Maximum positions reached for {symbol}")
                return False, "", 0, 0, 0
                
            # Check daily limits
            if self.check_daily_limits():
                logging.info("Daily limits reached")
                return False, "", 0, 0, 0
                
            # Trend following strategy with improved conditions
            trend_signal = 0
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            ema_200 = df['ema_200'].iloc[-1]
            
            # Strong uptrend: All EMAs aligned upward
            if (ema_9 > ema_21 and ema_21 > ema_50 and ema_50 > ema_200 and
                current_price > ema_9 and current_adx > 30):
                trend_signal = 1
            # Strong downtrend: All EMAs aligned downward
            elif (ema_9 < ema_21 and ema_21 < ema_50 and ema_50 < ema_200 and
                  current_price < ema_9 and current_adx > 30):
                trend_signal = -1
                
            # Mean reversion strategy
            mean_reversion_signal = 0
            if current_price < current_bb_lower and current_rsi < 30:
                mean_reversion_signal = 1
            elif current_price > current_bb_upper and current_rsi > 70:
                mean_reversion_signal = -1
                
            # Breakout strategy
            breakout_signal = 0
            if current_price > df['bb_upper'].iloc[-2] and df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]:
                breakout_signal = 1
            elif current_price < df['bb_lower'].iloc[-2] and df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1]:
                breakout_signal = -1
                
            # LSTM prediction signal
            lstm_signal = 0
            if lstm_prediction is not None:
                if lstm_prediction > current_price * 1.001:  # Predicted 0.1% increase
                    lstm_signal = 1
                elif lstm_prediction < current_price * 0.999:  # Predicted 0.1% decrease
                    lstm_signal = -1
                    
            # News sentiment signal (now simplified)
            sentiment_signal = 0  # Neutral sentiment for now
            
            # Combine signals with adjusted weights (removed news sentiment weight)
            combined_signal = (
                trend_signal * 0.3 +  # Increased trend following weight
                mean_reversion_signal * 0.25 +  # Increased mean reversion weight
                breakout_signal * 0.25 +  # Increased breakout weight
                lstm_signal * 0.2  # LSTM prediction weight
            )
            
            # Check for upcoming high-impact economic events
            economic_events = self.economic_calendar.get_economic_events(symbol)
            if economic_events:
                logging.info(f"High-impact economic events found for {symbol}")
                # Adjust risk parameters for high-impact events
                stop_loss_multiplier = 1.5
                take_profit_multiplier = 1.5
            else:
                stop_loss_multiplier = 1.0
                take_profit_multiplier = 1.0
                
            # Update market correlations
            correlations = self.correlation_analyzer.update_correlations(self.symbols, mt5.TIMEFRAME_H1)
            self.market_conditions['correlations'] = correlations
            
            # Adjust position size based on correlation
            correlation_factor = 1.0
            for other_symbol, corr in correlations.items():
                if other_symbol.startswith(symbol) and abs(corr) > 0.7:
                    correlation_factor *= 0.5  # Reduce position size for highly correlated pairs
            
            # Get account info for position sizing
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False, "", 0, 0, 0
                
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                current_price=current_price,
                stop_loss=current_atr * 2 * stop_loss_multiplier,  # Using ATR for stop loss
                account_balance=account_info.balance,
                volatility=current_atr / current_price,
                correlation_factor=correlation_factor
            )
            
            # Determine trade direction and calculate stop loss/take profit
            if combined_signal > 0.5:  # Strong buy signal
                stop_loss = current_price - current_atr * 2 * stop_loss_multiplier
                take_profit = current_price + current_atr * 4 * take_profit_multiplier
                return True, "buy", stop_loss, take_profit, position_size
            elif combined_signal < -0.5:  # Strong sell signal
                stop_loss = current_price + current_atr * 2 * stop_loss_multiplier
                take_profit = current_price - current_atr * 4 * take_profit_multiplier
                return True, "sell", stop_loss, take_profit, position_size
                
            return False, "", 0, 0, 0
            
        except Exception as e:
            logging.error(f"Error checking trade conditions: {e}")
            return False, "", 0, 0, 0
            
    def check_daily_limits(self) -> bool:
        """Check if daily trading limits are reached"""
        try:
            current_date = datetime.now().date()
            
            # Reset daily stats if it's a new day
            if self.performance_stats['last_trade_date'] != current_date:
                self.performance_stats['daily_trades'] = 0
                self.performance_stats['daily_profit'] = 0.0
                self.performance_stats['last_trade_date'] = current_date
                
            # Check daily profit target
            if self.performance_stats['daily_profit'] >= 100:  # $100 daily profit target
                logging.info(f"Daily profit target reached: ${self.performance_stats['daily_profit']:.2f}")
                return True
                
            # Check daily loss limit
            if self.performance_stats['daily_profit'] <= -50:  # $50 daily loss limit
                logging.info(f"Daily loss limit reached: ${self.performance_stats['daily_profit']:.2f}")
                return True
                
            # Check maximum daily trades
            if self.performance_stats['daily_trades'] >= 5:  # Maximum 5 trades per day
                logging.info(f"Maximum daily trades reached: {self.performance_stats['daily_trades']}")
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking daily limits: {e}")
            return True  # Return True to prevent trading on error
            
    def place_order(self, symbol: str, order_type: str, price: float, sl: float, tp: float, position_size: float) -> bool:
        """Place a new order with risk management"""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Failed to get symbol info for {symbol}")
                return False
                
            # Validate parameters
            if position_size <= 0:
                logging.error(f"Invalid position size: {position_size}")
                return False
                
            if sl <= 0 or tp <= 0:
                logging.error(f"Invalid stop loss or take profit: SL={sl}, TP={tp}")
                return False
                
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 123456,
                "comment": "Advanced Premium Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result.comment}")
                return False
                
            # Record trade in history
            trade_record = {
                'time': datetime.now().timestamp(),
                'symbol': symbol,
                'type': order_type,
                'price': price,
                'sl': sl,
                'tp': tp,
                'lot_size': position_size,
                'profit': 0,
                'balance': mt5.account_info().balance,
                'price_data': df.to_dict('records')
            }
            
            # Update risk manager
            self.risk_manager.update_performance_history(trade_record)
            
            # Send Telegram notification
            message = (
                f"🔔 New Trade\n\n"
                f"Symbol: {symbol}\n"
                f"Type: {order_type.upper()}\n"
                f"Price: {price:.5f}\n"
                f"Stop Loss: {sl:.5f}\n"
                f"Take Profit: {tp:.5f}\n"
                f"Position Size: {position_size:.2f} lots"
            )
            asyncio.create_task(self.send_telegram_message(message))
            
            logging.info(f"Order placed successfully: {symbol} {order_type} at {price}")
            return True
            
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return False
        
    def manage_open_positions(self):
        """Manage open positions and update trade history"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                return
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return
                
            # Update trade history for closed positions
            for position in positions:
                # Check if this position is in our trade history
                for trade in self.trade_history:
                    if (trade['symbol'] == position.symbol and 
                        trade['type'] == ('BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL') and
                        trade['profit'] == 0):  # Only update if profit hasn't been recorded yet
                        
                        # Update trade record with final profit
                        trade['profit'] = position.profit
                        trade['close_time'] = datetime.now().timestamp()
                        trade['close_price'] = position.price_current
                        trade['balance'] = account_info.balance
                        
                        # Send Telegram notification
                        message = (
                            f"🔔 Position Closed\n\n"
                            f"Symbol: {position.symbol}\n"
                            f"Type: {trade['type']}\n"
                            f"Profit: {position.profit:.2f}\n"
                            f"Balance: {account_info.balance:.2f}"
                        )
                        asyncio.create_task(self.send_telegram_message(message))
                        
                        break
            
            # Update trailing stops
            for position in positions:
                if position.type == mt5.POSITION_TYPE_BUY:
                    # For long positions
                    if position.price_current - position.price_open > self.trailing_stop * mt5.symbol_info(position.symbol).point:
                        new_sl = position.price_current - self.trailing_stop * mt5.symbol_info(position.symbol).point
                        if new_sl > position.sl:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": position.ticket,
                                "symbol": position.symbol,
                                "sl": new_sl,
                                "tp": position.tp
                            }
                            mt5.order_send(request)
                else:
                    # For short positions
                    if position.price_open - position.price_current > self.trailing_stop * mt5.symbol_info(position.symbol).point:
                        new_sl = position.price_current + self.trailing_stop * mt5.symbol_info(position.symbol).point
                        if new_sl < position.sl or position.sl == 0:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": position.ticket,
                                "symbol": position.symbol,
                                "sl": new_sl,
                                "tp": position.tp
                            }
                            mt5.order_send(request)
            
        except Exception as e:
            logging.error(f"Error managing positions: {e}")
            
    def data_collector(self):
        """Collect market data for all symbols and timeframes"""
        while self.running:
            try:
                for symbol in self.symbols:
                    # Verify symbol exists and is available for trading
                    symbol_info = mt5.symbol_info(symbol)
                    if symbol_info is None:
                        logging.error(f"Symbol {symbol} not found")
                        continue
                        
                    if not symbol_info.visible:
                        if not mt5.symbol_select(symbol, True):
                            logging.error(f"Failed to select symbol {symbol}")
                            continue
                            
                    for timeframe_name, timeframe in self.timeframes.items():
                        try:
                            # Request data with volume
                            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
                            if rates is None:
                                logging.error(f"Failed to get rates for {symbol} on {timeframe_name}")
                                continue
                                
                            if len(rates) < 200:  # Minimum required data points
                                logging.warning(f"Insufficient data for {symbol} on {timeframe_name}")
                                continue
                                
                            df = pd.DataFrame(rates)
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                            df.set_index('time', inplace=True)
                            
                            # Add volume column if it doesn't exist
                            if 'volume' not in df.columns:
                                df['volume'] = 0
                                
                            # Calculate indicators
                            df = self.calculate_advanced_indicators(df)
                            
                            # Validate data
                            if df.isnull().values.any():
                                logging.warning(f"NaN values detected in data for {symbol} on {timeframe_name}")
                                continue
                                
                            # Store in queue
                            self.data_queue.put((symbol, timeframe_name, df))
                            
                        except Exception as e:
                            logging.error(f"Error processing {symbol} on {timeframe_name}: {e}")
                            continue
                            
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logging.error(f"Error in data collector: {e}")
                time.sleep(5)  # Wait longer on error
                
    def update_performance_stats(self):
        """Update performance statistics based on trade history"""
        try:
            if not self.trade_history:
                return
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return
                
            # Update basic stats
            self.performance_stats['balance'] = account_info.balance
            self.performance_stats['equity'] = account_info.equity
            self.performance_stats['margin'] = account_info.margin
            self.performance_stats['free_margin'] = account_info.margin_free
            
            # Calculate trade statistics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
            winning_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) > 0)
            losing_profit = sum(trade.get('profit', 0) for trade in self.trade_history if trade.get('profit', 0) < 0)
            
            # Update performance stats
            self.performance_stats['total_trades'] = total_trades
            self.performance_stats['winning_trades'] = winning_trades
            self.performance_stats['losing_trades'] = losing_trades
            self.performance_stats['total_profit'] = total_profit
            
            if total_trades > 0:
                self.performance_stats['win_rate'] = (winning_trades / total_trades) * 100
                self.performance_stats['average_profit'] = winning_profit / winning_trades if winning_trades > 0 else 0
                self.performance_stats['average_loss'] = losing_profit / losing_trades if losing_trades > 0 else 0
                self.performance_stats['profit_factor'] = abs(winning_profit / losing_profit) if losing_profit != 0 else float('inf')
            
            # Calculate drawdown
            if self.trade_history:
                peak = self.trade_history[0]['balance']
                max_drawdown = 0
                current_drawdown = 0
                
                for trade in self.trade_history:
                    if trade['balance'] > peak:
                        peak = trade['balance']
                    drawdown = (peak - trade['balance']) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    current_drawdown = drawdown
                
                self.performance_stats['max_drawdown'] = max_drawdown
                self.performance_stats['current_drawdown'] = current_drawdown
            
            # Calculate daily statistics
            today = datetime.now().date()
            daily_trades = sum(1 for trade in self.trade_history 
                             if datetime.fromtimestamp(trade['time']).date() == today)
            daily_profit = sum(trade['profit'] for trade in self.trade_history 
                             if datetime.fromtimestamp(trade['time']).date() == today)
            
            self.performance_stats['daily_trades'] = daily_trades
            self.performance_stats['daily_profit'] = daily_profit
            self.performance_stats['last_trade_date'] = datetime.fromtimestamp(
                self.trade_history[-1]['time']).strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate risk-reward ratio
            if self.performance_stats['average_loss'] != 0:
                self.performance_stats['risk_reward_ratio'] = abs(
                    self.performance_stats['average_profit'] / self.performance_stats['average_loss']
                )
            
            # Calculate additional metrics
            if total_trades > 0:
                # Average trade duration
                trade_durations = []
                for i in range(1, len(self.trade_history)):
                    duration = self.trade_history[i]['time'] - self.trade_history[i-1]['time']
                    trade_durations.append(duration)
                self.performance_stats['average_trade_duration'] = sum(trade_durations) / len(trade_durations)
                
                # Profit per trade
                self.performance_stats['profit_per_trade'] = total_profit / total_trades
                
                # Maximum consecutive wins/losses
                consecutive_wins = 0
                consecutive_losses = 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                
                for trade in self.trade_history:
                    if trade.get('profit', 0) > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                self.performance_stats['max_consecutive_wins'] = max_consecutive_wins
                self.performance_stats['max_consecutive_losses'] = max_consecutive_losses
            
            logging.info("Performance stats updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating performance stats: {e}")
            
    def check_health(self) -> bool:
        """Check the health of the bot and its connections"""
        try:
            # Check MT5 connection
            if not mt5.initialize():
                logging.error("MT5 connection lost")
                return False
                
            # Check account status
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False
                
            # Check if trading is allowed
            if not account_info.trade_allowed:
                logging.error("Trading is not allowed")
                return False
                
            # Check if we have sufficient margin
            if account_info.margin_free < 100:
                logging.error("Insufficient free margin")
                return False
                
            # Check if we can get market data
            for symbol in self.symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logging.error(f"Failed to get tick data for {symbol}")
                    return False
                    
            # Check if we can get historical data
            for symbol in self.symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                if rates is None:
                    logging.error(f"Failed to get historical data for {symbol}")
                    return False
                    
            return True
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return False
            
    async def monitor_health(self):
        """Monitor bot health and send alerts"""
        while self.running:
            try:
                if not self.check_health():
                    message = (
                        "⚠️ Bot Health Alert\n\n"
                        "The bot has detected issues with its operation. "
                        "Please check the logs for more details."
                    )
                    await self.send_telegram_message(message)
                    
                # Check for unusual market conditions
                for symbol in self.symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                        
                    # Check for unusually high spread
                    if tick.ask - tick.bid > 0.0005:  # 5 pip spread
                        message = (
                            f"⚠️ High Spread Alert\n\n"
                            f"Symbol: {symbol}\n"
                            f"Spread: {(tick.ask - tick.bid) * 10000:.1f} pips"
                        )
                        await self.send_telegram_message(message)
                        
                    # Check for unusual price movement
                    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
                    if rates is not None and len(rates) >= 2:
                        price_change = abs(rates[1]['close'] - rates[0]['close']) / rates[0]['close'] * 100
                        if price_change > 0.5:  # 0.5% price change
                            message = (
                                f"⚠️ Unusual Price Movement\n\n"
                                f"Symbol: {symbol}\n"
                                f"Change: {price_change:.2f}%"
                            )
                            await self.send_telegram_message(message)
                            
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
                
    async def shutdown(self):
        """Graceful shutdown of the bot"""
        try:
            logging.info("Initiating bot shutdown...")
            self.running = False
            
            # Close all open positions
            positions = mt5.positions_get()
            if positions is not None:
                for position in positions:
                    try:
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": position.symbol,
                            "volume": position.volume,
                            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "position": position.ticket,
                            "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "shutdown",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            logging.error(f"Failed to close position {position.ticket}: {result.comment}")
                    except Exception as e:
                        logging.error(f"Error closing position {position.ticket}: {e}")
            
            # Send final performance report
            final_stats = (
                f"📊 Final Performance Report:\n"
                f"Total Trades: {self.performance_stats['total_trades']}\n"
                f"Win Rate: {self.performance_stats['win_rate']:.2%}\n"
                f"Total Profit: ${self.performance_stats['total_profit']:.2f}\n"
                f"Max Drawdown: {self.performance_stats['max_drawdown']:.2f}%"
            )
            await self.send_telegram_message(final_stats)
            
            # Shutdown MT5
            mt5.shutdown()
            logging.info("Bot shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            
    def save_state(self):
        """Save bot state to file"""
        try:
            state = {
                'performance_stats': self.performance_stats,
                'market_conditions': self.market_conditions,
                'correlation_matrix': self.correlation_matrix,
                'last_update': datetime.now().isoformat()
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=4)
                
            logging.info("Bot state saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving bot state: {e}")
            
    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                    
                self.performance_stats = state.get('performance_stats', self.performance_stats)
                self.market_conditions = state.get('market_conditions', {})
                self.correlation_matrix = state.get('correlation_matrix', {})
                
                logging.info("Bot state loaded successfully")
                
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")
            
    async def run(self):
        """Main bot loop"""
        print("\n" + "="*50)
        print("ADVANCED PREMIUM TRADING BOT STARTING")
        print("="*50 + "\n")
        
        if not self.initialize():
            return
            
        await self.initialize_telegram()
        
        # Load previous state
        self.load_state()
        
        # Start data collector thread
        data_thread = threading.Thread(target=self.data_collector)
        data_thread.daemon = True
        data_thread.start()
        
        # Start health monitoring
        health_monitor = asyncio.create_task(self.monitor_health())
        
        try:
            # Main trading loop
            while self.running:
                try:
                    # Process market data
                    while not self.data_queue.empty():
                        try:
                            symbol, timeframe, df = self.data_queue.get(timeout=1)
                            
                            # Validate data
                            if df is None or len(df) < 200:
                                logging.warning(f"Insufficient data for {symbol} on {timeframe}")
                                continue
                                
                            # Check for NaN values
                            if df.isnull().values.any():
                                logging.warning(f"NaN values detected in data for {symbol} on {timeframe}")
                                continue
                                
                            # Check trading conditions
                            should_trade, order_type, sl, tp, position_size = self.check_trade_conditions(df, symbol)
                            
                            if should_trade:
                                # Place order
                                if self.place_order(symbol, order_type, df['close'].iloc[-1], sl, tp, position_size):
                                    # Update statistics
                                    self.performance_stats['total_trades'] += 1
                                    if order_type == "buy":
                                        self.performance_stats['winning_trades'] += 1
                                    else:
                                        self.performance_stats['losing_trades'] += 1
                                        
                        except queue.Empty:
                            break
                        except Exception as e:
                            logging.error(f"Error processing data for {symbol}: {e}")
                            continue
                            
                    # Manage open positions
                    try:
                        self.manage_open_positions()
                    except Exception as e:
                        logging.error(f"Error managing positions: {e}")
                    
                    # Update performance stats every 5 minutes
                    if datetime.now().minute % 5 == 0:
                        try:
                            self.update_performance_stats()
                            self.save_state()  # Save state periodically
                            
                            # Send performance update via Telegram
                            stats_message = (
                                f"📊 Performance Update:\n"
                                f"Total Trades: {self.performance_stats['total_trades']}\n"
                                f"Win Rate: {self.performance_stats['win_rate']:.2%}\n"
                                f"Total Profit: ${self.performance_stats['total_profit']:.2f}\n"
                                f"Current Drawdown: {self.performance_stats['current_drawdown']:.2f}%"
                            )
                            await self.send_telegram_message(stats_message)
                        except Exception as e:
                            logging.error(f"Error updating performance stats: {e}")
                        
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logging.info("Bot task cancelled")
                    break
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    # Attempt to recover
                    try:
                        # Check MT5 connection
                        if not mt5.initialize():
                            logging.error("MT5 connection lost. Attempting to reconnect...")
                            if not self.initialize():
                                logging.error("Failed to reconnect to MT5")
                                break
                    except Exception as reconnect_error:
                        logging.error(f"Error during recovery attempt: {reconnect_error}")
                        break
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            print("\nShutting down bot...")
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
        finally:
            # Cleanup
            self.running = False
            health_monitor.cancel()
            try:
                await health_monitor
            except asyncio.CancelledError:
                pass
            await self.shutdown()
            
if __name__ == "__main__":
    try:
        bot = AdvancedPremiumBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("\nBot shutdown complete.") 