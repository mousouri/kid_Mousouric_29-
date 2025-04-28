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
import onnxruntime as ort
import io
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
    def __init__(self, window_size=60, account=None, password=None, server=None):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 20
        self.patience = 5  # Early stopping patience
        self.onnx_session = None
        self.onnx_model_path = 'lstm_model.onnx'
        self.account = account
        self.password = password
        self.server = server
        
    def initialize(self) -> bool:
        """Initialize the LSTM predictor"""
        try:
            # Initialize MT5 if credentials are provided
            if all([self.account, self.password, self.server]):
                if not mt5.initialize():
                    logging.error("Failed to initialize MT5")
                    return False
                    
                # Login to MT5 account
                if not mt5.login(self.account, password=self.password, server=self.server):
                    logging.error("Failed to login to MT5 account")
                    return False
                    
                # Get account info
                account_info = mt5.account_info()
                if account_info is None:
                    logging.error("Failed to get account info")
                    return False
                    
                logging.info(f"Connected to MT5 account: {account_info.login}")
            
            # Try to load existing ONNX model
            if os.path.exists(self.onnx_model_path):
                self.onnx_session = ort.InferenceSession(
                    self.onnx_model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.is_trained = True
                logging.info("Loaded existing ONNX model successfully")
                return True
            else:
                # Create realistic dummy data with proper indicators
                dummy_data = pd.DataFrame({
                    'close': np.random.normal(1.0, 0.01, 1000),  # Normal distribution around 1.0
                    'volume': np.random.randint(100, 1000, 1000),  # Random volume between 100-1000
                    'rsi': np.random.uniform(30, 70, 1000),  # RSI between 30-70 (typical range)
                    'macd': np.random.normal(0.0, 0.1, 1000)  # MACD around 0 with small variance
                })
                
                if self.train(dummy_data):
                    logging.info("Created initial LSTM model with valid dummy data")
                    return True
                else:
                    logging.error("Failed to create initial LSTM model")
                    return False
        except Exception as e:
            logging.error(f"Error initializing LSTM predictor: {e}")
            return False
        
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
        
    def export_to_onnx(self, model, input_shape):
        """Export PyTorch model to ONNX format"""
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape, device=self.device)
            
            # Export the model
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Create ONNX Runtime session
            self.onnx_session = ort.InferenceSession(
                self.onnx_model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            logging.info("Model successfully exported to ONNX format")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting model to ONNX: {e}")
            return False
            
    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for LSTM training with automatic indicator generation"""
        try:
            # Generate missing technical indicators if needed
            if 'rsi' not in df.columns:
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                
            if 'macd' not in df.columns:
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                
            # Ensure volume exists (create dummy volume if missing)
            if 'volume' not in df.columns:
                df['volume'] = 0  # Or calculate from tick data
                
            # Verify required columns after generation
            required_columns = ['close', 'volume', 'rsi', 'macd']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logging.error(f"Still missing columns after generation: {missing}")
                return None

            # Select and scale features
            features = df[required_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None
            
    def train(self, df: pd.DataFrame) -> bool:
        """Train the LSTM model with early stopping and export to ONNX"""
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
                        
            # Export model to ONNX
            if not self.export_to_onnx(self.model, (1, self.window_size, X.shape[2])):
                logging.error("Failed to export model to ONNX")
                return False
                
            self.is_trained = True
            logging.info("LSTM model trained and exported successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training LSTM model: {e}")
            return False
            
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """Make prediction using the ONNX model"""
        if not self.is_trained or self.onnx_session is None:
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
                
            # Convert to numpy array
            X = last_sequence.reshape(1, self.window_size, -1).astype(np.float32)
            
            # Make prediction using ONNX Runtime
            ort_inputs = {self.onnx_session.get_inputs()[0].name: X}
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            prediction = ort_outputs[0][0, 0]
            
            # Inverse transform the prediction
            dummy_array = np.zeros((1, 4))  # 4 features
            dummy_array[0, 0] = prediction
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            return predicted_price
            
        except Exception as e:
            logging.error(f"Error making LSTM prediction: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the bot and its components"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logging.error("Failed to initialize MT5")
                return False
                
            # Login to MT5 account
            if not mt5.login(self.account, password=self.password, server=self.server):
                logging.error("Failed to login to MT5 account")
                return False
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False
                
            logging.info(f"Connected to MT5 account: {account_info.login}")
            
            # Initialize LSTM predictor
            if not self.train(pd.DataFrame(np.zeros((1000, 4)))):
                logging.error("Failed to train LSTM predictor")
                return False
                
            # Initialize sentiment analyzer
            if not self.sentiment_analyzer.initialize():
                logging.error("Failed to initialize sentiment analyzer")
                return False
                
            # Initialize market correlation analyzer
            self.correlation_analyzer.update_correlations(self.symbols, mt5.TIMEFRAME_H1)
            
            # Set initial equity
            self.risk_manager.initial_equity = account_info.balance
            self.risk_manager.max_equity = account_info.balance
            
            # Mark as initialized
            self.initialized = True
            
            logging.info("Bot initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing bot: {e}")
            return False

class NewsSentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = None
        self.last_update = None
        self.sentiment_cache = {}
        self.cache_duration = 3600  # Cache for 1 hour
        
    def initialize(self) -> bool:
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
        self.max_daily_drawdown = 0.02  # Initial 2% max daily drawdown
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
        self.max_open_risk = 0.05  # 5% of account balance
        self.performance_window = 20  # Number of trades to consider for dynamic limits
        self.min_daily_drawdown = 0.01  # Minimum 1% daily drawdown limit
        self.max_daily_drawdown = 0.03  # Maximum 3% daily drawdown limit
        self.hedging_threshold = 0.8  # Correlation threshold for hedging detection
        self.atr_multiplier = 2.0  # Multiplier for ATR-based position sizing
        self.trailing_stop_atr_multiplier = 1.5  # Multiplier for ATR-based trailing stop
        self.min_trailing_stop = 10  # Minimum trailing stop in pips
        self.max_trailing_stop = 50  # Maximum trailing stop in pips
        
    def calculate_adaptive_position_size(self, 
                                       symbol: str, 
                                       current_price: float, 
                                       stop_loss: float, 
                                       account_balance: float,
                                       volatility: float,
                                       correlation_factor: float,
                                       atr: float) -> float:
        """Calculate position size based on ATR and market conditions"""
        try:
            if account_balance <= 0:
                logging.error("Invalid account balance")
                return 0.0
                
            # Base position size using ATR
            atr_based_size = (account_balance * 0.01) / (atr * self.atr_multiplier)
            
            # Volatility adjustment
            if volatility > self.volatility_threshold:
                atr_based_size *= self.risk_multipliers['high_volatility']
                logging.info(f"Volatility adjustment applied: {volatility:.2%}")
                
            # Correlation adjustment
            if correlation_factor < 1.0:
                atr_based_size *= correlation_factor
                logging.info(f"Correlation adjustment applied: {correlation_factor:.2%}")
                
            # Time-based risk adjustment
            time_risk = self.check_time_based_risk()
            if time_risk < 1.0:
                atr_based_size *= time_risk
                logging.info(f"Time-based risk adjustment applied: {time_risk:.2%}")
                
            # Check max open risk
            if not self.check_max_open_risk(atr_based_size, account_balance):
                logging.warning("Position size reduced due to max open risk limit")
                atr_based_size *= 0.5
                
            # Convert to lots
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            # Calculate pip value
            pip_value = symbol_info.trade_tick_value * (abs(current_price - stop_loss) / symbol_info.trade_tick_size)
            if pip_value <= 0:
                logging.error("Invalid pip value calculation")
                return 0.0
                
            position_size = atr_based_size / pip_value
            
            # Ensure position size is within limits
            position_size = self._round_position_size(position_size, symbol_info)
            
            logging.info(f"Calculated adaptive position size: {position_size:.2f} lots")
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating adaptive position size: {e}")
            return 0.0
            
    def check_max_open_risk(self, new_position_size: float, account_balance: float) -> bool:
        """Check if adding new position would exceed max open risk"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                return True
                
            # Calculate total current risk
            total_risk = 0
            for position in positions:
                # Calculate risk for each position (position size * stop loss distance)
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info is None:
                    continue
                    
                risk_amount = position.volume * abs(position.price_open - position.sl) * symbol_info.trade_tick_value
                total_risk += risk_amount
                
            # Add new position risk
            new_risk = new_position_size * account_balance * 0.01  # Assuming 1% risk per trade
            total_risk += new_risk
            
            # Check if total risk exceeds limit
            max_allowed_risk = account_balance * self.max_open_risk
            return total_risk <= max_allowed_risk
            
        except Exception as e:
            logging.error(f"Error checking max open risk: {e}")
            return False
            
    def detect_hedging(self, symbol: str, correlation_matrix: Dict[str, float]) -> bool:
        """Detect if new position would create hedging with existing positions"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None:
                return False
                
            # Check correlation with each open position
            for position in positions:
                if position.symbol == symbol:
                    continue
                    
                # Get correlation between symbols
                corr_key = f"{symbol}_{position.symbol}"
                if corr_key in correlation_matrix:
                    correlation = correlation_matrix[corr_key]
                    
                    # If high negative correlation and opposite direction, it's hedging
                    if abs(correlation) > self.hedging_threshold:
                        # Check if positions are in opposite directions
                        if (position.type == mt5.POSITION_TYPE_BUY and symbol.endswith('USD')) or \
                           (position.type == mt5.POSITION_TYPE_SELL and not symbol.endswith('USD')):
                            logging.warning(f"Hedging detected between {symbol} and {position.symbol}")
                            return True
                            
            return False
            
        except Exception as e:
            logging.error(f"Error detecting hedging: {e}")
            return False
            
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
                
            # Check max open risk
            if not self.check_max_open_risk(base_size, account_balance):
                logging.warning("Position size reduced due to max open risk limit")
                base_size *= 0.5
                
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
            
            # Get dynamic daily drawdown limit
            daily_drawdown_limit = self.calculate_dynamic_drawdown_limit()
            
            # Check daily drawdown
            today = datetime.now().date()
            daily_trades = [t for t in self.position_history 
                          if datetime.fromtimestamp(t['time']).date() == today]
            
            if daily_trades:
                daily_start_equity = daily_trades[0]['balance']
                daily_drawdown = (daily_start_equity - current_equity) / daily_start_equity
                if daily_drawdown > daily_drawdown_limit:
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

class StrategyManager:
    def __init__(self):
        self.strategies = {
            'trend_following': self._trend_following_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'breakout': self._breakout_strategy,
            'scalping': self._scalping_strategy
        }
        self.current_strategy = 'trend_following'
        self.strategy_weights = {
            'trend_following': 0.4,
            'mean_reversion': 0.3,
            'breakout': 0.2,
            'scalping': 0.1
        }
        self.market_conditions = {}
        self.strategy_performance = {}
        
    def determine_strategy(self, df: pd.DataFrame) -> str:
        """Determine the best strategy based on market conditions"""
        try:
            # Calculate market conditions
            volatility = df['atr'].iloc[-1] / df['close'].iloc[-1]
            trend_strength = df['adx'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Update market conditions
            self.market_conditions = {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'rsi': rsi
            }
            
            # Adjust strategy weights based on market conditions
            if trend_strength > 25:  # Strong trend
                self.strategy_weights['trend_following'] = 0.5
                self.strategy_weights['mean_reversion'] = 0.2
                self.strategy_weights['breakout'] = 0.2
                self.strategy_weights['scalping'] = 0.1
            elif volatility > 0.015:  # High volatility
                self.strategy_weights['breakout'] = 0.4
                self.strategy_weights['trend_following'] = 0.3
                self.strategy_weights['mean_reversion'] = 0.2
                self.strategy_weights['scalping'] = 0.1
            elif abs(rsi - 50) < 10:  # Range-bound market
                self.strategy_weights['mean_reversion'] = 0.4
                self.strategy_weights['scalping'] = 0.3
                self.strategy_weights['trend_following'] = 0.2
                self.strategy_weights['breakout'] = 0.1
            else:  # Normal conditions
                self.strategy_weights = {
                    'trend_following': 0.4,
                    'mean_reversion': 0.3,
                    'breakout': 0.2,
                    'scalping': 0.1
                }
            
            # Select strategy based on weights
            strategies = list(self.strategy_weights.keys())
            weights = list(self.strategy_weights.values())
            self.current_strategy = np.random.choice(strategies, p=weights)
            
            return self.current_strategy
            
        except Exception as e:
            logging.error(f"Error determining strategy: {e}")
            return 'trend_following'
            
    def check_multi_timeframe_confirmation(self, symbol: str, timeframe: str) -> bool:
        """Check if signals are confirmed across multiple timeframes"""
        try:
            # Get data for different timeframes
            timeframes = {
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4
            }
            
            signals = []
            for tf, mt5_tf in timeframes.items():
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 100)
                if rates is None:
                    continue
                    
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Get signal from current strategy
                signal = self.strategies[self.current_strategy](df)
                signals.append(signal)
            
            # Check if signals are aligned
            if len(signals) < 2:
                return False
                
            # Count positive and negative signals
            positive_signals = sum(1 for s in signals if s > 0)
            negative_signals = sum(1 for s in signals if s < 0)
            
            # Require majority agreement
            return positive_signals > len(signals)/2 or negative_signals > len(signals)/2
            
        except Exception as e:
            logging.error(f"Error checking multi-timeframe confirmation: {e}")
            return False
            
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic indicators for strategy analysis"""
        try:
            # Basic price indicators
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # Volatility indicators
            df['atr'] = ta.volatility.average_true_range(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # ADX
            df['adx'] = ta.trend.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return df
            
    def _trend_following_strategy(self, df: pd.DataFrame) -> float:
        """Trend following strategy using EMAs and ADX"""
        try:
            # Get latest values
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            adx = df['adx'].iloc[-1]
            
            # Check trend strength
            if adx < 25:
                return 0
                
            # Check trend direction
            if ema_9 > ema_21 and ema_21 > ema_50:
                return 1
            elif ema_9 < ema_21 and ema_21 < ema_50:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error in trend following strategy: {e}")
            return 0
            
    def _mean_reversion_strategy(self, df: pd.DataFrame) -> float:
        """Mean reversion strategy using RSI and Bollinger Bands"""
        try:
            # Get latest values
            rsi = df['rsi'].iloc[-1]
            
            # Check for required Bollinger Band columns
            if 'bb_middle' not in df.columns or 'bb_upper' not in df.columns:
                return 0
                
            bb_position = (df['close'].iloc[-1] - df['bb_middle'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_middle'].iloc[-1])
            
            # Check for oversold conditions
            if rsi < 30 and bb_position < -0.5:
                return 1
            # Check for overbought conditions
            elif rsi > 70 and bb_position > 0.5:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error in mean reversion strategy: {e}")
            return 0
            
    def _breakout_strategy(self, df: pd.DataFrame) -> float:
        """Breakout strategy using ATR and price action"""
        try:
            # Get latest values
            atr = df['atr'].iloc[-1]
            current_price = df['close'].iloc[-1]
            prev_high = df['high'].iloc[-2]
            prev_low = df['low'].iloc[-2]
            
            # Check for breakout
            if current_price > prev_high + atr:
                return 1
            elif current_price < prev_low - atr:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error in breakout strategy: {e}")
            return 0
            
    def _scalping_strategy(self, df: pd.DataFrame) -> float:
        """Scalping strategy using short-term indicators"""
        try:
            # Get latest values
            rsi = df['rsi'].iloc[-1]
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            
            # Check for oversold conditions
            if rsi < 30 and stoch_k < 20 and stoch_d < 20:
                return 1
            # Check for overbought conditions
            elif rsi > 70 and stoch_k > 80 and stoch_d > 80:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error in scalping strategy: {e}")
            return 0

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
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.lstm_predictor = LSTMPredictor(account=self.account, password=self.password, server=self.server)
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.economic_calendar = EconomicCalendar()
        self.correlation_analyzer = MarketCorrelationAnalyzer()
        
        # Initialize state variables
        self.initialized = False
        self.running = True
        self.data_queue = queue.Queue()
        self.trade_history = []
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
        
        # Initialize market conditions
        self.market_conditions = {}
        self.correlation_matrix = {}
        
    def check_trade_conditions(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, str, float, float, float]:
        """Check trading conditions using multiple strategies"""
        try:
            if len(df) < 200:  # Need enough data for indicators
                return False, "", 0, 0, 0
                
            # Get latest data
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            # Determine strategy based on market conditions
            strategy = self.strategy_manager.determine_strategy(df)
            
            # Check multi-timeframe confirmation
            if not self.strategy_manager.check_multi_timeframe_confirmation(symbol, 'M15'):
                logging.info(f"Multi-timeframe confirmation failed for {symbol}")
                return False, "", 0, 0, 0
                
            # Get account info for position sizing
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False, "", 0, 0, 0
                
            # Update market correlations
            correlations = self.correlation_analyzer.update_correlations(self.symbols, mt5.TIMEFRAME_H1)
            self.market_conditions['correlations'] = correlations
            
            # Check for hedging
            if self.risk_manager.detect_hedging(symbol, correlations):
                logging.info(f"Hedging detected for {symbol}, reducing position size")
                return False, "", 0, 0, 0
                
            # Calculate position size using adaptive method
            position_size = self.risk_manager.calculate_adaptive_position_size(
                symbol=symbol,
                current_price=current_price,
                stop_loss=current_atr * 2,  # Using ATR for stop loss
                account_balance=account_info.balance,
                volatility=current_atr / current_price,
                correlation_factor=1.0,  # Will be adjusted by risk manager
                atr=current_atr
            )
            
            # Check if position size is too small
            if position_size < 0.01:  # Minimum 0.01 lots
                logging.info(f"Position size too small for {symbol}")
                return False, "", 0, 0, 0
                
            # Get signals based on strategy
            if strategy == 'trend_following':
                signal = self._get_trend_following_signal(df)
            elif strategy == 'mean_reversion':
                signal = self._get_mean_reversion_signal(df)
            elif strategy == 'breakout':
                signal = self._get_breakout_signal(df)
            elif strategy == 'scalping':
                signal = self._get_scalping_signal(df)
            else:
                signal = 0
                
            # Determine trade direction and calculate stop loss/take profit
            if signal > 0.5:  # Strong buy signal
                stop_loss = current_price - current_atr * 2
                take_profit = current_price + current_atr * 4
                return True, "buy", stop_loss, take_profit, position_size
            elif signal < -0.5:  # Strong sell signal
                stop_loss = current_price + current_atr * 2
                take_profit = current_price - current_atr * 4
                return True, "sell", stop_loss, take_profit, position_size
                
            return False, "", 0, 0, 0
            
        except Exception as e:
            logging.error(f"Error checking trade conditions: {e}")
            return False, "", 0, 0, 0
            
    def _get_trend_following_signal(self, df: pd.DataFrame) -> float:
        """Get trend following signal"""
        try:
            # Calculate trend indicators
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            adx = df['adx'].iloc[-1]
            
            # Check trend strength
            if adx < 25:
                return 0
                
            # Check trend direction
            if ema_9 > ema_21 and ema_21 > ema_50:
                return 1
            elif ema_9 < ema_21 and ema_21 < ema_50:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error getting trend following signal: {e}")
            return 0
            
    def _get_mean_reversion_signal(self, df: pd.DataFrame) -> float:
        """Get mean reversion signal"""
        try:
            # Calculate mean reversion indicators
            rsi = df['rsi'].iloc[-1]
            bb_position = (df['close'].iloc[-1] - df['bb_middle'].iloc[-1]) / (df['bb_upper'].iloc[-1] - df['bb_middle'].iloc[-1])
            
            # Check for oversold conditions
            if rsi < 30 and bb_position < -0.5:
                return 1
            # Check for overbought conditions
            elif rsi > 70 and bb_position > 0.5:
                return -1
                
            return 0
            
        except Exception as e:
            logging.error(f"Error getting mean reversion signal: {e}")
            return 0
            
    def manage_open_positions(self):
        """Manage open positions and update trade history"""
        try:
            # Update trailing stops
            self.risk_manager.update_trailing_stops()
            
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
                        
        except Exception as e:
            logging.error(f"Error managing positions: {e}")
            
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
                'balance': mt5.account_info().balance
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
                            
                            # Ensure all required columns exist
                            required_columns = ['close', 'volume', 'rsi', 'macd']
                            if not all(col in df.columns for col in required_columns):
                                missing_columns = [col for col in required_columns if col not in df.columns]
                                logging.error(f"Missing required columns for {symbol} on {timeframe_name}: {missing_columns}")
                                continue
                                
                            # Validate data
                            if df.isnull().values.any():
                                logging.warning(f"NaN values detected in data for {symbol} on {timeframe_name}")
                                continue
                                
                            # Store in queue
                            self.data_queue.put((symbol, timeframe_name, df))
                            
                        except Exception as e:
                            logging.error(f"Error processing {symbol} on {timeframe_name}: {e}")
                            continue
                            
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in data collector: {e}")
                self.running = False
        
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
            # Convert datetime objects to strings
            state = {
                'performance_stats': self.performance_stats.copy(),
                'market_conditions': self.market_conditions.copy(),
                'correlation_matrix': self.correlation_matrix.copy(),
                'last_update': datetime.now().isoformat()
            }
            
            # Convert any datetime objects in performance_stats to strings
            if 'last_trade_date' in state['performance_stats']:
                if isinstance(state['performance_stats']['last_trade_date'], datetime):
                    state['performance_stats']['last_trade_date'] = state['performance_stats']['last_trade_date'].isoformat()
            
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
                
                # Convert string dates back to datetime objects
                if 'last_trade_date' in self.performance_stats:
                    try:
                        self.performance_stats['last_trade_date'] = datetime.fromisoformat(self.performance_stats['last_trade_date'])
                    except (ValueError, TypeError):
                        self.performance_stats['last_trade_date'] = None
                
                logging.info("Bot state loaded successfully")
                
        except Exception as e:
            logging.error(f"Error loading bot state: {e}")
            
    async def run(self):
        """Main bot loop"""
        print("\n" + "="*50)
        print("ADVANCED PREMIUM TRADING BOT STARTING")
        print("="*50 + "\n")
        
        if not self.initialize():  # Add parentheses to call the method
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
            
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators for trading strategies"""
        try:
            # Basic price indicators
            df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # Volatility indicators
            df['atr'] = ta.volatility.average_true_range(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # ADX for trend strength
            df['adx'] = ta.trend.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                smooth_window=3
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            df['mfi'] = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=14
            )
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(
                high=df['high'],
                low=df['low'],
                window1=9,
                window2=26,
                window3=52
            )
            df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
            df['kijun_sen'] = ichimoku.ichimoku_base_line()
            df['senkou_span_a'] = ichimoku.ichimoku_a()
            df['senkou_span_b'] = ichimoku.ichimoku_b()
            
            # Clean up NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            # Verify required columns exist
            required_columns = ['close', 'volume', 'rsi', 'macd']
            if not all(col in df.columns for col in required_columns):
                missing_columns = [col for col in required_columns if col not in df.columns]
                logging.error(f"Missing required columns after calculating indicators: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    async def initialize_telegram(self):
        """Initialize Telegram bot and set up command handlers"""
        try:
            if not self.telegram_token or not self.telegram_chat_id:
                logging.warning("Telegram credentials not found. Telegram notifications disabled.")
                return
                
            # Create application
            self.telegram_app = Application.builder().token(self.telegram_token).build()
            
            # Add command handlers
            self.telegram_app.add_handler(CommandHandler("start", self._telegram_start))
            self.telegram_app.add_handler(CommandHandler("status", self._telegram_status))
            self.telegram_app.add_handler(CommandHandler("stop", self._telegram_stop))
            
            # Start the bot
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            await self.telegram_app.updater.start_polling()
            
            # Send startup message
            await self.send_telegram_message(
                "🤖 Advanced Premium Trading Bot Started\n\n"
                "Available commands:\n"
                "/start - Start the bot\n"
                "/status - Get current status\n"
                "/stop - Stop the bot"
            )
            
            logging.info("Telegram bot initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Telegram bot: {e}")
            
    async def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        try:
            if not hasattr(self, 'telegram_app') or not self.telegram_chat_id:
                return
                
            await self.telegram_app.bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='HTML'
            )
            
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            
    async def _telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            await update.message.reply_text(
                "🤖 Advanced Premium Trading Bot\n\n"
                "Bot is running and monitoring the market.\n"
                "Use /status to check current status."
            )
        except Exception as e:
            logging.error(f"Error handling /start command: {e}")
            
    async def _telegram_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                await update.message.reply_text("❌ Failed to get account information")
                return
                
            # Format status message
            status_message = (
                "📊 Bot Status\n\n"
                f"Account Balance: ${account_info.balance:.2f}\n"
                f"Equity: ${account_info.equity:.2f}\n"
                f"Free Margin: ${account_info.margin_free:.2f}\n\n"
                f"Total Trades: {self.performance_stats['total_trades']}\n"
                f"Win Rate: {self.performance_stats['win_rate']:.2%}\n"
                f"Total Profit: ${self.performance_stats['total_profit']:.2f}\n"
                f"Current Drawdown: {self.performance_stats['current_drawdown']:.2f}%"
            )
            
            await update.message.reply_text(status_message)
            
        except Exception as e:
            logging.error(f"Error handling /status command: {e}")
            
    async def _telegram_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        try:
            await update.message.reply_text("🛑 Stopping bot...")
            self.running = False
        except Exception as e:
            logging.error(f"Error handling /stop command: {e}")

    def initialize(self) -> bool:
        """Initialize the bot and its components"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logging.error("Failed to initialize MT5")
                return False
                
            # Login to MT5 account
            if not mt5.login(self.account, password=self.password, server=self.server):
                logging.error("Failed to login to MT5 account")
                return False
                
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Failed to get account info")
                return False
                
            logging.info(f"Connected to MT5 account: {account_info.login}")
            
            # Initialize LSTM predictor
            if not self.lstm_predictor.initialize():
                logging.error("Failed to initialize LSTM predictor")
                return False
                
            # Initialize sentiment analyzer
            if not self.sentiment_analyzer.initialize():
                logging.error("Failed to initialize sentiment analyzer")
                return False
                
            # Initialize market correlation analyzer
            self.correlation_analyzer.update_correlations(self.symbols, mt5.TIMEFRAME_H1)
            
            # Set initial equity
            self.risk_manager.initial_equity = account_info.balance
            self.risk_manager.max_equity = account_info.balance
            
            # Mark as initialized
            self.initialized = True
            
            logging.info("Bot initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing bot: {e}")
            return False

    def get_open_trades(self) -> List[Dict]:
        """Get list of open trades"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
                
            open_trades = []
            for position in positions:
                trade = {
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'swap': position.swap,
                    'time': datetime.fromtimestamp(position.time).isoformat()
                }
                open_trades.append(trade)
                
            return open_trades
            
        except Exception as e:
            logging.error(f"Error getting open trades: {e}")
            return []
            
    def get_latest_prices(self) -> Dict[str, Dict]:
        """Get latest prices for all symbols"""
        try:
            prices = {}
            for symbol in self.symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                    
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
            
    def get_trade_updates(self) -> List[Dict]:
        """Get updates for closed trades"""
        try:
            updates = []
            for trade in self.trade_history:
                if trade.get('profit', 0) != 0 and not trade.get('notified', False):
                    updates.append({
                        'symbol': trade['symbol'],
                        'type': trade['type'],
                        'profit': trade['profit'],
                        'time': datetime.fromtimestamp(trade['time']).isoformat()
                    })
                    trade['notified'] = True
                    
            return updates
            
        except Exception as e:
            logging.error(f"Error getting trade updates: {e}")
            return []

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