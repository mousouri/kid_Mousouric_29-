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
import random
import ctypes
from ctypes import cdll, c_double, c_int, c_char_p, POINTER
import statistics
from collections import deque
from trading_optimizer import calculate_twap as py_calculate_twap, optimize_order_split as py_optimize_order_split

# Configure logging to suppress the C++ library warning
logging.getLogger().setLevel(logging.ERROR)

# Load C++ library for performance-critical operations
USE_CPP_LIB = False
trading_lib = None

# Only try to load the C++ library if it exists
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_optimizer.dll')
if os.path.exists(dll_path):
    try:
        trading_lib = cdll.LoadLibrary(dll_path)
        trading_lib.calculate_twap.argtypes = [POINTER(c_double), c_int, c_int]
        trading_lib.calculate_twap.restype = c_double
        trading_lib.optimize_order_split.argtypes = [c_double, POINTER(c_double), c_int]
        trading_lib.optimize_order_split.restype = POINTER(c_double)
        USE_CPP_LIB = True
    except Exception:
        pass

# Restore logging level
logging.getLogger().setLevel(logging.INFO)

class OrderRouter:
    def __init__(self):
        self.brokers = {
            'mt5_primary': {
                'name': 'MT5 Primary',
                'min_lot': 0.01,
                'max_lot': 100.0,
                'slippage': 0.0001,
                'latency': 0.1
            },
            'mt5_secondary': {
                'name': 'MT5 Secondary',
                'min_lot': 0.01,
                'max_lot': 50.0,
                'slippage': 0.0002,
                'latency': 0.15
            }
        }
        self.order_history = []
        self.slippage_history = {}
        self.latency_history = {}
        self.tick_data = {}
        self.websocket_connections = {}
        self.tick_data_queues = {}
        self.last_twap_calculation = {}
        self.twap_windows = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'H1': 3600
        }
        
    def initialize_websocket(self, symbol: str) -> bool:
        """Initialize WebSocket connection for real-time tick data"""
        try:
            if symbol in self.websocket_connections:
                return True
                
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker",
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            self.websocket_connections[symbol] = ws
            self.tick_data_queues[symbol] = queue.Queue()
            
            return True
            
        except Exception as e:
            logging.error(f"Error initializing WebSocket for {symbol}: {e}")
            return False
            
    def _on_websocket_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            symbol = data['s']
            if symbol in self.tick_data_queues:
                self.tick_data_queues[symbol].put({
                    'price': float(data['c']),
                    'volume': float(data['v']),
                    'timestamp': int(data['E'])
                })
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")
            
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket errors"""
        logging.error(f"WebSocket error: {error}")
        
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logging.info("WebSocket connection closed")
        
    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """Get latest tick data from WebSocket"""
        try:
            if symbol not in self.tick_data_queues:
                if not self.initialize_websocket(symbol):
                    return None
                    
            try:
                return self.tick_data_queues[symbol].get_nowait()
            except queue.Empty:
                return None
                
        except Exception as e:
            logging.error(f"Error getting tick data for {symbol}: {e}")
            return None
            
    def calculate_twap(self, symbol: str, timeframe: str, size: float) -> float:
        """Calculate Time-Weighted Average Price"""
        try:
            current_time = time.time()
            
            # Check if we need to recalculate
            if (symbol in self.last_twap_calculation and 
                current_time - self.last_twap_calculation[symbol] < 60):
                return self.tick_data.get(symbol, {}).get('twap', 0)
                
            # Get tick data
            ticks = []
            window = self.twap_windows[timeframe]
            start_time = current_time - window
            
            while not self.tick_data_queues[symbol].empty():
                tick = self.tick_data_queues[symbol].get_nowait()
                if tick['timestamp'] >= start_time:
                    ticks.append(tick)
                    
            if not ticks:
                return 0
                
            # Use C++ or Python implementation
            if USE_CPP_LIB:
                # Convert to C++ compatible format
                prices = (c_double * len(ticks))()
                for i, tick in enumerate(ticks):
                    prices[i] = tick['price']
                    
                twap = trading_lib.calculate_twap(prices, len(ticks), window)
            else:
                # Python implementation
                prices = [tick['price'] for tick in ticks]
                twap = py_calculate_twap(prices, window)
                
            # Cache result
            if symbol not in self.tick_data:
                self.tick_data[symbol] = {}
            self.tick_data[symbol]['twap'] = twap
            self.last_twap_calculation[symbol] = current_time
            
            return twap
            
        except Exception as e:
            logging.error(f"Error calculating TWAP for {symbol}: {e}")
            return 0
            
    def optimize_order_split(self, symbol: str, total_size: float) -> Dict[str, float]:
        """Optimize order split across brokers"""
        try:
            # Get broker parameters
            broker_params = []
            for broker_id, broker in self.brokers.items():
                broker_params.append({
                    'id': broker_id,
                    'min_lot': broker['min_lot'],
                    'max_lot': broker['max_lot'],
                    'slippage': broker['slippage'],
                    'latency': broker['latency']
                })
                
            # Use C++ or Python implementation
            if USE_CPP_LIB:
                # Convert to C++ compatible format
                params = (c_double * len(broker_params))()
                for i, param in enumerate(broker_params):
                    params[i] = param['slippage'] * param['latency']
                    
                splits = trading_lib.optimize_order_split(total_size, params, len(broker_params))
                
                # Convert result to dictionary
                result = {}
                for i, broker in enumerate(broker_params):
                    result[broker['id']] = splits[i]
                return result
            else:
                # Python implementation
                return py_optimize_order_split(total_size, broker_params)
                
        except Exception as e:
            logging.error(f"Error optimizing order split: {e}")
            return {}
            
    def track_slippage(self, symbol: str, intended_price: float, actual_price: float) -> None:
        """Track slippage for post-trade analysis"""
        try:
            if symbol not in self.slippage_history:
                self.slippage_history[symbol] = []
                
            slippage = (actual_price - intended_price) / intended_price
            self.slippage_history[symbol].append(slippage)
            
            # Keep only last 1000 values
            if len(self.slippage_history[symbol]) > 1000:
                self.slippage_history[symbol] = self.slippage_history[symbol][-1000:]
                
        except Exception as e:
            logging.error(f"Error tracking slippage: {e}")
            
    def track_latency(self, symbol: str, order_time: float, fill_time: float) -> None:
        """Track execution latency for post-trade analysis"""
        try:
            if symbol not in self.latency_history:
                self.latency_history[symbol] = []
                
            latency = fill_time - order_time
            self.latency_history[symbol].append(latency)
            
            # Keep only last 1000 values
            if len(self.latency_history[symbol]) > 1000:
                self.latency_history[symbol] = self.latency_history[symbol][-1000:]
                
        except Exception as e:
            logging.error(f"Error tracking latency: {e}")
            
    def get_slippage_stats(self, symbol: str) -> Dict[str, float]:
        """Get slippage statistics for a symbol"""
        try:
            if symbol not in self.slippage_history or not self.slippage_history[symbol]:
                return {
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'max': 0,
                    'min': 0
                }
                
            slippages = self.slippage_history[symbol]
            return {
                'mean': statistics.mean(slippages),
                'median': statistics.median(slippages),
                'std': statistics.stdev(slippages) if len(slippages) > 1 else 0,
                'max': max(slippages),
                'min': min(slippages)
            }
            
        except Exception as e:
            logging.error(f"Error getting slippage stats: {e}")
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'max': 0,
                'min': 0
            }
            
    def get_latency_stats(self, symbol: str) -> Dict[str, float]:
        """Get latency statistics for a symbol"""
        try:
            if symbol not in self.latency_history or not self.latency_history[symbol]:
                return {
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'max': 0,
                    'min': 0
                }
                
            latencies = self.latency_history[symbol]
            return {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'max': max(latencies),
                'min': min(latencies)
            }
            
        except Exception as e:
            logging.error(f"Error getting latency stats: {e}")
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'max': 0,
                'min': 0
            }

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
        
        # Data-related attributes
        self.required_columns = ['close', 'volume', 'rsi', 'macd']
        self.data_version = "1.2"  # Update when data format changes
        
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
                dummy_data = pd.DataFrame(np.zeros((1000, 4)), columns=['close', 'volume', 'rsi', 'macd'])
                
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
            # Check for essential 'close' column first
            if 'close' not in df.columns:
                logging.error("Missing 'close' column in the DataFrame")
                return None

            # Generate missing technical indicators if needed
            if 'rsi' not in df.columns:
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            if 'macd' not in df.columns:
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
            if 'volume' not in df.columns:
                df['volume'] = 0  # Or calculate from tick data if available

            # Verify all required columns are present
            if not all(col in df.columns for col in self.required_columns):
                missing = [col for col in self.required_columns if col not in df.columns]
                logging.error(f"Missing columns after generation: {missing}")
                return None

            # Select and scale features
            features = df[self.required_columns].values
            scaled_features = self.scaler.fit_transform(features)
            return scaled_features
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None
            
    def train(self, df: pd.DataFrame) -> bool:
        """Train the LSTM model with early stopping and export to ONNX"""
        try:
            # Check data quality before training
            if df.isnull().values.any():
                logging.error("Training data contains NaN values")
                return False
            if len(df) < 1000:
                logging.warning(f"Training on small dataset (n={len(df)})")
                
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

class AdvancedRiskManager:
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.5,    # 50th percentile
            'medium': 0.7,  # 70th percentile
            'high': 0.9    # 90th percentile
        }
        self.circuit_breaker_threshold = 0.05  # 5% move in 1 minute
        self.monte_carlo_simulations = 10000
        self.volatility_history = {}
        self.price_history = {}
        self.simulation_results = {}
        self.last_simulation_time = None
        self.simulation_interval = 3600  # Run simulation every hour
        
    def calculate_dynamic_position_size(self, symbol: str, base_size: float, atr: float) -> float:
        """Calculate position size based on ATR percentiles"""
        try:
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
                
            # Add current ATR to history
            self.volatility_history[symbol].append(atr)
            
            # Keep only last 1000 values
            if len(self.volatility_history[symbol]) > 1000:
                self.volatility_history[symbol] = self.volatility_history[symbol][-1000:]
                
            # Calculate current ATR percentile
            current_percentile = sum(1 for x in self.volatility_history[symbol] if x <= atr) / len(self.volatility_history[symbol])
            
            # Adjust position size based on volatility
            if current_percentile >= self.volatility_thresholds['high']:
                return base_size * 0.5  # Reduce by 50% in high volatility
            elif current_percentile >= self.volatility_thresholds['medium']:
                return base_size * 0.75  # Reduce by 25% in medium volatility
            else:
                return base_size
                
        except Exception as e:
            logging.error(f"Error calculating dynamic position size: {e}")
            return base_size
            
    def check_circuit_breaker(self, symbol: str, current_price: float) -> bool:
        """Check if circuit breaker should be triggered"""
        try:
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            # Add current price to history
            self.price_history[symbol].append((time.time(), current_price))
            
            # Keep only last 60 seconds of data
            current_time = time.time()
            self.price_history[symbol] = [(t, p) for t, p in self.price_history[symbol] if current_time - t <= 60]
            
            if len(self.price_history[symbol]) < 2:
                return False
                
            # Calculate maximum price change in last minute
            max_price = max(p for _, p in self.price_history[symbol])
            min_price = min(p for _, p in self.price_history[symbol])
            price_change = (max_price - min_price) / min_price
            
            return price_change >= self.circuit_breaker_threshold
            
        except Exception as e:
            logging.error(f"Error checking circuit breaker: {e}")
            return False
            
    def run_monte_carlo_simulation(self, symbol: str, current_price: float, volatility: float) -> Dict:
        """Run Monte Carlo simulation to estimate risk of ruin"""
        try:
            current_time = time.time()
            
            # Check if we need to run simulation
            if (self.last_simulation_time and 
                current_time - self.last_simulation_time < self.simulation_interval and
                symbol in self.simulation_results):
                return self.simulation_results[symbol]
                
            # Parameters for simulation
            initial_balance = mt5.account_info().balance
            position_size = initial_balance * 0.01  # 1% risk per trade
            num_trades = 100  # Simulate 100 trades
            win_rate = 0.55  # Assume 55% win rate
            risk_reward = 2.0  # Assume 1:2 risk-reward ratio
            
            # Run simulations
            results = {
                'ruin_probability': 0.0,
                'max_drawdown': 0.0,
                'expected_return': 0.0,
                'sharpe_ratio': 0.0
            }
            
            final_balances = []
            max_drawdowns = []
            
            for _ in range(self.monte_carlo_simulations):
                balance = initial_balance
                peak_balance = initial_balance
                max_drawdown = 0.0
                
                for _ in range(num_trades):
                    # Simulate trade outcome
                    if random.random() < win_rate:
                        # Winning trade
                        balance += position_size * risk_reward
                    else:
                        # Losing trade
                        balance -= position_size
                        
                    # Update peak and drawdown
                    peak_balance = max(peak_balance, balance)
                    current_drawdown = (peak_balance - balance) / peak_balance
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                final_balances.append(balance)
                max_drawdowns.append(max_drawdown)
                
            # Calculate statistics
            results['ruin_probability'] = sum(1 for b in final_balances if b <= initial_balance * 0.5) / self.monte_carlo_simulations
            results['max_drawdown'] = sum(max_drawdowns) / self.monte_carlo_simulations
            results['expected_return'] = (sum(final_balances) / self.monte_carlo_simulations - initial_balance) / initial_balance
            
            # Calculate Sharpe ratio
            returns = [(b - initial_balance) / initial_balance for b in final_balances]
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            results['sharpe_ratio'] = avg_return / std_return if std_return != 0 else 0
            
            # Cache results
            self.simulation_results[symbol] = results
            self.last_simulation_time = current_time
            
            return results
            
        except Exception as e:
            logging.error(f"Error running Monte Carlo simulation: {e}")
            return {
                'ruin_probability': 0.0,
                'max_drawdown': 0.0,
                'expected_return': 0.0,
                'sharpe_ratio': 0.0
            }
            
    def adjust_risk_parameters(self, symbol: str, simulation_results: Dict) -> Dict:
        """Adjust risk parameters based on simulation results"""
        try:
            risk_adjustments = {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0
            }
            
            # Adjust based on ruin probability
            if simulation_results['ruin_probability'] > 0.1:  # 10% ruin probability
                risk_adjustments['position_size_multiplier'] *= 0.5
                risk_adjustments['stop_loss_multiplier'] *= 0.8
                risk_adjustments['take_profit_multiplier'] *= 1.2
            elif simulation_results['ruin_probability'] > 0.05:  # 5% ruin probability
                risk_adjustments['position_size_multiplier'] *= 0.75
                risk_adjustments['stop_loss_multiplier'] *= 0.9
                risk_adjustments['take_profit_multiplier'] *= 1.1
                
            # Adjust based on max drawdown
            if simulation_results['max_drawdown'] > 0.2:  # 20% max drawdown
                risk_adjustments['position_size_multiplier'] *= 0.6
                risk_adjustments['stop_loss_multiplier'] *= 0.7
            elif simulation_results['max_drawdown'] > 0.1:  # 10% max drawdown
                risk_adjustments['position_size_multiplier'] *= 0.8
                risk_adjustments['stop_loss_multiplier'] *= 0.85
                
            # Adjust based on Sharpe ratio
            if simulation_results['sharpe_ratio'] < 0.5:
                risk_adjustments['position_size_multiplier'] *= 0.7
            elif simulation_results['sharpe_ratio'] > 2.0:
                risk_adjustments['position_size_multiplier'] *= 1.2
                
            return risk_adjustments
            
        except Exception as e:
            logging.error(f"Error adjusting risk parameters: {e}")
            return {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0
            }

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
            elif rsi < 30 or rsi > 70:  # Extreme RSI
                self.strategy_weights['mean_reversion'] = 0.4
                self.strategy_weights['trend_following'] = 0.3
                self.strategy_weights['breakout'] = 0.2
                self.strategy_weights['scalping'] = 0.1
            else:  # Normal market conditions
                self.strategy_weights['trend_following'] = 0.4
                self.strategy_weights['mean_reversion'] = 0.3
                self.strategy_weights['breakout'] = 0.2
                self.strategy_weights['scalping'] = 0.1
                
            # Select strategy based on highest weight
            self.current_strategy = max(self.strategy_weights.items(), key=lambda x: x[1])[0]
            return self.current_strategy
            
        except Exception as e:
            logging.error(f"Error determining strategy: {e}")
            return 'trend_following'  # Default to trend following on error
            
    def _trend_following_strategy(self, df: pd.DataFrame) -> Optional[Dict]:
        """Trend following strategy implementation"""
        try:
            # Check for trend conditions
            if df['adx'].iloc[-1] < 25:  # Weak trend
                return None
                
            # Determine trend direction
            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:  # Uptrend
                return {'direction': 'buy', 'confidence': df['adx'].iloc[-1] / 100}
            else:  # Downtrend
                return {'direction': 'sell', 'confidence': df['adx'].iloc[-1] / 100}
                
        except Exception as e:
            logging.error(f"Error in trend following strategy: {e}")
            return None
            
    def _mean_reversion_strategy(self, df: pd.DataFrame) -> Optional[Dict]:
        """Mean reversion strategy implementation"""
        try:
            # Check for mean reversion conditions
            if df['rsi'].iloc[-1] < 30:  # Oversold
                return {'direction': 'buy', 'confidence': (30 - df['rsi'].iloc[-1]) / 30}
            elif df['rsi'].iloc[-1] > 70:  # Overbought
                return {'direction': 'sell', 'confidence': (df['rsi'].iloc[-1] - 70) / 30}
            return None
            
        except Exception as e:
            logging.error(f"Error in mean reversion strategy: {e}")
            return None
            
    def _breakout_strategy(self, df: pd.DataFrame) -> Optional[Dict]:
        """Breakout strategy implementation"""
        try:
            # Check for breakout conditions
            if df['close'].iloc[-1] > df['high'].iloc[-2]:  # Bullish breakout
                return {'direction': 'buy', 'confidence': 0.7}
            elif df['close'].iloc[-1] < df['low'].iloc[-2]:  # Bearish breakout
                return {'direction': 'sell', 'confidence': 0.7}
            return None
            
        except Exception as e:
            logging.error(f"Error in breakout strategy: {e}")
            return None
            
    def _scalping_strategy(self, df: pd.DataFrame) -> Optional[Dict]:
        """Scalping strategy implementation"""
        try:
            # Check for scalping conditions
            if df['atr'].iloc[-1] / df['close'].iloc[-1] < 0.001:  # Low volatility
                return None
                
            # Use Bollinger Bands for scalping
            if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:  # Buy signal
                return {'direction': 'buy', 'confidence': 0.6}
            elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:  # Sell signal
                return {'direction': 'sell', 'confidence': 0.6}
            return None
            
        except Exception as e:
            logging.error(f"Error in scalping strategy: {e}")
            return None

class AdvancedPremiumBot:
    def __init__(self):
        self.symbols = []
        self.positions = {}
        self.websocket_connections = {}
        self.tick_data = {}
        self.quote_history = {}
        self.order_history = []
        self.performance_metrics = {}
        self.risk_parameters = {
            'max_position_size': 10.0,
            'max_daily_trades': 50,
            'max_drawdown_percent': 5.0,
            'risk_per_trade_percent': 1.0
        }
        self.telegram_bot = None
        self.telegram_chat_id = None
        self.running = False
        self.last_state_save = datetime.now()
        self.state_save_interval = 300  # 5 minutes

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