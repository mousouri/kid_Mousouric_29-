import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import tensorflow as tf
from tensorflow import keras
import torch
from transformers import TFAutoModelForSequenceClassification
from logger import Logger

class EnsembleModel:
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger
        self.models = {}
        self.feature_importance = {}
        
    def create_lstm_model(self, input_shape: Tuple) -> keras.Sequential:
        """Create LSTM model for time series prediction"""
        model = keras.Sequential([
            keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(units=50, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
        
    def create_transformer_model(self, input_shape: Tuple):
        """Create Temporal Fusion Transformer model"""
        # Note: This is a simplified version. You might want to use a proper TFT implementation
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "tft-base",
            num_labels=1
        )
        return model
        
    def create_tree_models(self):
        """Create Random Forest and Gradient Boosting models"""
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        gbm_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        return rf_model, gbm_model
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        # Technical indicators
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['RSI'] = self._calculate_rsi(data['close'])
        data['MACD'] = self._calculate_macd(data['close'])
        
        # Remove NaN values
        data = data.dropna()
        
        # Prepare features and target
        features = data[['SMA_20', 'SMA_50', 'RSI', 'MACD']].values
        target = data['close'].shift(-1).values[:-1]  # Next day's price
        
        return features[:-1], target
        
    def train_ensemble(self, data: pd.DataFrame):
        """Train all models in the ensemble"""
        try:
            # Prepare data
            X, y = self.prepare_features(data)
            
            # Train LSTM
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            lstm_model = self.create_lstm_model((1, X.shape[1]))
            lstm_model.fit(X_lstm, y, epochs=50, batch_size=32, verbose=0)
            self.models['lstm'] = lstm_model
            
            # Train Transformer
            transformer_model = self.create_transformer_model(X.shape)
            transformer_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            self.models['transformer'] = transformer_model
            
            # Train tree-based models
            rf_model, gbm_model = self.create_tree_models()
            rf_model.fit(X, y)
            gbm_model.fit(X, y)
            self.models['random_forest'] = rf_model
            self.models['gradient_boosting'] = gbm_model
            
            # Calculate feature importance
            self._calculate_feature_importance(X, y)
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            
    def predict(self, data: pd.DataFrame) -> Dict:
        """Generate predictions from all models"""
        try:
            X, _ = self.prepare_features(data)
            predictions = {}
            
            # LSTM prediction
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            predictions['lstm'] = self.models['lstm'].predict(X_lstm)
            
            # Transformer prediction
            predictions['transformer'] = self.models['transformer'].predict(X)
            
            # Tree-based predictions
            predictions['random_forest'] = self.models['random_forest'].predict(X)
            predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(X)
            
            # Ensemble prediction (weighted average)
            weights = self._get_model_weights()
            ensemble_pred = np.zeros_like(predictions['lstm'])
            for model_name, pred in predictions.items():
                ensemble_pred += pred * weights[model_name]
                
            predictions['ensemble'] = ensemble_pred
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            return {}
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
        
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importance from tree-based models"""
        rf_model = self.models['random_forest']
        gbm_model = self.models['gradient_boosting']
        
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        self.feature_importance['gradient_boosting'] = gbm_model.feature_importances_
        
    def _get_model_weights(self) -> Dict[str, float]:
        """Get model weights based on performance"""
        # This is a simplified version. You might want to implement
        # a more sophisticated weight calculation based on validation performance
        return {
            'lstm': 0.3,
            'transformer': 0.3,
            'random_forest': 0.2,
            'gradient_boosting': 0.2
        } 