import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from typing import Dict, List
from logger import Logger

class TradingEnv(gym.Env):
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),  # [position_size, action_type]
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # [price, SMA_20, SMA_50, RSI, MACD, balance]
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        return self._get_observation()
        
    def step(self, action):
        # Get current price
        current_price = self.data[self.current_step, 0]
        
        # Execute action
        position_size, action_type = action
        reward = 0
        
        if action_type < 0.33:  # Buy
            if self.position <= 0:
                self.position = position_size
                self.trades.append(('buy', current_price))
        elif action_type > 0.66:  # Sell
            if self.position >= 0:
                self.position = -position_size
                self.trades.append(('sell', current_price))
        else:  # Hold
            pass
            
        # Calculate reward
        if len(self.trades) >= 2:
            last_trade = self.trades[-1]
            prev_trade = self.trades[-2]
            if last_trade[0] != prev_trade[0]:  # Different actions
                reward = (last_trade[1] - prev_trade[1]) * self.position
                
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
        
    def _get_observation(self):
        return np.array([
            self.data[self.current_step, 0],  # price
            self.data[self.current_step, 1],  # SMA_20
            self.data[self.current_step, 2],  # SMA_50
            self.data[self.current_step, 3],  # RSI
            self.data[self.current_step, 4],  # MACD
            self.balance
        ])

class RLAgent:
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger
        self.model = None
        
    def create_env(self, data: np.ndarray) -> gym.Env:
        """Create trading environment"""
        env = TradingEnv(data)
        check_env(env)
        return env
        
    def train(self, data: np.ndarray, total_timesteps: int = 100000):
        """Train the RL agent"""
        try:
            # Create environment
            env = self.create_env(data)
            env = DummyVecEnv([lambda: env])
            
            # Initialize PPO model
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            # Train the model
            self.model.learn(total_timesteps=total_timesteps)
            
        except Exception as e:
            self.logger.error(f"Error training RL agent: {e}")
            
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Generate action from observation"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
                
            action, _ = self.model.predict(observation)
            return action
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return np.array([0, 0.5])  # Default to hold
            
    def save_model(self, path: str):
        """Save trained model"""
        try:
            if self.model is not None:
                self.model.save(path)
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            
    def load_model(self, path: str):
        """Load trained model"""
        try:
            self.model = PPO.load(path)
        except Exception as e:
            self.logger.error(f"Error loading model: {e}") 