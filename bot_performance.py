import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

class BotPerformance:
    def __init__(self):
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'win_rate': 0.0,
            'risk_reward_ratio': 0.0,
            'daily_returns': [],
            'monthly_returns': []
        }
        
        # Rating criteria weights
        self.rating_weights = {
            'win_rate': 0.25,
            'profit_factor': 0.20,
            'max_drawdown': 0.20,
            'sharpe_ratio': 0.15,
            'risk_reward_ratio': 0.10,
            'consistency': 0.10
        }
        
        # Rating thresholds
        self.rating_thresholds = {
            'win_rate': {
                'excellent': 0.60,
                'good': 0.55,
                'average': 0.50,
                'poor': 0.45
            },
            'profit_factor': {
                'excellent': 2.0,
                'good': 1.5,
                'average': 1.2,
                'poor': 1.0
            },
            'max_drawdown': {
                'excellent': 0.05,  # 5%
                'good': 0.10,      # 10%
                'average': 0.15,   # 15%
                'poor': 0.20       # 20%
            },
            'sharpe_ratio': {
                'excellent': 2.0,
                'good': 1.5,
                'average': 1.0,
                'poor': 0.5
            },
            'risk_reward_ratio': {
                'excellent': 3.0,
                'good': 2.5,
                'average': 2.0,
                'poor': 1.5
            }
        }
        
    def add_trade(self, trade_data: Dict):
        """Add a new trade to the performance tracking"""
        try:
            self.trades.append(trade_data)
            self._update_metrics()
        except Exception as e:
            logging.error(f"Error adding trade: {e}")
            
    def _update_metrics(self):
        """Update all performance metrics"""
        try:
            if not self.trades:
                return
                
            # Convert trades to DataFrame for easier calculations
            df = pd.DataFrame(self.trades)
            
            # Basic metrics
            self.performance_metrics['total_trades'] = len(df)
            self.performance_metrics['winning_trades'] = len(df[df['profit'] > 0])
            self.performance_metrics['losing_trades'] = len(df[df['profit'] < 0])
            self.performance_metrics['total_profit'] = df[df['profit'] > 0]['profit'].sum()
            self.performance_metrics['total_loss'] = abs(df[df['profit'] < 0]['profit'].sum())
            
            # Win rate
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
            
            # Profit factor
            self.performance_metrics['profit_factor'] = (
                self.performance_metrics['total_profit'] / 
                self.performance_metrics['total_loss'] if self.performance_metrics['total_loss'] > 0 else float('inf')
            )
            
            # Average win/loss
            self.performance_metrics['average_win'] = (
                df[df['profit'] > 0]['profit'].mean() if len(df[df['profit'] > 0]) > 0 else 0
            )
            self.performance_metrics['average_loss'] = (
                abs(df[df['profit'] < 0]['profit'].mean()) if len(df[df['profit'] < 0]) > 0 else 0
            )
            
            # Risk-reward ratio
            if self.performance_metrics['average_loss'] > 0:
                self.performance_metrics['risk_reward_ratio'] = (
                    self.performance_metrics['average_win'] / 
                    self.performance_metrics['average_loss']
                )
                
            # Calculate drawdown
            df['cumulative_profit'] = df['profit'].cumsum()
            df['peak'] = df['cumulative_profit'].cummax()
            df['drawdown'] = (df['peak'] - df['cumulative_profit']) / df['peak']
            self.performance_metrics['max_drawdown'] = df['drawdown'].max()
            
            # Calculate Sharpe ratio
            if len(df) > 1:
                returns = df['profit'].pct_change().dropna()
                if len(returns) > 0:
                    self.performance_metrics['sharpe_ratio'] = (
                        returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    )
                    
            # Calculate daily and monthly returns
            df['date'] = pd.to_datetime(df['timestamp'])
            daily_returns = df.groupby(df['date'].dt.date)['profit'].sum()
            monthly_returns = df.groupby(df['date'].dt.to_period('M'))['profit'].sum()
            
            self.performance_metrics['daily_returns'] = daily_returns.tolist()
            self.performance_metrics['monthly_returns'] = monthly_returns.tolist()
            
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")
            
    def calculate_rating(self) -> Tuple[float, str]:
        """Calculate overall bot rating"""
        try:
            # Calculate individual component scores
            scores = {
                'win_rate': self._calculate_component_score(
                    'win_rate', 
                    self.performance_metrics['win_rate']
                ),
                'profit_factor': self._calculate_component_score(
                    'profit_factor', 
                    self.performance_metrics['profit_factor']
                ),
                'max_drawdown': self._calculate_component_score(
                    'max_drawdown', 
                    self.performance_metrics['max_drawdown']
                ),
                'sharpe_ratio': self._calculate_component_score(
                    'sharpe_ratio', 
                    self.performance_metrics['sharpe_ratio']
                ),
                'risk_reward_ratio': self._calculate_component_score(
                    'risk_reward_ratio', 
                    self.performance_metrics['risk_reward_ratio']
                ),
                'consistency': self._calculate_consistency_score()
            }
            
            # Calculate weighted average
            total_score = sum(
                score * self.rating_weights[component]
                for component, score in scores.items()
            )
            
            # Determine rating category
            if total_score >= 0.9:
                rating_category = "Excellent"
            elif total_score >= 0.8:
                rating_category = "Very Good"
            elif total_score >= 0.7:
                rating_category = "Good"
            elif total_score >= 0.6:
                rating_category = "Average"
            else:
                rating_category = "Poor"
                
            return total_score, rating_category
            
        except Exception as e:
            logging.error(f"Error calculating rating: {e}")
            return 0.0, "Not Rated"
            
    def _calculate_component_score(self, component: str, value: float) -> float:
        """Calculate score for a specific component"""
        try:
            thresholds = self.rating_thresholds[component]
            
            if value >= thresholds['excellent']:
                return 1.0
            elif value >= thresholds['good']:
                return 0.8
            elif value >= thresholds['average']:
                return 0.6
            elif value >= thresholds['poor']:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logging.error(f"Error calculating component score: {e}")
            return 0.0
            
    def _calculate_consistency_score(self) -> float:
        """Calculate consistency score based on returns distribution"""
        try:
            if not self.performance_metrics['daily_returns']:
                return 0.0
                
            returns = np.array(self.performance_metrics['daily_returns'])
            
            # Calculate positive return days percentage
            positive_days = np.sum(returns > 0) / len(returns)
            
            # Calculate return volatility
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Combine factors for consistency score
            consistency_score = (positive_days * 0.7) + (1 / (1 + volatility) * 0.3)
            
            return min(max(consistency_score, 0), 1)
            
        except Exception as e:
            logging.error(f"Error calculating consistency score: {e}")
            return 0.0
            
    def generate_performance_report(self) -> str:
        """Generate detailed performance report"""
        try:
            score, rating = self.calculate_rating()
            
            report = (
                f"🤖 Trading Bot Performance Report\n\n"
                f"📊 Overall Rating: {rating} ({score:.2f})\n\n"
                f"📈 Performance Metrics:\n"
                f"  • Total Trades: {self.performance_metrics['total_trades']}\n"
                f"  • Win Rate: {self.performance_metrics['win_rate']:.2%}\n"
                f"  • Profit Factor: {self.performance_metrics['profit_factor']:.2f}\n"
                f"  • Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}\n"
                f"  • Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}\n"
                f"  • Risk/Reward Ratio: {self.performance_metrics['risk_reward_ratio']:.2f}\n\n"
                f"💰 Profit Analysis:\n"
                f"  • Total Profit: {self.performance_metrics['total_profit']:.2f}\n"
                f"  • Total Loss: {self.performance_metrics['total_loss']:.2f}\n"
                f"  • Average Win: {self.performance_metrics['average_win']:.2f}\n"
                f"  • Average Loss: {self.performance_metrics['average_loss']:.2f}\n\n"
                f"📅 Trading Consistency:\n"
                f"  • Daily Returns: {len(self.performance_metrics['daily_returns'])} days\n"
                f"  • Monthly Returns: {len(self.performance_metrics['monthly_returns'])} months"
            )
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating performance report: {e}")
            return "Error generating performance report" 