import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

class BotPerformance:
    def __init__(self):
        self.trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'scalping_performance': {
                'scalping_trades': 0,
                'scalping_win_rate': 0.0,
                'scalping_profit': 0.0,
                'average_holding_time': 0.0
            }
        }
        self.rating_criteria = {
            'win_rate': {'weight': 0.2, 'thresholds': [0.4, 0.5, 0.6, 0.7]},
            'profit_factor': {'weight': 0.2, 'thresholds': [1.0, 1.5, 2.0, 2.5]},
            'max_drawdown': {'weight': 0.15, 'thresholds': [0.1, 0.08, 0.06, 0.04]},
            'sharpe_ratio': {'weight': 0.15, 'thresholds': [0.5, 1.0, 1.5, 2.0]},
            'scalping_win_rate': {'weight': 0.15, 'thresholds': [0.45, 0.55, 0.65, 0.75]},
            'average_holding_time': {'weight': 0.15, 'thresholds': [30, 20, 15, 10]}  # in minutes
        }

    def add_trade(self, trade_data: Dict):
        """Add a new trade to performance tracking"""
        self.trades.append({
            'time': datetime.now(),
            'symbol': trade_data['symbol'],
            'direction': trade_data['direction'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data['exit_price'],
            'profit': trade_data['profit'],
            'holding_time': trade_data['holding_time'],
            'strategy': trade_data['strategy']
        })
        self._update_metrics()

    def _update_metrics(self):
        """Update performance metrics based on trade history"""
        if not self.trades:
            return

        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        self.performance_metrics['total_trades'] = len(df)
        self.performance_metrics['winning_trades'] = len(df[df['profit'] > 0])
        self.performance_metrics['losing_trades'] = len(df[df['profit'] < 0])
        self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
        
        # Profit metrics
        self.performance_metrics['total_profit'] = df['profit'].sum()
        self.performance_metrics['average_win'] = df[df['profit'] > 0]['profit'].mean()
        self.performance_metrics['average_loss'] = df[df['profit'] < 0]['profit'].mean()
        
        # Risk metrics
        winning_profits = df[df['profit'] > 0]['profit'].sum()
        losing_profits = abs(df[df['profit'] < 0]['profit'].sum())
        self.performance_metrics['profit_factor'] = winning_profits / losing_profits if losing_profits > 0 else float('inf')
        
        # Calculate max drawdown
        cumulative_returns = df['profit'].cumsum()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - rolling_max
        self.performance_metrics['max_drawdown'] = abs(drawdowns.min() / rolling_max.max()) if rolling_max.max() > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = df['profit'].pct_change()
        self.performance_metrics['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Scalping specific metrics
        scalping_trades = df[df['strategy'] == 'scalping']
        self.performance_metrics['scalping_performance']['scalping_trades'] = len(scalping_trades)
        self.performance_metrics['scalping_performance']['scalping_win_rate'] = len(scalping_trades[scalping_trades['profit'] > 0]) / len(scalping_trades) if len(scalping_trades) > 0 else 0
        self.performance_metrics['scalping_performance']['scalping_profit'] = scalping_trades['profit'].sum()
        self.performance_metrics['scalping_performance']['average_holding_time'] = scalping_trades['holding_time'].mean() if len(scalping_trades) > 0 else 0

    def calculate_rating(self) -> Dict:
        """Calculate overall bot rating"""
        rating = 0.0
        detailed_ratings = {}
        
        for criterion, config in self.rating_criteria.items():
            value = self.performance_metrics.get(criterion, 0)
            if criterion == 'max_drawdown':
                value = -value  # Convert to positive for rating calculation
                
            # Calculate score based on thresholds
            score = 0
            for i, threshold in enumerate(config['thresholds']):
                if value >= threshold:
                    score = i + 1
                    
            # Normalize score to 0-1 range
            normalized_score = score / len(config['thresholds'])
            detailed_ratings[criterion] = {
                'score': normalized_score,
                'value': value
            }
            rating += normalized_score * config['weight']
            
        # Convert to letter grade
        if rating >= 0.9:
            grade = 'A+'
        elif rating >= 0.8:
            grade = 'A'
        elif rating >= 0.7:
            grade = 'B'
        elif rating >= 0.6:
            grade = 'C'
        elif rating >= 0.5:
            grade = 'D'
        else:
            grade = 'F'
            
        return {
            'overall_rating': rating,
            'grade': grade,
            'detailed_ratings': detailed_ratings,
            'performance_metrics': self.performance_metrics
        }

    def get_performance_report(self) -> str:
        """Generate a detailed performance report"""
        rating = self.calculate_rating()
        
        report = (
            f"🤖 Bot Performance Report\n\n"
            f"📊 Overall Rating: {rating['grade']} ({rating['overall_rating']:.2f})\n\n"
            f"📈 Trading Performance:\n"
            f"  - Total Trades: {self.performance_metrics['total_trades']}\n"
            f"  - Win Rate: {self.performance_metrics['win_rate']:.2%}\n"
            f"  - Profit Factor: {self.performance_metrics['profit_factor']:.2f}\n"
            f"  - Total Profit: {self.performance_metrics['total_profit']:.2f}\n"
            f"  - Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}\n"
            f"  - Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}\n\n"
            f"🎯 Scalping Performance:\n"
            f"  - Scalping Trades: {self.performance_metrics['scalping_performance']['scalping_trades']}\n"
            f"  - Scalping Win Rate: {self.performance_metrics['scalping_performance']['scalping_win_rate']:.2%}\n"
            f"  - Scalping Profit: {self.performance_metrics['scalping_performance']['scalping_profit']:.2f}\n"
            f"  - Avg Holding Time: {self.performance_metrics['scalping_performance']['average_holding_time']:.1f} minutes\n\n"
            f"📋 Detailed Ratings:\n"
        )
        
        for criterion, details in rating['detailed_ratings'].items():
            report += f"  - {criterion}: {details['score']:.2f} ({details['value']:.2f})\n"
            
        return report 