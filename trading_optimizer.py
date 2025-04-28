import numpy as np
from typing import List, Dict
import logging

def calculate_twap(prices: List[float], window: int) -> float:
    """Calculate Time-Weighted Average Price"""
    try:
        if not prices:
            return 0.0
            
        # Convert to numpy array for efficient calculations
        prices_array = np.array(prices)
        
        # Calculate weights based on time
        weights = np.linspace(1, 0, len(prices_array))
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted average
        twap = np.sum(prices_array * weights)
        
        return float(twap)
        
    except Exception as e:
        logging.error(f"Error in TWAP calculation: {e}")
        return 0.0

def optimize_order_split(total_size: float, broker_params: List[Dict]) -> Dict[str, float]:
    """Optimize order split across brokers"""
    try:
        if not broker_params:
            return {}
            
        # Calculate broker scores based on slippage and latency
        scores = []
        for param in broker_params:
            score = 1 / (param['slippage'] * param['latency'])
            scores.append(score)
            
        # Normalize scores
        total_score = sum(scores)
        if total_score == 0:
            return {}
            
        # Calculate splits based on normalized scores
        splits = {}
        for i, param in enumerate(broker_params):
            broker_id = param['id']
            normalized_score = scores[i] / total_score
            split_size = total_size * normalized_score
            
            # Ensure split size is within broker limits
            split_size = max(param['min_lot'], min(param['max_lot'], split_size))
            splits[broker_id] = split_size
            
        return splits
        
    except Exception as e:
        logging.error(f"Error in order split optimization: {e}")
        return {} 