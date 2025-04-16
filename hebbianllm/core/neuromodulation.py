"""
Neuromodulation system for the Hebbian SNN.

This module implements the neuromodulation system that modulates
synaptic plasticity based on novelty, surprise, and reward signals.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class NeuromodulationSystem:
    """Neuromodulation system that regulates synaptic plasticity."""
    
    def __init__(self):
        """Initialize the neuromodulation system."""
        # Base modulation levels - increase baseline for stronger learning
        self.baseline_modulation = 1.5
        
        # Specific modulation signals
        self.novelty_signal = 0.2  # Start with some novelty
        self.surprise_signal = 0.0
        self.reward_signal = 0.0
        
        # Parameters for modulation - increase novelty weight for stronger learning
        self.novelty_weight = 8.0  # Increased from 5.0
        self.surprise_weight = 3.0
        self.reward_weight = 2.0
        
        # Signal decay rates
        self.novelty_decay = 0.95
        self.surprise_decay = 0.9
        self.reward_decay = 0.8
        
        # Activity history for novelty detection
        self.activity_history = []
        self.history_window = 100  # Time steps to maintain history
        
        # Prediction system for surprise detection
        self.predicted_activity = None
    
    def update(self, activity: np.ndarray, predicted_activity: Optional[np.ndarray] = None) -> float:
        """
        Update neuromodulatory signals based on current network activity.
        
        Args:
            activity: Current activity pattern (binary spike array)
            predicted_activity: Predicted activity pattern, if available
            
        Returns:
            float: Current modulation level
        """
        # Decay existing signals
        self.novelty_signal *= self.novelty_decay
        self.surprise_signal *= self.surprise_decay
        self.reward_signal *= self.reward_decay
        
        # Update novelty signal based on activity history
        self._update_novelty(activity)
        
        # Update surprise signal if prediction is available
        if predicted_activity is not None:
            self._update_surprise(activity, predicted_activity)
        
        # Compute current modulation level
        modulation = (self.baseline_modulation + 
                      self.novelty_weight * self.novelty_signal +
                      self.surprise_weight * self.surprise_signal +
                      self.reward_weight * self.reward_signal)
        
        # Clip to reasonable range
        modulation = np.clip(modulation, 0.1, 5.0)
        
        return modulation
    
    def _update_novelty(self, activity: np.ndarray):
        """
        Update novelty signal based on how different current activity 
        is from recent history.
        
        Args:
            activity: Current activity pattern
        """
        # Add current activity to history
        self.activity_history.append(activity.copy())
        
        # Maintain history window
        if len(self.activity_history) > self.history_window:
            self.activity_history.pop(0)
        
        # Not enough history to calculate novelty
        if len(self.activity_history) < 5:
            self.novelty_signal = 0.5  # Medium novelty when starting
            return
        
        # Calculate average similarity to recent patterns
        similarity_sum = 0.0
        num_comparisons = min(10, len(self.activity_history) - 1)
        
        for i in range(1, num_comparisons + 1):
            past_activity = self.activity_history[-i-1]
            
            # Calculate Jaccard similarity: intersection over union
            intersection = np.sum(activity & past_activity)
            union = np.sum(activity | past_activity)
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 1.0  # Both patterns empty
                
            similarity_sum += similarity
        
        # Average similarity
        avg_similarity = similarity_sum / num_comparisons
        
        # Novelty is inverse of similarity
        self.novelty_signal = max(0.0, 1.0 - avg_similarity)
    
    def _update_surprise(self, activity: np.ndarray, predicted_activity: np.ndarray):
        """
        Update surprise signal based on prediction error.
        
        Args:
            activity: Actual activity pattern
            predicted_activity: Predicted activity pattern
        """
        # Calculate prediction error (normalized Hamming distance)
        prediction_error = np.mean(np.abs(activity - predicted_activity))
        
        # Update surprise signal based on prediction error
        self.surprise_signal = prediction_error
    
    def set_reward(self, reward: float):
        """
        Set reward signal.
        
        Args:
            reward: Reward value (0.0 to 1.0)
        """
        self.reward_signal = np.clip(reward, 0.0, 1.0)
    
    def get_modulation(self) -> Dict[str, float]:
        """
        Get current modulation state.
        
        Returns:
            Dict with modulation values
        """
        return {
            'total': self.baseline_modulation + 
                     self.novelty_weight * self.novelty_signal +
                     self.surprise_weight * self.surprise_signal +
                     self.reward_weight * self.reward_signal,
            'baseline': self.baseline_modulation,
            'novelty': self.novelty_signal,
            'surprise': self.surprise_signal,
            'reward': self.reward_signal
        } 