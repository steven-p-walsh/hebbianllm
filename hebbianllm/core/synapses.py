"""
Synapse implementation for Hebbian SNN.

This module implements synaptic connections and STDP learning.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from hebbianllm.core.neurons import Neuron


class Synapse:
    """Basic synapse with STDP learning."""
    
    def __init__(self, 
                 pre_neuron: Neuron,
                 post_neuron: Neuron,
                 weight: float = 0.1,
                 delay: float = 1.0,
                 plastic: bool = True):
        """
        Initialize a synapse between two neurons.
        
        Args:
            pre_neuron: Pre-synaptic neuron
            post_neuron: Post-synaptic neuron
            weight: Initial synaptic weight
            delay: Synaptic delay in ms
            plastic: Whether the synapse is plastic (can learn)
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.plastic = plastic
        
        # STDP parameters (Strongly favor LTP over LTD)
        self.A_plus = 0.08   # Significantly increased LTP factor
        self.A_minus = 0.02  # Significantly reduced LTD factor
        self.tau_plus = 20.0  # LTP time constant (ms)
        self.tau_minus = 20.0 # LTD time constant (ms)
        
        # Synapse metadata
        self.creation_time = 0.0
        self.last_update = 0.0
        self.activation_count = 0
        self.plasticity_regime = "fast"  # "fast" or "slow"
        
    def transmit_spike(self, current_time: float) -> Tuple[float, float]:
        """
        Transmit a spike from pre- to post-synaptic neuron.
        
        Args:
            current_time: Current simulation time in ms
            
        Returns:
            Tuple[float, float]: The weight and delivery time
        """
        delivery_time = current_time + self.delay
        
        # Update activation count
        self.activation_count += 1
        
        # Update pre-synaptic trace for STDP
        self.pre_neuron.pre_trace += 1.0
        
        return self.weight, delivery_time
    
    def apply_stdp(self, current_time: float, modulation: float = 1.0) -> float:
        """
        Apply STDP learning rule based on trace values.
        
        Args:
            current_time: Current simulation time in ms
            modulation: Neuromodulatory signal to scale plasticity
            
        Returns:
            float: The weight change (delta w)
        """
        if not self.plastic:
            return 0.0
        
        # Select STDP parameters based on plasticity regime
        if self.plasticity_regime == "slow":
            a_plus = self.A_plus * 0.2   # 20% of fast learning rate
            a_minus = self.A_minus * 0.2
            tau_plus = self.tau_plus * 2.0
            tau_minus = self.tau_minus * 2.0
        else:
            a_plus = self.A_plus
            a_minus = self.A_minus
            tau_plus = self.tau_plus
            tau_minus = self.tau_minus
        
        # Calculate weight change based on trace values
        dw = 0.0
        
        # LTP: Pre-synaptic activity followed by post-synaptic activity
        dw += a_plus * self.pre_neuron.pre_trace * self.post_neuron.post_trace
        
        # LTD: Post-synaptic activity followed by pre-synaptic activity
        dw -= a_minus * self.post_neuron.post_trace * self.pre_neuron.pre_trace
        
        # Apply neuromodulation
        dw *= modulation
        
        # Apply weight change with bounds
        old_weight = self.weight
        self.weight = jnp.clip(self.weight + dw, 0.0, 1.0)
        
        # Update last update time
        self.last_update = current_time
        
        return self.weight - old_weight
    
    def check_consolidation(self, min_age: float = 1000.0, threshold: int = 50) -> bool:
        """
        Check if synapse should be consolidated (moved to slow plasticity regime).
        
        Args:
            min_age: Minimum age in ms before consolidation
            threshold: Activation count threshold for consolidation
            
        Returns:
            bool: True if synapse should be consolidated
        """
        if (self.plasticity_regime == "fast" and 
            self.activation_count > threshold and 
            (self.last_update - self.creation_time) > min_age):
            
            self.plasticity_regime = "slow"
            return True
        
        return False


class SparseConnectivity:
    """Sparse connectivity manager for efficient synapse storage."""
    
    def __init__(self, 
                 n_neurons: int,
                 connectivity_density: float = 0.1,
                 seed: int = 42):
        """
        Initialize sparse connectivity.
        
        Args:
            n_neurons: Number of neurons in the network
            connectivity_density: Proportion of possible connections to create
            seed: Random seed for initialization
        """
        self.n_neurons = n_neurons
        self.connectivity_density = connectivity_density
        self.rng = np.random.RandomState(seed)
        
        # Sparse representation
        self.initialize_connectivity()
    
    def initialize_connectivity(self):
        """Initialize random sparse connectivity."""
        # Calculate number of connections
        n_connections = int(self.n_neurons * self.n_neurons * self.connectivity_density)
        
        # Generate random pre-post pairs
        pre_indices = self.rng.randint(0, self.n_neurons, n_connections)
        post_indices = self.rng.randint(0, self.n_neurons, n_connections)
        
        # Filter out self-connections
        mask = pre_indices != post_indices
        pre_indices = pre_indices[mask]
        post_indices = post_indices[mask]
        
        # Initialize weights with log-normal distribution
        weights = np.exp(self.rng.normal(-1.5, 0.5, len(pre_indices)))
        weights = np.clip(weights, 0.01, 1.0)
        
        # Initialize delays (1-5ms uniform)
        delays = self.rng.uniform(1.0, 5.0, len(pre_indices))
        
        # Store in COO format
        self.pre_indices = pre_indices
        self.post_indices = post_indices
        self.weights = weights
        self.delays = delays
        
        # Additional synaptic properties
        self.plasticity_regime = np.array(['fast'] * len(pre_indices))
        self.creation_time = np.zeros(len(pre_indices))
        self.last_update = np.zeros(len(pre_indices))
        self.activation_count = np.zeros(len(pre_indices), dtype=np.int32)
    
    def get_connections(self, neuron_idx: int) -> List[Tuple[int, float, float]]:
        """
        Get all outgoing connections for a neuron.
        
        Args:
            neuron_idx: Index of the pre-synaptic neuron
            
        Returns:
            List of (post_idx, weight, delay) tuples
        """
        # Find all connections where pre_idx matches neuron_idx
        connections = []
        for i, pre_idx in enumerate(self.pre_indices):
            if pre_idx == neuron_idx:
                connections.append((
                    self.post_indices[i],
                    self.weights[i],
                    self.delays[i]
                ))
        
        return connections
    
    def add_synapse(self, pre_idx: int, post_idx: int, weight: float = 0.1, delay: float = 1.0):
        """
        Add a new synapse to the connectivity matrix.
        
        Args:
            pre_idx: Pre-synaptic neuron index
            post_idx: Post-synaptic neuron index
            weight: Initial weight
            delay: Synaptic delay in ms
        """
        # Add to COO format
        self.pre_indices = np.append(self.pre_indices, pre_idx)
        self.post_indices = np.append(self.post_indices, post_idx)
        self.weights = np.append(self.weights, weight)
        self.delays = np.append(self.delays, delay)
        
        # Add synaptic properties
        self.plasticity_regime = np.append(self.plasticity_regime, 'fast')
        self.creation_time = np.append(self.creation_time, 0.0)  # Should be current time
        self.last_update = np.append(self.last_update, 0.0)  # Should be current time
        self.activation_count = np.append(self.activation_count, 0)
    
    def remove_synapse(self, pre_idx: int, post_idx: int):
        """
        Remove a synapse from the connectivity matrix.
        
        Args:
            pre_idx: Pre-synaptic neuron index
            post_idx: Post-synaptic neuron index
        """
        # Find the synapse
        mask = (self.pre_indices == pre_idx) & (self.post_indices == post_idx)
        
        if np.any(mask):
            # Remove from all arrays
            self.pre_indices = self.pre_indices[~mask]
            self.post_indices = self.post_indices[~mask]
            self.weights = self.weights[~mask]
            self.delays = self.delays[~mask]
            
            # Remove synaptic properties
            self.plasticity_regime = self.plasticity_regime[~mask]
            self.creation_time = self.creation_time[~mask]
            self.last_update = self.last_update[~mask]
            self.activation_count = self.activation_count[~mask]
    
    def update_weight(self, pre_idx: int, post_idx: int, new_weight: float):
        """
        Update weight of a specific synapse.
        
        Args:
            pre_idx: Pre-synaptic neuron index
            post_idx: Post-synaptic neuron index
            new_weight: New weight value
        """
        # Find the synapse
        mask = (self.pre_indices == pre_idx) & (self.post_indices == post_idx)
        
        if np.any(mask):
            # Update weight
            self.weights[mask] = new_weight
    
    def get_weight(self, pre_idx: int, post_idx: int) -> Optional[float]:
        """
        Get weight of a specific synapse.
        
        Args:
            pre_idx: Pre-synaptic neuron index
            post_idx: Post-synaptic neuron index
            
        Returns:
            Weight of the synapse or None if it doesn't exist
        """
        # Find the synapse
        mask = (self.pre_indices == pre_idx) & (self.post_indices == post_idx)
        
        if np.any(mask):
            return self.weights[mask][0]
        
        return None 