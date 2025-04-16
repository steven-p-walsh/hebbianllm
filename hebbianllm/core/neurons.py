"""
Neuron implementation for Hebbian SNN.

This module contains the implementations of various neuron types
used in the Hebbian SNN for language modeling.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class Neuron:
    """Base neuron class for Hebbian SNN."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 resting_potential: float = 0.0,
                 refractory_period: float = 2.0,
                 membrane_time_constant: float = 20.0):
        """
        Initialize a basic neuron.
        
        Args:
            threshold: Firing threshold
            resting_potential: Resting membrane potential
            refractory_period: Refractory period in ms
            membrane_time_constant: Membrane time constant in ms
        """
        self.threshold = threshold
        self.resting_potential = resting_potential
        self.refractory_period = refractory_period
        self.membrane_time_constant = membrane_time_constant
        
        # State variables
        self.membrane_potential = resting_potential
        self.last_spike_time = -1000.0  # Initialize to a large negative value
        self.is_refractory = False
        
        # STDP traces
        self.pre_trace = 0.0
        self.post_trace = 0.0
        
        # Accumulated input current for the next update step
        self.input_current = 0.0
    
    def reset(self):
        """Reset neuron state."""
        self.membrane_potential = self.resting_potential
        self.last_spike_time = -1000.0
        self.is_refractory = False
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.input_current = 0.0
    
    def receive_input(self, input_current: float, current_time: float) -> bool:
        """
        Process incoming input current and update neuron state.
        
        Args:
            input_current: Input current to add to membrane potential
            current_time: Current simulation time in ms
            
        Returns:
            bool: True if the neuron fired, False otherwise
        """
        # Check if neuron is in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.is_refractory = True
            return False
        
        self.is_refractory = False
        
        # Accumulate input current for the update step
        self.input_current += input_current
        
        return False
    
    def update(self, dt: float, current_time: float) -> bool:
        """
        Update neuron state for the given time step.
        
        Args:
            dt: Time step in ms
            current_time: Current simulation time in ms
            
        Returns:
            bool: True if the neuron fired, False otherwise
        """
        # Decay membrane potential towards resting potential + apply input current
        # dv/dt = -(v - v_rest)/tau_m + I(t)/C. Assume C=1.
        dv = dt * (-(self.membrane_potential - self.resting_potential) / self.membrane_time_constant + self.input_current)
        self.membrane_potential += dv
        
        # Reset input current for next timestep
        self.input_current = 0.0
        
        # Decay STDP traces
        tau_plus = 20.0  # ms
        tau_minus = 20.0  # ms
        self.pre_trace *= jnp.exp(-dt / tau_plus)
        self.post_trace *= jnp.exp(-dt / tau_minus)
        
        # No input current in this update, so just check threshold
        if not self.is_refractory and self.membrane_potential >= self.threshold:
            self.membrane_potential = self.resting_potential  # Reset after spike
            self.last_spike_time = current_time
            self.post_trace += 1.0  # Update post-synaptic trace
            return True
            
        # Check if neuron is in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.is_refractory = True
        else:
            self.is_refractory = False
            
        return False


class SensoryNeuron(Neuron):
    """Sensory neuron implementation for input processing."""
    
    def __init__(self, token_id: int = None, **kwargs):
        """
        Initialize a sensory neuron.
        
        Args:
            token_id: ID of the token this neuron responds to
            **kwargs: Arguments passed to parent Neuron class
        """
        # Default parameters for sensory neurons from the spec
        kwargs.setdefault('threshold', 0.5)
        kwargs.setdefault('resting_potential', 0.0)
        kwargs.setdefault('refractory_period', 2.0)
        
        super().__init__(**kwargs)
        self.token_id = token_id


class AssociativeNeuron(Neuron):
    """Associative neuron for recurrent processing."""
    
    def __init__(self, **kwargs):
        """
        Initialize an associative neuron.
        
        Args:
            **kwargs: Arguments passed to parent Neuron class
        """
        # Default parameters for associative neurons from the spec
        kwargs.setdefault('threshold', 0.6)
        kwargs.setdefault('resting_potential', -0.1)
        kwargs.setdefault('refractory_period', 4.0)
        
        super().__init__(**kwargs)
        
        # Adaptive threshold parameters
        self.adaptation_time_constant = 500.0  # ms
        self.theta = 0.0  # Threshold adaptation
        self.theta_increment = 0.1  # Increment after spike
    
    def update(self, dt: float, current_time: float) -> bool:
        """
        Update associative neuron state with adaptive threshold.
        
        Args:
            dt: Time step in ms
            current_time: Current simulation time in ms
            
        Returns:
            bool: True if the neuron fired, False otherwise
        """
        # Decay adaptive threshold
        self.theta *= jnp.exp(-dt / self.adaptation_time_constant)
        
        # Decay membrane potential towards resting potential + apply input current
        dv = dt * (-(self.membrane_potential - self.resting_potential) / self.membrane_time_constant + self.input_current)
        self.membrane_potential += dv
        
        # Reset input current for next timestep
        self.input_current = 0.0
        
        # Decay STDP traces
        tau_plus = 20.0  # ms
        tau_minus = 20.0  # ms
        self.pre_trace *= jnp.exp(-dt / tau_plus)
        self.post_trace *= jnp.exp(-dt / tau_minus)
        
        # Check if neuron is in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.is_refractory = True
            return False
            
        self.is_refractory = False
        
        # Check threshold with adaptation
        if self.membrane_potential >= (self.threshold + self.theta):
            self.membrane_potential = self.resting_potential  # Reset after spike
            self.last_spike_time = current_time
            self.theta += self.theta_increment  # Increase adaptive threshold
            self.post_trace += 1.0  # Update post-synaptic trace
            return True
            
        return False


class InhibitoryNeuron(Neuron):
    """Inhibitory neuron for network regulation."""
    
    def __init__(self, **kwargs):
        """
        Initialize an inhibitory neuron.
        
        Args:
            **kwargs: Arguments passed to parent Neuron class
        """
        # Default parameters for inhibitory neurons from the spec
        kwargs.setdefault('threshold', 0.4)
        kwargs.setdefault('resting_potential', -0.2)
        kwargs.setdefault('refractory_period', 3.0)
        
        super().__init__(**kwargs)
        self.inhibition_strength = 1.2  # Reduced from 2.0 to allow better signal propagation


class OutputNeuron(Neuron):
    """Output neuron for generating language outputs."""
    
    def __init__(self, token_id: int = None, **kwargs):
        """
        Initialize an output neuron.
        
        Args:
            token_id: ID of the token this neuron outputs
            **kwargs: Arguments passed to parent Neuron class
        """
        # Default parameters for output neurons from the spec
        kwargs.setdefault('threshold', 0.8)
        kwargs.setdefault('resting_potential', -0.1)
        kwargs.setdefault('refractory_period', 5.0)
        
        super().__init__(**kwargs)
        self.token_id = token_id 