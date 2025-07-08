"""
JAX-optimized operations for Hebbian SNN.

This module provides GPU/Metal-accelerated operations for the Hebbian SNN
with vectorized implementations of neural dynamics and learning rules.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import Tuple, Dict, Any, Optional
import numpy as np

# Configure JAX for best available backend
available_backends = jax.lib.xla_bridge.get_backend().platform
if 'gpu' in str(jax.devices()).lower() or 'metal' in str(jax.devices()).lower():
    try:
        jax.config.update('jax_platform_name', 'gpu')
        print("JAX configured for GPU/Metal backend")
    except:
        jax.config.update('jax_platform_name', 'cpu')
        print("JAX configured for CPU backend (GPU/Metal not available)")
else:
    jax.config.update('jax_platform_name', 'cpu')
    print("JAX configured for CPU backend")


@jit
def leaky_integrate_fire(v: jnp.ndarray, 
                        i_input: jnp.ndarray,
                        threshold: jnp.ndarray,
                        resting_potential: jnp.ndarray,
                        tau_m: jnp.ndarray,
                        refractory_mask: jnp.ndarray,
                        dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized leaky integrate-and-fire neuron dynamics.
    
    Args:
        v: Membrane potentials (n_neurons,)
        i_input: Input currents (n_neurons,)
        threshold: Firing thresholds (n_neurons,)
        resting_potential: Resting potentials (n_neurons,)
        tau_m: Membrane time constants (n_neurons,)
        refractory_mask: Refractory mask (n_neurons,)
        dt: Time step in ms
        
    Returns:
        Tuple of (new_voltages, spike_mask)
    """
    # Membrane dynamics: dv/dt = -(v - v_rest)/tau_m + I
    dv = dt * (-(v - resting_potential) / tau_m + i_input)
    v_new = v + dv
    
    # Apply refractory mask
    v_new = jnp.where(refractory_mask, resting_potential, v_new)
    
    # Check for spikes
    spike_mask = (v_new >= threshold) & (~refractory_mask)
    
    # Reset spiking neurons
    v_new = jnp.where(spike_mask, resting_potential, v_new)
    
    return v_new, spike_mask


@jit
def update_stdp_traces(pre_trace: jnp.ndarray,
                      post_trace: jnp.ndarray,
                      spike_mask: jnp.ndarray,
                      tau_plus: float = 20.0,
                      tau_minus: float = 20.0,
                      dt: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Update STDP traces for all neurons.
    
    Args:
        pre_trace: Pre-synaptic traces (n_neurons,)
        post_trace: Post-synaptic traces (n_neurons,)
        spike_mask: Neurons that spiked (n_neurons,)
        tau_plus: LTP time constant
        tau_minus: LTD time constant
        dt: Time step
        
    Returns:
        Tuple of (new_pre_trace, new_post_trace)
    """
    # Exponential decay
    pre_trace_new = pre_trace * jnp.exp(-dt / tau_plus)
    post_trace_new = post_trace * jnp.exp(-dt / tau_minus)
    
    # Add spike contributions
    pre_trace_new = jnp.where(spike_mask, pre_trace_new + 1.0, pre_trace_new)
    post_trace_new = jnp.where(spike_mask, post_trace_new + 1.0, post_trace_new)
    
    return pre_trace_new, post_trace_new


def sparse_matmul(pre_indices: jnp.ndarray,
                 post_indices: jnp.ndarray,
                 weights: jnp.ndarray,
                 spikes: jnp.ndarray,
                 n_neurons: int) -> jnp.ndarray:
    """
    Efficient sparse matrix multiplication for spike propagation.
    
    Args:
        pre_indices: Pre-synaptic neuron indices
        post_indices: Post-synaptic neuron indices
        weights: Synaptic weights
        spikes: Spike vector (n_neurons,)
        n_neurons: Total number of neurons
        
    Returns:
        Input currents to post-synaptic neurons
    """
    # Get spike values for pre-synaptic neurons
    pre_spikes = spikes[pre_indices]
    
    # Calculate weighted inputs
    weighted_inputs = pre_spikes * weights
    
    # Sum inputs for each post-synaptic neuron
    input_currents = jnp.zeros(n_neurons)
    input_currents = input_currents.at[post_indices].add(weighted_inputs)
    
    return input_currents


def apply_stdp_update(pre_indices: jnp.ndarray,
                     post_indices: jnp.ndarray,
                     weights: jnp.ndarray,
                     pre_traces: jnp.ndarray,
                     post_traces: jnp.ndarray,
                     spike_mask: jnp.ndarray,
                     a_plus: float = 0.05,
                     a_minus: float = 0.02,
                     modulation: float = 1.0) -> jnp.ndarray:
    """
    Apply STDP learning rule to synaptic weights.
    
    Args:
        pre_indices: Pre-synaptic neuron indices
        post_indices: Post-synaptic neuron indices
        weights: Current synaptic weights
        pre_traces: Pre-synaptic traces
        post_traces: Post-synaptic traces
        spike_mask: Neurons that spiked
        a_plus: LTP learning rate
        a_minus: LTD learning rate
        modulation: Neuromodulatory signal
        
    Returns:
        Updated synaptic weights
    """
    # Get traces for pre and post neurons
    pre_trace_vals = pre_traces[pre_indices]
    post_trace_vals = post_traces[post_indices]
    
    # Get spike status for post neurons
    post_spikes = spike_mask[post_indices]
    
    # Calculate weight changes
    # LTP when post-neuron spikes and pre-trace is high
    ltp = a_plus * pre_trace_vals * post_spikes
    
    # LTD when pre-trace is high but post-neuron doesn't spike
    ltd = a_minus * pre_trace_vals * post_trace_vals
    
    # Net weight change
    dw = (ltp - ltd) * modulation
    
    # Update weights with bounds
    new_weights = jnp.clip(weights + dw, 0.0, 1.0)
    
    return new_weights


@jit
def compute_novelty_signal(activity: jnp.ndarray,
                          baseline_activity: jnp.ndarray,
                          alpha: float = 0.1) -> float:
    """
    Compute novelty signal based on activity deviation from baseline.
    
    Args:
        activity: Current activity pattern
        baseline_activity: Rolling average of activity
        alpha: Novelty sensitivity
        
    Returns:
        Novelty signal (0-1)
    """
    # Calculate deviation from baseline
    deviation = jnp.mean(jnp.abs(activity - baseline_activity))
    
    # Convert to novelty signal with sigmoid
    novelty = 1.0 / (1.0 + jnp.exp(-alpha * deviation))
    
    return novelty


@jit
def update_baseline_activity(baseline: jnp.ndarray,
                           current: jnp.ndarray,
                           tau: float = 1000.0,
                           dt: float = 1.0) -> jnp.ndarray:
    """
    Update baseline activity with exponential moving average.
    
    Args:
        baseline: Current baseline activity
        current: Current activity
        tau: Time constant for baseline update
        dt: Time step
        
    Returns:
        Updated baseline activity
    """
    decay = jnp.exp(-dt / tau)
    return baseline * decay + current * (1 - decay)


class JAXHebSNN:
    """
    JAX-accelerated Hebbian SNN implementation.
    
    This class provides GPU/Metal-accelerated neural network simulation
    with vectorized operations for improved performance.
    """
    
    def __init__(self,
                 n_sensory: int = 1000,
                 n_associative: int = 5000,
                 n_inhibitory: int = 1000,
                 n_output: int = 1000,
                 connectivity_density: float = 0.1,
                 seed: int = 42):
        """
        Initialize JAX-accelerated Hebbian SNN.
        
        Args:
            n_sensory: Number of sensory neurons
            n_associative: Number of associative neurons
            n_inhibitory: Number of inhibitory neurons
            n_output: Number of output neurons
            connectivity_density: Connection density
            seed: Random seed
        """
        self.n_sensory = n_sensory
        self.n_associative = n_associative
        self.n_inhibitory = n_inhibitory
        self.n_output = n_output
        self.n_neurons = n_sensory + n_associative + n_inhibitory + n_output
        
        # Initialize random key
        self.key = jax.random.PRNGKey(seed)
        
        # Initialize neural parameters
        self._init_neural_params()
        
        # Initialize connectivity
        self._init_connectivity(connectivity_density)
        
        # Initialize state
        self.reset()
    
    def _init_neural_params(self):
        """Initialize neural parameters for all neuron types."""
        # Membrane potentials
        self.v_rest = jnp.zeros(self.n_neurons)
        self.v_rest = self.v_rest.at[self.n_sensory:self.n_sensory+self.n_associative].set(-0.1)
        self.v_rest = self.v_rest.at[self.n_sensory+self.n_associative:self.n_sensory+self.n_associative+self.n_inhibitory].set(-0.2)
        self.v_rest = self.v_rest.at[self.n_sensory+self.n_associative+self.n_inhibitory:].set(-0.1)
        
        # Firing thresholds
        self.threshold = jnp.ones(self.n_neurons) * 0.5
        self.threshold = self.threshold.at[self.n_sensory:self.n_sensory+self.n_associative].set(0.6)
        self.threshold = self.threshold.at[self.n_sensory+self.n_associative:self.n_sensory+self.n_associative+self.n_inhibitory].set(0.4)
        self.threshold = self.threshold.at[self.n_sensory+self.n_associative+self.n_inhibitory:].set(0.8)
        
        # Membrane time constants
        self.tau_m = jnp.ones(self.n_neurons) * 20.0
        
        # Refractory periods
        self.refractory_period = jnp.ones(self.n_neurons) * 2.0
        self.refractory_period = self.refractory_period.at[self.n_sensory:self.n_sensory+self.n_associative].set(4.0)
        self.refractory_period = self.refractory_period.at[self.n_sensory+self.n_associative:self.n_sensory+self.n_associative+self.n_inhibitory].set(3.0)
        self.refractory_period = self.refractory_period.at[self.n_sensory+self.n_associative+self.n_inhibitory:].set(5.0)
    
    def _init_connectivity(self, density: float):
        """Initialize sparse connectivity matrix."""
        # Calculate number of connections
        n_connections = int(self.n_neurons * self.n_neurons * density)
        
        # Generate random connections
        self.key, subkey = jax.random.split(self.key)
        pre_indices = jax.random.randint(subkey, (n_connections,), 0, self.n_neurons)
        
        self.key, subkey = jax.random.split(self.key)
        post_indices = jax.random.randint(subkey, (n_connections,), 0, self.n_neurons)
        
        # Filter out self-connections
        mask = pre_indices != post_indices
        self.pre_indices = pre_indices[mask]
        self.post_indices = post_indices[mask]
        
        # Initialize weights
        self.key, subkey = jax.random.split(self.key)
        n_valid = len(self.pre_indices)
        weights = jax.random.lognormal(subkey, shape=(n_valid,), sigma=0.5) * 0.1
        self.weights = jnp.clip(weights, 0.01, 1.0)
        
        # Initialize delays
        self.key, subkey = jax.random.split(self.key)
        self.delays = jax.random.uniform(subkey, shape=(n_valid,), minval=1.0, maxval=5.0)
    
    def reset(self):
        """Reset network state."""
        self.v = jnp.copy(self.v_rest)
        self.pre_traces = jnp.zeros(self.n_neurons)
        self.post_traces = jnp.zeros(self.n_neurons)
        self.refractory_until = jnp.zeros(self.n_neurons)
        self.baseline_activity = jnp.zeros(self.n_neurons)
        self.current_time = 0.0
        self.spike_history = []
    
    def step(self, input_spikes: jnp.ndarray, dt: float = 1.0) -> Tuple[jnp.ndarray, float]:
        """
        Execute one simulation step.
        
        Args:
            input_spikes: External input spikes (n_neurons,)
            dt: Time step in ms
            
        Returns:
            Tuple of (spike_mask, novelty_signal)
        """
        # Update refractory mask
        refractory_mask = self.refractory_until > self.current_time
        
        # Compute input currents from spikes
        spike_inputs = sparse_matmul(
            self.pre_indices, self.post_indices, self.weights,
            input_spikes, self.n_neurons
        )
        
        # Apply inhibitory scaling
        inhibitory_start = self.n_sensory + self.n_associative
        inhibitory_end = inhibitory_start + self.n_inhibitory
        
        # Scale inhibitory inputs
        inhibitory_mask = (self.pre_indices >= inhibitory_start) & (self.pre_indices < inhibitory_end)
        spike_inputs = spike_inputs.at[self.post_indices].add(
            jnp.where(inhibitory_mask, -1.2 * self.weights, 0.0)
        )
        
        # Update membrane potentials
        self.v, spike_mask = leaky_integrate_fire(
            self.v, spike_inputs, self.threshold, self.v_rest,
            self.tau_m, refractory_mask, dt
        )
        
        # Update refractory periods
        self.refractory_until = jnp.where(
            spike_mask, self.current_time + self.refractory_period, self.refractory_until
        )
        
        # Update STDP traces
        self.pre_traces, self.post_traces = update_stdp_traces(
            self.pre_traces, self.post_traces, spike_mask, dt=dt
        )
        
        # Compute novelty signal
        activity = spike_mask.astype(jnp.float32)
        novelty = compute_novelty_signal(activity, self.baseline_activity)
        
        # Update baseline activity
        self.baseline_activity = update_baseline_activity(
            self.baseline_activity, activity, dt=dt
        )
        
        # Apply STDP learning
        self.weights = apply_stdp_update(
            self.pre_indices, self.post_indices, self.weights,
            self.pre_traces, self.post_traces, spike_mask,
            modulation=novelty
        )
        
        # Update time
        self.current_time += dt
        
        # Store spike history
        self.spike_history.append(spike_mask)
        
        return spike_mask, novelty
    
    def run(self, duration: float, input_fn=None, dt: float = 1.0) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Duration in ms
            input_fn: Optional input function
            dt: Time step in ms
            
        Returns:
            Dictionary with simulation results
        """
        n_steps = int(duration / dt)
        spike_history = []
        novelty_history = []
        
        for step in range(n_steps):
            # Get input
            if input_fn is not None:
                input_spikes = input_fn(self.current_time)
            else:
                input_spikes = jnp.zeros(self.n_neurons)
            
            # Execute step
            spikes, novelty = self.step(input_spikes, dt)
            
            # Store results
            spike_history.append(spikes)
            novelty_history.append(novelty)
        
        return {
            'spikes': jnp.array(spike_history),
            'novelty': jnp.array(novelty_history),
            'final_weights': self.weights
        }
    
    def get_output_activity(self, window_size: int = 100) -> Dict[int, float]:
        """
        Get recent output neuron activity.
        
        Args:
            window_size: Number of recent time steps to consider
            
        Returns:
            Dictionary mapping output token IDs to firing rates
        """
        if len(self.spike_history) == 0:
            return {}
        
        # Get recent activity
        recent_spikes = jnp.array(self.spike_history[-window_size:])
        
        # Extract output neuron activity
        output_start = self.n_sensory + self.n_associative + self.n_inhibitory
        output_activity = recent_spikes[:, output_start:]
        
        # Calculate firing rates
        firing_rates = jnp.mean(output_activity, axis=0)
        
        # Convert to dictionary
        output_dict = {}
        for i, rate in enumerate(firing_rates):
            output_dict[i] = float(rate)
        
        return output_dict