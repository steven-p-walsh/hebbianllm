"""
Plastic Hebbian SNN - Fixed Size Network with Dynamic Connectivity

This implements true neural plasticity:
- Fixed number of neurons (like biological brains)
- Dynamic synaptic weights that strengthen/weaken with use
- Synaptic pruning and formation
- Homeostatic plasticity to maintain stability
- Fast learning through connection rewiring, not network growth

Key insight: Babies learn fast through plasticity, not by growing more neurons!
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import partial

from hebbianllm.core.network import HebSNN


class SynapticPlasticity:
    """Manages synaptic plasticity mechanisms."""
    
    def __init__(self, n_neurons: int, dtype=jnp.float32):
        self.n_neurons = n_neurons
        self.dtype = dtype
        
        # Plasticity parameters - optimized for learning
        self.ltp_rate = 0.005     # Long-term potentiation rate (increased for faster learning)
        self.ltd_rate = 0.002     # Long-term depression rate (balanced)  
        self.decay_rate = 0.003   # Synaptic decay (reduced to preserve learning)
        self.pruning_threshold = 0.01  # Below this, connections are pruned
        self.formation_threshold = 0.8 # Above this, new connections form
        
        # Homeostatic parameters - tuned for sparse coding
        self.target_activity = 0.05   # Target 5% activity (sparse coding)
        self.homeostatic_rate = 0.005  # Moderate homeostatic adjustment
        
        # Metaplasticity (learning to learn)
        self.learning_rate_adaptation = 0.99  # Adapt learning rates
    
    @partial(jit, static_argnums=(0,))
    def update_weights(self, weights: jnp.ndarray, 
                      pre_activity: jnp.ndarray, 
                      post_activity: jnp.ndarray,
                      eligibility_trace: jnp.ndarray) -> jnp.ndarray:
        """
        Update synaptic weights based on pre/post activity.
        
        Implements:
        - Hebbian LTP/LTD
        - Synaptic decay
        - Homeostatic scaling
        """
        # Hebbian plasticity: strengthen when pre and post fire together
        hebbian_update = jnp.outer(post_activity, pre_activity) * self.ltp_rate
        
        # Anti-Hebbian: weaken when only one fires
        anti_hebbian = (jnp.outer(post_activity, 1.0 - pre_activity) + 
                       jnp.outer(1.0 - post_activity, pre_activity)) * self.ltd_rate
        
        # Apply plasticity with eligibility trace
        plasticity_update = (hebbian_update - anti_hebbian) * eligibility_trace
        
        # Synaptic decay (prevents runaway growth)
        decay_update = -weights * self.decay_rate
        
        # Update weights
        new_weights = weights + plasticity_update + decay_update
        
        # Apply homeostatic scaling
        new_weights = self._apply_homeostatic_scaling(new_weights, post_activity)
        
        # Clip weights to reasonable range
        new_weights = jnp.clip(new_weights, -1.0, 1.0)
        
        return new_weights
    
    @partial(jit, static_argnums=(0,))
    def _apply_homeostatic_scaling(self, weights: jnp.ndarray, 
                                  activity: jnp.ndarray) -> jnp.ndarray:
        """Apply homeostatic scaling to maintain target activity levels."""
        current_activity = jnp.mean(activity)
        activity_error = self.target_activity - current_activity
        
        # Scale all incoming weights to adjust overall excitability
        scaling_factor = 1.0 + activity_error * self.homeostatic_rate
        scaled_weights = weights * scaling_factor
        
        return scaled_weights
    
    @partial(jit, static_argnums=(0,))
    def prune_and_form_synapses(self, weights: jnp.ndarray, 
                               activity_correlation: jnp.ndarray) -> jnp.ndarray:
        """
        Prune weak synapses and form new ones based on activity correlations.
        
        This simulates structural plasticity where new synapses form between
        frequently co-active neurons.
        """
        # Prune very weak connections
        pruned_weights = jnp.where(jnp.abs(weights) < self.pruning_threshold, 
                                  0.0, weights)
        
        # Form new connections where there's high correlation but no connection
        no_connection_mask = (jnp.abs(pruned_weights) < 1e-6)
        high_correlation_mask = (activity_correlation > self.formation_threshold)
        
        # Create new weak connections
        new_connections = (no_connection_mask & high_correlation_mask) * 0.1
        
        final_weights = pruned_weights + new_connections
        
        return final_weights


class TokenMapper:
    """Maps tokens to fixed neural populations dynamically."""
    
    def __init__(self, n_neurons: int, vocab_size: int = 1000):
        self.n_neurons = n_neurons
        self.vocab_size = vocab_size
        
        # Each token gets a population of neurons for distributed representation
        # Enforce minimum of 10 neurons per token for biological realism
        self.neurons_per_token = max(10, n_neurons // vocab_size)
        
        # Adjust effective vocab size to fit neuron constraint
        self.effective_vocab_size = min(vocab_size, n_neurons // self.neurons_per_token)
        
        # Dynamic mapping that can change as we learn
        self.token_mappings = {}  # token_id -> neuron_indices
        self.neuron_usage = jnp.zeros(n_neurons)  # Track which neurons are used
        
        # Initialize with some basic mappings
        self._init_basic_mappings()
    
    def _init_basic_mappings(self):
        """Initialize mappings for common tokens."""
        common_tokens = [0, 1, 2, 3, 4]  # PAD, UNK, BOS, EOS, SPACE
        
        for i, token_id in enumerate(common_tokens):
            start_idx = i * self.neurons_per_token
            end_idx = start_idx + self.neurons_per_token
            self.token_mappings[token_id] = list(range(start_idx, min(end_idx, self.n_neurons)))
    
    def get_token_neurons(self, token_id: int) -> List[int]:
        """Get neuron indices for a token, creating mapping if needed."""
        if token_id not in self.token_mappings:
            self._create_token_mapping(token_id)
        
        return self.token_mappings[token_id]
    
    def _create_token_mapping(self, token_id: int):
        """Create a new mapping for an unseen token."""
        # Find least used neurons
        available_neurons = jnp.argsort(self.neuron_usage)[:self.neurons_per_token]
        
        self.token_mappings[token_id] = available_neurons.tolist()
        
        # Update usage
        self.neuron_usage = self.neuron_usage.at[jnp.array(available_neurons)].add(1.0)
    
    def encode_tokens_to_activity(self, token_ids: List[int]) -> jnp.ndarray:
        """Convert token sequence to neural activity pattern."""
        activity = jnp.zeros(self.n_neurons)
        
        for i, token_id in enumerate(token_ids):
            neurons = self.get_token_neurons(token_id)
            
            # Temporal weighting: recent tokens have higher activity
            temporal_weight = jnp.exp(-0.1 * (len(token_ids) - 1 - i))
            
            # Set activity for this token's neurons
            activity = activity.at[jnp.array(neurons)].add(temporal_weight)
        
        return jnp.clip(activity, 0.0, 1.0)
    
    def decode_activity_to_tokens(self, activity: jnp.ndarray, top_k: int = 5) -> List[int]:
        """Convert neural activity back to likely tokens."""
        token_activities = {}
        
        for token_id, neurons in self.token_mappings.items():
            token_activity = jnp.sum(activity[jnp.array(neurons)])
            token_activities[token_id] = float(token_activity)
        
        # Get top-k most active tokens
        sorted_tokens = sorted(token_activities.items(), key=lambda x: x[1], reverse=True)
        return [token_id for token_id, activity in sorted_tokens[:top_k]]


class PlasticHebSNN(HebSNN):
    """
    Fixed-size Hebbian SNN with neural plasticity.
    
    Key features:
    - Fixed number of neurons (no growth)
    - Dynamic synaptic weights
    - Synaptic pruning and formation
    - Homeostatic plasticity
    - Fast learning through rewiring
    """
    
    def __init__(self, 
                 n_neurons: int = 2000,
                 vocab_size: int = 500,
                 initial_connectivity: float = 0.1,
                 **kwargs):
        
        # Calculate layer sizes for fixed total
        n_sensory = n_neurons // 4
        n_associative = n_neurons // 2  
        n_inhibitory = n_neurons // 8
        n_output = n_neurons - n_sensory - n_associative - n_inhibitory
        
        # Initialize base network
        super().__init__(
            n_sensory=n_sensory,
            n_associative=n_associative, 
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=initial_connectivity,
            **kwargs
        )
        
        # Store initialization parameters
        self.initial_connectivity = initial_connectivity
        
        # Plasticity components
        self.plasticity = SynapticPlasticity(self.n_neurons, self.dtype)
        self.token_mapper = TokenMapper(self.n_neurons, vocab_size)
        
        # Plasticity state
        self.synaptic_weights = self._init_plastic_weights()
        self.eligibility_trace = jnp.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)
        self.activity_history = []
        self.learning_step = 0
        
        # Current activity for tracking
        self.current_activity = jnp.zeros(self.n_neurons, dtype=self.dtype)
        
        print(f"PlasticHebSNN initialized:")
        print(f"  Fixed neurons: {self.n_neurons:,}")
        print(f"  Vocab capacity: {vocab_size:,}")
        print(f"  Initial connectivity: {initial_connectivity:.1%}")
        print(f"  Neurons per token: {self.token_mapper.neurons_per_token}")
        print(f"  Potential synapses: {self.n_neurons**2:,}")
        print(f"  Using device: {self.devices[0] if self.devices else 'CPU'}")
    
    def _init_plastic_weights(self) -> jnp.ndarray:
        """Initialize synaptic weights for plasticity."""
        # Start with sparse random weights based on initial_connectivity
        key = jax.random.PRNGKey(42)
        
        # Create sparse connectivity matrix
        n_connections = int(self.n_neurons * self.n_neurons * self.initial_connectivity)
        
        # Start with zeros
        weights = jnp.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)
        
        # Add random connections
        key, subkey = jax.random.split(key)
        pre_indices = jax.random.randint(subkey, (n_connections,), 0, self.n_neurons)
        key, subkey = jax.random.split(key)
        post_indices = jax.random.randint(subkey, (n_connections,), 0, self.n_neurons)
        
        # Random weights for these connections
        key, subkey = jax.random.split(key)
        connection_weights = jax.random.normal(subkey, (n_connections,), dtype=self.dtype) * 0.1
        
        # Set the connections
        weights = weights.at[post_indices, pre_indices].set(connection_weights)
        
        # Zero self-connections
        weights = weights.at[jnp.diag_indices(self.n_neurons)].set(0.0)
        
        return weights
    
    def process_tokens(self, token_ids: List[int], learning: bool = True) -> Dict:
        """
        Process tokens through the plastic network.
        
        Args:
            token_ids: List of token IDs to process
            learning: Whether to apply plasticity updates
            
        Returns:
            Dictionary with processing results
        """
        # Convert tokens to activity
        input_activity = self.token_mapper.encode_tokens_to_activity(token_ids)
        
        # Run network dynamics
        activity_sequence = self._run_plastic_dynamics(input_activity, n_steps=10)
        
        # Get final activity
        final_activity = activity_sequence[-1]
        self.current_activity = final_activity
        
        # Apply plasticity if learning
        if learning:
            self._apply_plasticity_updates(activity_sequence)
        
        # Update activity history
        self.activity_history.append(final_activity)
        if len(self.activity_history) > 20:  # Keep recent history
            self.activity_history.pop(0)
        
        return {
            'activity': final_activity,
            'activity_sequence': activity_sequence,
            'synaptic_weights': self.synaptic_weights,
            'learning_step': self.learning_step
        }
    
    def _run_plastic_dynamics(self, input_activity: jnp.ndarray, n_steps: int) -> List[jnp.ndarray]:
        """Run network dynamics with current synaptic weights."""
        activities = [input_activity]
        current_activity = input_activity
        
        for step in range(n_steps):
            # Apply synaptic transmission
            synaptic_input = jnp.dot(self.synaptic_weights, current_activity)
            
            # Add external input on first step
            if step == 0:
                total_input = synaptic_input + input_activity
            else:
                total_input = synaptic_input
            
            # Apply activation function (sigmoid for bounded activity)
            sigmoid_activity = jax.nn.sigmoid(total_input)
            
            # Add tonic activity for early learning (like baby brain baseline firing)
            tonic_activity = self._add_tonic_activity(sigmoid_activity, step)
            
            # Implement sparse coding: only top 5% of neurons fire (like real brains)
            k = max(1, int(0.05 * len(tonic_activity)))  # Top 5% of neurons
            
            # Find the k most active neurons using top_k
            _, top_indices = jax.lax.top_k(tonic_activity, k)
            
            # Create sparse activity pattern
            new_activity = jnp.zeros_like(tonic_activity)
            new_activity = new_activity.at[top_indices].set(tonic_activity[top_indices])
            
            # Normalize the active neurons to maintain signal strength
            if jnp.sum(new_activity) > 0:
                new_activity = new_activity / jnp.max(new_activity)  # Normalize to [0,1]
            
            # Add exploration noise to active neurons
            key = jax.random.PRNGKey(self.learning_step + step)
            noise = jax.random.normal(key, new_activity.shape, dtype=self.dtype) * 0.01
            new_activity = jnp.where(new_activity > 0, 
                                   jnp.clip(new_activity + noise, 0.0, 1.0),
                                   0.0)
            
            activities.append(new_activity)
            current_activity = new_activity
        
        return activities
    
    def _apply_plasticity_updates(self, activity_sequence: List[jnp.ndarray]):
        """Apply plasticity updates based on activity sequence."""
        self.learning_step += 1
        
        # Update eligibility trace (decaying memory of recent activity)
        if len(activity_sequence) >= 2:
            pre_activity = activity_sequence[-2]
            post_activity = activity_sequence[-1]
            
            # Store old weights for change tracking
            old_weights = self.synaptic_weights.copy()
            
            # Update eligibility trace with decay
            self.eligibility_trace = self.eligibility_trace * 0.9 + jnp.outer(post_activity, pre_activity) * 0.1
            
            # Apply synaptic plasticity
            self.synaptic_weights = self.plasticity.update_weights(
                self.synaptic_weights,
                pre_activity,
                post_activity, 
                self.eligibility_trace
            )
            
            # Log weight changes to verify plasticity is working
            weight_change = jnp.sum(jnp.abs(self.synaptic_weights - old_weights))
            if self.learning_step % 10 == 0:  # Log every 10 steps to avoid spam
                print(f"Step {self.learning_step}: Weight change sum = {float(weight_change):.6f}")
            
            # Apply connectivity cap to prevent saturation
            self._apply_connectivity_cap()
        
        # Structural plasticity every few steps
        if self.learning_step % 10 == 0:
            self._apply_structural_plasticity()
    
    def _apply_connectivity_cap(self, max_connectivity: float = 0.08):
        """Apply connectivity cap with sleep-like consolidation."""
        # Calculate current connectivity
        non_zero_weights = jnp.abs(self.synaptic_weights) > 0.001
        current_connectivity = float(jnp.mean(non_zero_weights))
        
        print(f"Current connectivity: {current_connectivity:.3f}")
        
        # If too connected, trigger "sleep" consolidation
        if current_connectivity > max_connectivity:
            print(f"🛌 Network entering sleep mode (connectivity: {current_connectivity:.1%})")
            self.synaptic_weights = self._sleep_consolidation()
            
            # Verify reduction
            new_connectivity = float(jnp.mean(jnp.abs(self.synaptic_weights) > 0.001))
            print(f"After sleep: {new_connectivity:.1%} connectivity")
    
    def _sleep_consolidation(self):
        """Sleep-like consolidation: strengthen important connections, prune weak ones."""
        # Sleep mechanism: only keep the most important connections
        weight_magnitude = jnp.abs(self.synaptic_weights)
        
        # Top 5% of connections survive (biological sparsity)
        survival_threshold = jnp.percentile(weight_magnitude, 95)
        survival_mask = weight_magnitude >= survival_threshold
        
        # Apply sleep consolidation
        consolidated_weights = jnp.where(
            survival_mask,
            self.synaptic_weights * 1.1,  # Strengthen survivors slightly
            jnp.zeros_like(self.synaptic_weights)  # Prune the rest
        )
        
        return consolidated_weights
    
    def _apply_structural_plasticity(self):
        """Apply structural plasticity (synaptic pruning/formation)."""
        if len(self.activity_history) >= 2:
            # Compute activity correlations
            recent_activities = jnp.array(self.activity_history[-10:])  # Last 10 steps
            correlations = jnp.corrcoef(recent_activities.T)
            correlations = jnp.nan_to_num(correlations)  # Handle NaN from constant activity
            
            # Apply structural plasticity
            self.synaptic_weights = self.plasticity.prune_and_form_synapses(
                self.synaptic_weights, 
                jnp.abs(correlations)
            )
    
    def generate_tokens(self, context_tokens: List[int], max_length: int = 10) -> List[int]:
        """Generate tokens based on current network state with sparse coding support."""
        generated = context_tokens.copy()
        
        # If network is too young/sparse, use babbling mode
        if self.learning_step < 5 or len(context_tokens) == 0:
            return self._generate_babbling(max_length=3)
        
        for _ in range(max_length):
            # Process current context
            result = self.process_tokens(generated[-6:], learning=False)  # Shorter context for sparse nets
            
            # Decode activity to get next token
            next_tokens = self.token_mapper.decode_activity_to_tokens(result['activity'], top_k=10)
            
            if next_tokens:
                # Sample from top tokens (with some randomness)
                weights = jnp.array([1.0 / (i + 1) for i in range(len(next_tokens))])
                weights = weights / jnp.sum(weights)
                
                key = jax.random.PRNGKey(self.learning_step + len(generated))
                choice_idx = jax.random.categorical(key, jnp.log(weights))
                next_token = next_tokens[choice_idx]
                
                generated.append(next_token)
                
                # For sparse networks, generate shorter sequences
                if len(generated) - len(context_tokens) >= 3:
                    break
            else:
                # Fall back to babbling if no tokens generated
                babble_tokens = self._generate_babbling(max_length=2)
                if babble_tokens:
                    generated.extend(babble_tokens)
                break
        
        return generated
    
    def _add_tonic_activity(self, activity: jnp.ndarray, step: int) -> jnp.ndarray:
        """
        Add tonic (baseline) activity for early learning, like baby brains.
        
        Real baby brains have spontaneous baseline firing that helps bootstrap learning.
        This activity decreases as the network matures and learns patterns.
        """
        # Tonic activity strength decreases with learning experience
        maturity_factor = jnp.exp(-self.learning_step / 50.0)  # Decays over ~50 learning steps
        tonic_strength = 0.3 * maturity_factor  # Start at 30% baseline, decay to ~0%
        
        # Generate consistent tonic pattern (not random each time)
        tonic_seed = self.learning_step // 10 + step  # Changes slowly
        key = jax.random.PRNGKey(tonic_seed)
        
        # Create tonic baseline activity
        tonic_pattern = jax.random.uniform(key, activity.shape, dtype=self.dtype) * tonic_strength
        
        # Add tonic activity to existing activity
        enhanced_activity = activity + tonic_pattern
        
        return enhanced_activity
    
    def _generate_babbling(self, max_length: int = 3) -> List[int]:
        """Generate baby-like babbling sounds for early learning."""
        # Get available tokens, preferring higher IDs (learned patterns)
        available_tokens = list(self.token_mapper.token_mappings.keys())
        
        if not available_tokens:
            # Fallback: use token range that avoids special tokens
            available_tokens = list(range(5, 25))
        
        # Bias towards recently learned tokens for more realistic babbling
        if len(available_tokens) > 10:
            # Use the most recent 60% of tokens (representing active vocabulary)
            cutoff = max(10, int(len(available_tokens) * 0.6))
            available_tokens = available_tokens[-cutoff:]
        
        # Generate babbling with slight preference for repetition (like babies)
        result = []
        if available_tokens:
            key = jax.random.PRNGKey(self.learning_step + 42)
            
            # First token
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (), 0, len(available_tokens))
            first_token = available_tokens[idx]
            result.append(first_token)
            
            # Additional tokens with repetition bias
            for i in range(1, max_length):
                key, subkey = jax.random.split(key)
                # 40% chance to repeat previous token (babbling behavior)
                if jax.random.uniform(subkey) < 0.4 and len(result) > 0:
                    result.append(result[-1])
                else:
                    idx = jax.random.randint(subkey, (), 0, len(available_tokens))
                    result.append(available_tokens[idx])
        
        return result
    
    def get_plasticity_stats(self) -> Dict:
        """Get statistics about current plasticity state."""
        weight_stats = {
            'mean_weight': float(jnp.mean(jnp.abs(self.synaptic_weights))),
            'max_weight': float(jnp.max(jnp.abs(self.synaptic_weights))),
            'n_connections': int(jnp.sum(jnp.abs(self.synaptic_weights) > 0.01)),
            'connectivity': float(jnp.mean(jnp.abs(self.synaptic_weights) > 0.01))
        }
        
        activity_stats = {
            'current_activity_mean': float(jnp.mean(self.current_activity)),
            'current_activity_max': float(jnp.max(self.current_activity)),
            'n_active_neurons': int(jnp.sum(self.current_activity > 0.1))
        }
        
        # Add tonic activity stats
        maturity_factor = float(jnp.exp(-self.learning_step / 50.0))
        tonic_strength = 0.3 * maturity_factor
        
        return {
            'learning_step': self.learning_step,
            'weights': weight_stats,
            'activity': activity_stats,
            'token_mappings': len(self.token_mapper.token_mappings),
            'tonic_strength': tonic_strength,
            'maturity_factor': maturity_factor,
            'is_mature': maturity_factor < 0.1
        }
    
    def reset_plasticity(self):
        """Reset plasticity state (like sleep/forgetting)."""
        # Don't reset weights completely, just reduce them
        self.synaptic_weights = self.synaptic_weights * 0.9
        self.eligibility_trace = jnp.zeros_like(self.eligibility_trace)
        self.activity_history = []
        print("Plasticity state reset (like sleep)")
    
    def get_activity(self) -> jnp.ndarray:
        """Get current network activity."""
        return self.current_activity
    
    def save_network_state(self, filepath: str):
        """Save the complete network state including synaptic weights."""
        import numpy as np
        import os
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            network_state = {
                'synaptic_weights': np.array(self.synaptic_weights),
                'eligibility_trace': np.array(self.eligibility_trace),
                'current_activity': np.array(self.current_activity),
                'learning_step': self.learning_step,
                'token_mappings': self.token_mapper.token_mappings,
                'neuron_usage': np.array(self.token_mapper.neuron_usage),
                'plasticity_stats': self.get_plasticity_stats(),
                'network_config': {
                    'n_neurons': self.n_neurons,
                    'initial_connectivity': self.initial_connectivity,
                    'neurons_per_token': self.token_mapper.neurons_per_token
                }
            }
            
            np.savez_compressed(filepath, **network_state)
            print(f"Network state saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to save network state to {filepath}: {e}")
            raise
    
    def load_network_state(self, filepath: str):
        """Load previously saved network state."""
        import numpy as np
        
        try:
            data = np.load(filepath, allow_pickle=True)
            
            self.synaptic_weights = jnp.array(data['synaptic_weights'])
            self.eligibility_trace = jnp.array(data['eligibility_trace'])
            self.current_activity = jnp.array(data['current_activity'])
            self.learning_step = int(data['learning_step'])
            
            # Restore token mappings
            self.token_mapper.token_mappings = data['token_mappings'].item()
            self.token_mapper.neuron_usage = jnp.array(data['neuron_usage'])
            
            print(f"Network state loaded from {filepath}")
            print(f"Restored learning step: {self.learning_step}")
            
        except Exception as e:
            print(f"Failed to load network state: {e}")
            print("Continuing with fresh network state")