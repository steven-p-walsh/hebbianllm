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


class Neuromodulator:
    """Manages neuromodulation signals (dopamine-like, acetylcholine-like)."""
    
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype
        
        # Neuromodulator levels
        self.dopamine_level = 0.0    # Reward signal
        self.acetylcholine_level = 0.0  # Attention/novelty signal
        
        # History for adaptation
        self.reward_history = []
        self.novelty_history = []
        
        # Decay rates
        self.dopamine_decay = 0.8
        self.acetylcholine_decay = 0.7
    
    def update_dopamine(self, reward: float):
        """Update dopamine based on reward signal."""
        self.dopamine_level = self.dopamine_level * self.dopamine_decay + reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 20:
            self.reward_history.pop(0)
    
    def update_acetylcholine(self, novelty: float):
        """Update acetylcholine based on novelty/attention signal."""
        self.acetylcholine_level = self.acetylcholine_level * self.acetylcholine_decay + novelty
        self.novelty_history.append(novelty)
        if len(self.novelty_history) > 20:
            self.novelty_history.pop(0)
    
    def get_ltp_modulation(self) -> float:
        """Get LTP modulation based on neuromodulator levels."""
        # Dopamine boosts LTP (reward-based learning)
        dopamine_boost = 1.0 + self.dopamine_level * 0.5
        # Acetylcholine boosts attention to novel patterns
        attention_boost = 1.0 + self.acetylcholine_level * 0.3
        return dopamine_boost * attention_boost
    
    def get_ltd_modulation(self) -> float:
        """Get LTD modulation based on neuromodulator levels."""
        # High dopamine reduces forgetting
        dopamine_protection = 1.0 - self.dopamine_level * 0.3
        return max(0.1, dopamine_protection)


class SynapticPlasticity:
    """Manages synaptic plasticity mechanisms."""
    
    def __init__(self, n_neurons: int, dtype=jnp.float32):
        self.n_neurons = n_neurons
        self.dtype = dtype
        
        # Plasticity parameters - optimized for faster bootstrapping
        self.base_ltp_rate = 0.01     # Increased base LTP for rapid learning
        self.base_ltd_rate = 0.004    # Balanced LTD rate
        self.decay_rate = 0.001       # Reduced decay to preserve learning longer
        self.pruning_threshold = 0.01  # Below this, connections are pruned
        self.formation_threshold = 0.6 # Lowered for more exploratory connections
        
        # Current rates (will be modulated by maturity and neuromodulators)
        self.ltp_rate = self.base_ltp_rate
        self.ltd_rate = self.base_ltd_rate
        
        # Homeostatic parameters - gentler for gradual growth
        self.target_activity = 0.05   # Target 5% activity (sparse coding)
        self.homeostatic_rate = 0.005  # Moderate homeostatic adjustment
        
        # Metaplasticity (learning to learn)
        self.learning_rate_adaptation = 0.99  # Adapt learning rates
        
        # Neuromodulation
        self.neuromodulator = Neuromodulator(dtype)
    
    def update_weights(self, weights: jnp.ndarray, 
                      pre_activity: jnp.ndarray, 
                      post_activity: jnp.ndarray,
                      eligibility_trace: jnp.ndarray,
                      error_signal: float = 0.0) -> jnp.ndarray:
        """
        Update synaptic weights based on pre/post activity with neuromodulation.
        
        Implements:
        - Hebbian LTP/LTD with neuromodulation
        - Error-driven plasticity
        - Synaptic decay
        - Homeostatic scaling
        """
        # Get neuromodulation factors
        ltp_modulation = self.neuromodulator.get_ltp_modulation()
        ltd_modulation = self.neuromodulator.get_ltd_modulation()
        
        # Error-driven plasticity: scale LTD based on error signal
        error_ltd_scaling = 1.0 + abs(error_signal) * 0.5  # Stronger LTD for errors
        error_ltp_scaling = 1.0 - abs(error_signal) * 0.3 if error_signal < 0 else 1.0
        
        # Modulated plasticity rates with error-driven scaling
        modulated_ltp_rate = self.ltp_rate * ltp_modulation * error_ltp_scaling
        modulated_ltd_rate = self.ltd_rate * ltd_modulation * error_ltd_scaling
        
        # Hebbian plasticity: strengthen when pre and post fire together
        hebbian_update = jnp.outer(post_activity, pre_activity) * modulated_ltp_rate
        
        # Anti-Hebbian: weaken when only one fires
        anti_hebbian = (jnp.outer(post_activity, 1.0 - pre_activity) + 
                       jnp.outer(1.0 - post_activity, pre_activity)) * modulated_ltd_rate
        
        # Apply plasticity with eligibility trace
        # For errors, use stronger eligibility trace for better learning
        error_eligibility_boost = 1.0 + abs(error_signal) * 0.3
        boosted_eligibility = eligibility_trace * error_eligibility_boost
        
        plasticity_update = (hebbian_update - anti_hebbian) * boosted_eligibility
        
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
        """Create a new mapping for an unseen token with semantic clustering."""
        # Enhanced mapping strategy: consider token relationships
        available_neurons = jnp.argsort(self.neuron_usage)[:self.neurons_per_token * 2]  # Get more candidates
        
        # If possible, try to map related tokens to nearby neurons
        best_neurons = available_neurons[:self.neurons_per_token]
        
        # For learned patterns, try to place them near similar tokens
        if len(self.token_mappings) > 5:  # If we have some existing mappings
            # Find recently used tokens (assume higher IDs are more recent)
            recent_tokens = [tid for tid in self.token_mappings.keys() if tid > max(5, len(self.token_mappings) - 20)]
            
            if recent_tokens:
                # Get neurons used by recent tokens
                recent_neuron_ranges = []
                for recent_tid in recent_tokens[-3:]:  # Last 3 recent tokens
                    recent_neurons = self.token_mappings[recent_tid]
                    if recent_neurons:
                        recent_neuron_ranges.extend(recent_neurons)
                
                if recent_neuron_ranges:
                    # Try to find neurons close to recent ones
                    avg_recent = jnp.mean(jnp.array(recent_neuron_ranges))
                    # Prefer neurons within a reasonable distance
                    distances = jnp.abs(available_neurons - avg_recent)
                    close_indices = jnp.argsort(distances)[:self.neurons_per_token]
                    best_neurons = available_neurons[close_indices]
        
        self.token_mappings[token_id] = best_neurons.tolist()
        
        # Update usage
        self.neuron_usage = self.neuron_usage.at[jnp.array(best_neurons)].add(1.0)
    
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
        """Convert neural activity back to likely tokens with contextual pause insertion."""
        token_activities = {}
        
        for token_id, neurons in self.token_mappings.items():
            token_activity = jnp.sum(activity[jnp.array(neurons)])
            token_activities[token_id] = float(token_activity)
        
        # Get top-k most active tokens
        sorted_tokens = sorted(token_activities.items(), key=lambda x: x[1], reverse=True)
        candidate_tokens = [token_id for token_id, activity in sorted_tokens[:top_k]]
        
        # Post-process to intelligently insert pause tokens
        if len(candidate_tokens) >= 2:
            # Check if we should insert a pause between content tokens
            top_token_activities = [token_activities[tid] for tid in candidate_tokens[:2]]
            
            # If two content tokens have high but similar activity, suggest pause insertion
            if (len(top_token_activities) >= 2 and 
                abs(top_token_activities[0] - top_token_activities[1]) < 0.2 and
                all(tid != 4 for tid in candidate_tokens[:2])):  # Not already pause tokens
                
                # Insert pause token with moderate priority
                if 4 not in candidate_tokens:
                    candidate_tokens.insert(1, 4)  # Insert at position 1 for moderate priority
                elif candidate_tokens.index(4) > 2:
                    # Move pause token to higher priority if it's too low
                    candidate_tokens.remove(4)
                    candidate_tokens.insert(1, 4)
        
        # Ensure pause token gets reasonable representation
        if 4 in self.token_mappings and 4 not in candidate_tokens:
            # Add pause token if it has any activity
            pause_activity = token_activities.get(4, 0.0)
            if pause_activity > 0.01:  # Threshold for minimal activity
                candidate_tokens.append(4)
        
        return candidate_tokens[:top_k]


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
        
        # Memory replay mechanism
        self.replay_buffer = []  # Store successful sequences for consolidation
        self.replay_buffer_size = 50
        self.replay_interval = 50  # Replay every 50 steps
        self.last_replay_step = 0
        
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
        
        # Update adaptive plasticity rates based on maturity
        self._update_adaptive_plasticity_rates()
        
        # Update eligibility trace (decaying memory of recent activity)
        if len(activity_sequence) >= 2:
            pre_activity = activity_sequence[-2]
            post_activity = activity_sequence[-1]
            
            # Store old weights for change tracking
            old_weights = self.synaptic_weights.copy()
            
            # Update eligibility trace with decay
            self.eligibility_trace = self.eligibility_trace * 0.9 + jnp.outer(post_activity, pre_activity) * 0.1
            
            # Apply synaptic plasticity with error signal
            error_signal = getattr(self, 'current_error_signal', 0.0)
            self.synaptic_weights = self.plasticity.update_weights(
                self.synaptic_weights,
                pre_activity,
                post_activity, 
                self.eligibility_trace,
                error_signal
            )
            
            # Log weight changes to verify plasticity is working (only in debug mode)
            weight_change = jnp.sum(jnp.abs(self.synaptic_weights - old_weights))
            if hasattr(self, '_debug_mode') and self._debug_mode and self.learning_step % 10 == 0:
                print(f"Step {self.learning_step}: Weight change sum = {float(weight_change):.6f}")
            
            # Apply gentler connectivity cap to prevent saturation
            self._apply_connectivity_cap()
        
        # Structural plasticity every few steps
        if self.learning_step % 10 == 0:
            self._apply_structural_plasticity()
            
        # Memory replay for consolidation
        if self.learning_step - self.last_replay_step >= self.replay_interval:
            self._perform_memory_replay()
    
    def _update_adaptive_plasticity_rates(self):
        """Update plasticity rates based on network maturity (metaplasticity)."""
        # Maturity factor: high plasticity early, gradually stabilizing
        maturity_factor = jnp.exp(-self.learning_step / 200.0)  # Slower decay for extended learning
        
        # Update rates with maturity modulation
        self.plasticity.ltp_rate = self.plasticity.base_ltp_rate * (0.5 + 0.5 * maturity_factor)
        self.plasticity.ltd_rate = self.plasticity.base_ltd_rate * (0.5 + 0.5 * maturity_factor)
    
    def _apply_connectivity_cap(self, max_connectivity: float = 0.15):
        """Apply gentler connectivity cap with sleep-like consolidation."""
        # Calculate current connectivity
        non_zero_weights = jnp.abs(self.synaptic_weights) > 0.001
        current_connectivity = float(jnp.mean(non_zero_weights))
        
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"Current connectivity: {current_connectivity:.3f}")
        
        # Only trigger sleep consolidation at higher connectivity threshold
        if current_connectivity > max_connectivity:
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🛌 Network entering sleep mode (connectivity: {current_connectivity:.1%})")
            self.synaptic_weights = self._sleep_consolidation()
            
            # Verify reduction
            new_connectivity = float(jnp.mean(jnp.abs(self.synaptic_weights) > 0.001))
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"After sleep: {new_connectivity:.1%} connectivity")
    
    def _sleep_consolidation(self):
        """Gentler sleep-like consolidation: strengthen important connections, prune weak ones."""
        # Sleep mechanism: keep more connections for gradual learning
        weight_magnitude = jnp.abs(self.synaptic_weights)
        
        # Top 15% of connections survive (gentler pruning for gradual growth)
        survival_threshold = jnp.percentile(weight_magnitude, 85)
        survival_mask = weight_magnitude >= survival_threshold
        
        # Apply gentler sleep consolidation
        consolidated_weights = jnp.where(
            survival_mask,
            self.synaptic_weights * 1.05,  # Slight strengthening
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
    
    def generate_tokens(self, context_tokens: List[int], max_length: int = 10, learning: bool = False) -> List[int]:
        """Generate tokens based on current network state with sparse coding support."""
        generated = context_tokens.copy()
        
        # If network is too young/sparse, use babbling mode
        if self.learning_step < 5 or len(context_tokens) == 0:
            return self._generate_babbling(max_length=3)
        
        for _ in range(max_length):
            # Process current context WITH exploration plasticity if enabled
            result = self.process_tokens(generated[-6:], learning=learning)  # Enable plasticity during generation
            
            # Decode activity to get next token
            next_tokens = self.token_mapper.decode_activity_to_tokens(result['activity'], top_k=10)
            
            if next_tokens:
                # Create more balanced sampling weights that give PAUSE tokens a fair chance
                # Use a gentler decay that doesn't heavily penalize later positions
                weights = jnp.array([0.9 ** i for i in range(len(next_tokens))])
                
                # Enhanced pause token boosting strategy
                if 4 in next_tokens:
                    pause_idx = next_tokens.index(4)
                    
                    # Check if we haven't generated a pause recently
                    recent_pause_count = sum(1 for t in generated[-3:] if t == 4)
                    
                    # Boost pause tokens more aggressively if no recent pauses
                    if recent_pause_count == 0:
                        # Strong boost if no pauses in recent context
                        weights = weights.at[pause_idx].set(weights[pause_idx] * 5.0)
                    else:
                        # Moderate boost if we have recent pauses
                        weights = weights.at[pause_idx].set(weights[pause_idx] * 2.0)
                
                # Temperature scaling for pause insertion
                # If no pause in last 3 tokens, use lower temperature (more decisive)
                if len(generated) >= 3 and all(t != 4 for t in generated[-3:]):
                    # Lower temperature makes pause tokens more likely to be selected
                    if 4 in next_tokens:
                        pause_idx = next_tokens.index(4)
                        weights = weights.at[pause_idx].set(weights[pause_idx] * 2.0)
                
                # Add a minimum weight floor to prevent any token from being too unlikely
                min_weight = jnp.max(weights) * 0.1  # At least 10% of the max weight
                weights = jnp.maximum(weights, min_weight)
                
                # Normalize weights
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
    
    def add_to_replay_buffer(self, token_sequence: List[int], reward: float = 0.0):
        """Add a token sequence to the replay buffer for consolidation."""
        if len(token_sequence) > 0:
            replay_entry = {
                'tokens': token_sequence,
                'reward': reward,
                'timestamp': self.learning_step,
                'surprise': abs(reward)  # Use absolute reward as surprise signal
            }
            
            self.replay_buffer.append(replay_entry)
            
            # Keep buffer size manageable
            if len(self.replay_buffer) > self.replay_buffer_size:
                # Remove oldest entries, but preferentially keep high-reward ones
                sorted_buffer = sorted(self.replay_buffer, key=lambda x: x['reward'], reverse=True)
                self.replay_buffer = sorted_buffer[:self.replay_buffer_size]
    
    def _perform_memory_replay(self):
        """Perform memory replay for offline consolidation."""
        if not self.replay_buffer:
            return
            
        self.last_replay_step = self.learning_step
        
        # Sample sequences for replay based on surprise/reward
        replay_candidates = sorted(self.replay_buffer, key=lambda x: x['surprise'], reverse=True)
        num_replays = min(5, len(replay_candidates))  # Replay top 5 sequences
        
        # Store original plasticity rates
        original_ltp = self.plasticity.ltp_rate
        original_ltd = self.plasticity.ltd_rate
        
        # Reduce plasticity during replay (consolidation, not new learning)
        self.plasticity.ltp_rate *= 0.8
        self.plasticity.ltd_rate *= 0.8
        
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"🌙 Memory replay: consolidating {num_replays} sequences")
        
        for i in range(num_replays):
            entry = replay_candidates[i]
            
            # Replay the sequence
            replay_result = self.process_tokens(entry['tokens'], learning=True)
            
            # Additional consolidation for high-reward sequences
            if entry['reward'] > 0.5:
                # Strengthen pathways for highly rewarded sequences
                self.plasticity.ltp_rate *= 1.2
                self.process_tokens(entry['tokens'], learning=True)
                self.plasticity.ltp_rate /= 1.2
        
        # Restore original plasticity rates
        self.plasticity.ltp_rate = original_ltp
        self.plasticity.ltd_rate = original_ltd
    
    def _add_tonic_activity(self, activity: jnp.ndarray, step: int) -> jnp.ndarray:
        """
        Add tonic (baseline) activity for early learning, like baby brains.
        
        Real baby brains have spontaneous baseline firing that helps bootstrap learning.
        This activity decreases as the network matures and learns patterns.
        Now adaptive to vocabulary growth for better exploration.
        """
        # Get current vocabulary size for adaptive scaling
        current_vocab_size = len(self.token_mapper.token_mappings)
        
        # Adaptive tonic strength based on vocabulary growth
        # Higher noise for low vocab (exploration), lower for high vocab (refinement)
        vocab_factor = jnp.exp(-current_vocab_size / 50.0)  # Decays as vocab grows
        maturity_factor = jnp.exp(-self.learning_step / 100.0)  # Slower decay for extended learning
        
        # Combine factors for adaptive exploration
        base_tonic_strength = 0.1 * vocab_factor + 0.03 * maturity_factor
        
        # Additional boost for very early learning (first 10 steps)
        if self.learning_step < 10:
            base_tonic_strength *= 2.0
        
        # Scale down if we have sufficient vocabulary
        if current_vocab_size > 100:
            base_tonic_strength *= 0.5
            
        # Generate consistent tonic pattern (not random each time)
        tonic_seed = self.learning_step // 10 + step  # Changes slowly
        key = jax.random.PRNGKey(tonic_seed)
        
        # Create gentle tonic baseline activity
        tonic_pattern = jax.random.uniform(key, activity.shape, dtype=self.dtype) * base_tonic_strength
        
        # Add tonic activity to existing activity
        enhanced_activity = activity + tonic_pattern
        
        return enhanced_activity
    
    def set_error_signal(self, error: float):
        """Set current error signal for error-driven plasticity."""
        self.current_error_signal = error
    
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