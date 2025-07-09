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
import time
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
        
        # Phase 3: Adenosine fatigue parameters
        self.adenosine_buildup_rate = 0.02   # How fast fatigue accumulates
        self.adenosine_decay_rate = 0.05     # How fast fatigue recovers
        self.fatigue_threshold = 0.7         # Above this, neurons get sluggish
        
        # Short-Term Plasticity (STP) parameters
        self.stp_facilitation_tau = 0.5      # Facilitation time constant
        self.stp_depression_tau = 0.8        # Depression time constant
        self.stp_buffer_decay = 0.1          # How fast STP buffer decays
        
        # Phase 4: Sleep replay & Synaptic tagging parameters
        self.tag_threshold = 0.3             # Minimum activity for tagging
        self.tag_decay_rate = 0.02           # How fast tags decay
        self.tag_capture_rate = 0.8          # How much tagged synapses get consolidated
        self.replay_strength = 0.5           # Strength of replay consolidation
        self.experience_importance_threshold = 0.7  # Threshold for storing experiences
    
    @partial(jit, static_argnums=(0,))
    def update_weights(self, weights: jnp.ndarray, 
                      pre_activity: jnp.ndarray, 
                      post_activity: jnp.ndarray,
                      eligibility_trace: jnp.ndarray,
                      modulators: Dict[str, float] = None) -> jnp.ndarray:
        """
        Update synaptic weights based on pre/post activity with neuromodulation.
        
        Implements:
        - Hebbian LTP/LTD
        - Synaptic decay
        - Homeostatic scaling
        - Neuromodulator gating (dopamine, acetylcholine, etc.)
        """
        # Initialize modulator dict if not provided
        if modulators is None:
            modulators = {}
        
        # Extract key modulators (default to 0.0 if not present)
        dopamine = modulators.get('dopamine', 0.0)
        acetylcholine = modulators.get('acetylcholine', 0.0)
        norepinephrine = modulators.get('norepinephrine', 0.0)
        adenosine = modulators.get('adenosine', 0.0)
        
        # Phase 1: Dopamine RPE-gated learning
        # Positive dopamine (RPE > 0) enhances LTP, reduces LTD
        # Negative dopamine (RPE < 0) reduces LTP, enhances LTD
        # Zero dopamine (RPE = 0) maintains baseline learning
        
        # Dopamine modulation parameters
        γ_DA = 0.3  # Dopamine gain factor (increased from 0.2 for stronger effect)
        
        # JAX-compatible dopamine modulation using lax.cond
        # Asymmetric modulation: positive dopamine enhances LTP more than LTD
        
        # For positive dopamine: enhance LTP, reduce LTD
        positive_ltp_rate = self.ltp_rate * (1.0 + γ_DA * dopamine)
        positive_ltd_rate = self.ltd_rate * (1.0 - 0.5 * γ_DA * dopamine)
        
        # For negative dopamine: reduce LTP, enhance LTD  
        negative_ltp_rate = self.ltp_rate * (1.0 + γ_DA * dopamine)  # Negative reduces LTP
        negative_ltd_rate = self.ltd_rate * (1.0 - γ_DA * dopamine)   # More forgetting
        
        # Use JAX conditionals for JIT compatibility
        modulated_ltp_rate = jnp.where(dopamine > 0, positive_ltp_rate,
                                      jnp.where(dopamine < 0, negative_ltp_rate, self.ltp_rate))
        modulated_ltd_rate = jnp.where(dopamine > 0, positive_ltd_rate,
                                      jnp.where(dopamine < 0, negative_ltd_rate, self.ltd_rate))
        
        # Phase 2: Acetylcholine attention + Norepinephrine novelty gain
        # Acetylcholine: enhances learning for attended patterns, reduces for unattended
        # Norepinephrine: boosts learning for novel/unexpected patterns
        
        # Acetylcholine modulation parameters
        γ_ACh = 0.4  # Acetylcholine gain factor (stronger than dopamine)
        
        # High acetylcholine: enhance learning for current pattern
        # Low acetylcholine: reduce learning (attention elsewhere)
        ach_ltp_boost = 1.0 + γ_ACh * acetylcholine
        ach_ltd_reduction = 1.0 - 0.3 * γ_ACh * acetylcholine  # Less forgetting when attending
        
        # Norepinephrine modulation parameters
        γ_NE = 0.25  # Norepinephrine gain factor (moderate)
        
        # High norepinephrine: boost learning for novel patterns
        # Affects both LTP and LTD for rapid adaptation
        ne_learning_boost = 1.0 + γ_NE * norepinephrine
        
        # Apply combined neuromodulation
        # Order: base rates -> dopamine -> acetylcholine -> norepinephrine -> adenosine
        phase2_ltp_rate = modulated_ltp_rate * ach_ltp_boost * ne_learning_boost
        phase2_ltd_rate = modulated_ltd_rate * ach_ltd_reduction * ne_learning_boost
        
        # Phase 3: Adenosine fatigue modulation
        # High adenosine: reduce learning (neurons are tired)
        # Low adenosine: normal learning (neurons are fresh)
        
        # Adenosine modulation parameters
        γ_ADO = 0.5  # Adenosine suppression factor
        
        # Adenosine suppresses both LTP and LTD (tired neurons learn less)
        adenosine_suppression = 1.0 - γ_ADO * adenosine
        
        # Final rates with adenosine fatigue
        final_ltp_rate = phase2_ltp_rate * adenosine_suppression
        final_ltd_rate = phase2_ltd_rate * adenosine_suppression
        # Hebbian plasticity: strengthen when pre and post fire together
        hebbian_update = jnp.outer(post_activity, pre_activity) * final_ltp_rate
        
        # Anti-Hebbian: weaken when only one fires
        anti_hebbian = (jnp.outer(post_activity, 1.0 - pre_activity) + 
                       jnp.outer(1.0 - post_activity, pre_activity)) * final_ltd_rate
        
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
    
    @partial(jit, static_argnums=(0,))
    def update_adenosine_fatigue(self, fatigue: jnp.ndarray, activity: jnp.ndarray) -> jnp.ndarray:
        """
        Update adenosine fatigue levels based on neural activity.
        
        Adenosine accumulates with activity and decays over time.
        High adenosine = tired neurons that learn less.
        """
        # Adenosine buildup proportional to activity
        adenosine_buildup = activity * self.adenosine_buildup_rate
        
        # Adenosine decay (clearance during rest)
        adenosine_decay = fatigue * self.adenosine_decay_rate
        
        # Update fatigue levels
        new_fatigue = fatigue + adenosine_buildup - adenosine_decay
        
        # Clamp fatigue to [0, 1] range
        new_fatigue = jnp.clip(new_fatigue, 0.0, 1.0)
        
        return new_fatigue
    
    @partial(jit, static_argnums=(0,))
    def update_stp_buffers(self, facilitation: jnp.ndarray, depression: jnp.ndarray, 
                          stp_buffer: jnp.ndarray, pre_activity: jnp.ndarray, 
                          post_activity: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Update Short-Term Plasticity (STP) buffers.
        
        STP provides temporary synaptic changes that can either facilitate or depress
        transmission based on recent activity patterns.
        """
        # STP facilitation: synapses get stronger with use (up to a point)
        # Facilitation increases with coincident activity
        facilitation_increase = jnp.outer(post_activity, pre_activity) * 0.1
        
        # Facilitation decays over time
        facilitation_decay = (facilitation - 1.0) / self.stp_facilitation_tau
        
        # Update facilitation
        new_facilitation = facilitation + facilitation_increase - facilitation_decay
        new_facilitation = jnp.clip(new_facilitation, 0.1, 2.0)  # Bounded facilitation
        
        # STP depression: synapses get weaker with overuse
        # Depression increases with high activity
        depression_increase = jnp.outer(post_activity, pre_activity) * 0.05
        
        # Depression recovers over time
        depression_recovery = (1.0 - depression) / self.stp_depression_tau
        
        # Update depression
        new_depression = depression - depression_increase + depression_recovery
        new_depression = jnp.clip(new_depression, 0.1, 1.0)  # Bounded depression
        
        # STP buffer: temporary weight changes
        # Buffer accumulates recent activity-dependent changes
        buffer_update = jnp.outer(post_activity, pre_activity) * (new_facilitation * new_depression - 1.0) * 0.1
        
        # Buffer decays over time
        buffer_decay = stp_buffer * self.stp_buffer_decay
        
        # Update buffer
        new_stp_buffer = stp_buffer + buffer_update - buffer_decay
        new_stp_buffer = jnp.clip(new_stp_buffer, -0.5, 0.5)  # Bounded buffer
        
        return new_facilitation, new_depression, new_stp_buffer
    
    @partial(jit, static_argnums=(0,))
    def update_synaptic_tags(self, tags: jnp.ndarray, pre_activity: jnp.ndarray, 
                           post_activity: jnp.ndarray, importance: float) -> jnp.ndarray:
        """
        Update synaptic tags for consolidation based on activity and importance.
        
        Synaptic tagging marks synapses that participated in important experiences
        for later consolidation during sleep replay.
        """
        # Tag synapses that are active during important experiences
        activity_based_tagging = jnp.outer(post_activity, pre_activity)
        
        # Only tag if activity is above threshold and experience is important
        should_tag = (activity_based_tagging > self.tag_threshold) & (importance > 0.5)
        
        # Increase tags for important active synapses
        tag_increase = jnp.where(should_tag, 
                                activity_based_tagging * importance * 0.3, 
                                0.0)
        
        # Tags decay over time
        tag_decay = tags * self.tag_decay_rate
        
        # Update tags
        new_tags = tags + tag_increase - tag_decay
        
        # Clamp tags to [0, 1] range
        new_tags = jnp.clip(new_tags, 0.0, 1.0)
        
        return new_tags
    
    @partial(jit, static_argnums=(0,))
    def apply_synaptic_capture(self, weights: jnp.ndarray, tags: jnp.ndarray, 
                              stp_buffer: jnp.ndarray) -> jnp.ndarray:
        """
        Apply synaptic capture: consolidate STP changes into LTP based on tags.
        
        Tagged synapses get their STP changes consolidated into permanent weights
        more effectively than untagged synapses.
        """
        # Calculate capture strength based on tags
        capture_strength = tags * self.tag_capture_rate
        
        # Consolidate STP buffer into permanent weights proportional to tags
        consolidation = stp_buffer * capture_strength
        
        # Apply consolidation to weights
        new_weights = weights + consolidation
        
        # Clamp weights to reasonable range
        new_weights = jnp.clip(new_weights, -1.0, 1.0)
        
        return new_weights
    
    @partial(jit, static_argnums=(0,))
    def generate_replay_activity(self, stored_pattern: jnp.ndarray, 
                                replay_traces: jnp.ndarray) -> jnp.ndarray:
        """
        Generate replay activity pattern based on stored experience.
        
        Creates a noisy, compressed version of the original activity pattern
        for sleep replay consolidation.
        """
        # Add noise to stored pattern (replay is imperfect)
        noise_level = 0.1
        key = jax.random.PRNGKey(42)  # Fixed seed for deterministic replay
        noise = jax.random.normal(key, stored_pattern.shape) * noise_level
        
        # Combine stored pattern with replay traces (now both 1D)
        replay_activity = stored_pattern * 0.7 + replay_traces * 0.3 + noise
        
        # Apply sparse coding (only top activations)
        k = max(1, int(0.03 * len(replay_activity)))  # Even sparser than wake (3%)
        _, top_indices = jax.lax.top_k(replay_activity, k)
        
        # Create sparse replay pattern
        sparse_replay = jnp.zeros_like(replay_activity)
        sparse_replay = sparse_replay.at[top_indices].set(replay_activity[top_indices])
        
        # Normalize and scale for replay (JAX-compatible)
        sum_activity = jnp.sum(sparse_replay)
        max_activity = jnp.max(sparse_replay)
        
        # Use JAX conditional to avoid boolean conversion error
        normalized_replay = jnp.where(
            sum_activity > 0,
            sparse_replay / jnp.maximum(max_activity, 1e-8) * self.replay_strength,
            sparse_replay
        )
        
        return normalized_replay


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
        
        # Phase 3: Adenosine fatigue + STP buffers (memory-optimized)
        self.neuron_fatigue = jnp.zeros(self.n_neurons, dtype=self.dtype)  # Adenosine levels
        # Use much smaller STP buffers - only track active connections
        max_stp_connections = min(50000, self.n_neurons * 10)  # Limit STP buffer size  
        self.stp_indices = jnp.zeros((max_stp_connections, 2), dtype=jnp.int32)  # (pre, post) indices
        self.stp_facilitation = jnp.ones(max_stp_connections, dtype=self.dtype)  # STP facilitation values
        self.stp_depression = jnp.ones(max_stp_connections, dtype=self.dtype)    # STP depression values
        self.stp_buffer = jnp.zeros(max_stp_connections, dtype=self.dtype)       # Temporary changes
        self.stp_active_count = 0  # Number of active STP connections
        
        # Phase 4: Sleep replay & Synaptic tagging (memory-optimized)
        # Use sparse representation for synaptic tags - only track tagged synapses
        max_tagged_synapses = min(10000, self.n_neurons * 5)  # Limit tagged synapses
        self.tag_indices = jnp.zeros((max_tagged_synapses, 2), dtype=jnp.int32)  # (pre, post) indices  
        self.synaptic_tags = jnp.zeros(max_tagged_synapses, dtype=self.dtype)    # Tag strength values
        self.tagged_count = 0  # Number of active tags
        self.experience_buffer = []  # Store important experiences for replay
        # Use smaller replay traces - just track neuron-level activity
        self.replay_traces = jnp.zeros(self.n_neurons, dtype=self.dtype)  # Activity traces for replay
        
        # Current activity for tracking
        self.current_activity = jnp.zeros(self.n_neurons, dtype=self.dtype)
        
        # Phase 5: Add modulators for proper neuromodulation
        from hebbianllm.core.network import Modulators
        self.modulators = Modulators()
        
        # Phase 5: Optimization state for incremental updates
        self.correlation_matrix = jnp.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)
        self.correlation_update_count = 0
        self.last_structural_update = 0
        
        # Phase 5: Performance monitoring
        self.performance_stats = {
            'structural_updates': 0,
            'structural_skips': 0,
            'connectivity_caps': 0,
            'sleep_consolidations': 0
        }
        
        print(f"PlasticHebSNN initialized:")
        print(f"  Fixed neurons: {self.n_neurons:,}")
        print(f"  Vocab capacity: {vocab_size:,}")
        print(f"  Initial connectivity: {initial_connectivity:.1%}")
        print(f"  Neurons per token: {self.token_mapper.neurons_per_token}")
        print(f"  Potential synapses: {self.n_neurons**2:,}")
        print(f"  Using device: {self.devices[0] if self.devices else 'CPU'}")
        print(f"  Initial synaptic tags: {float(jnp.mean(self.synaptic_tags)):.3f}")
    
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
    
    def process_tokens(self, token_ids: List[int], learning: bool = True, modulators: Dict[str, float] = None) -> Dict:
        """
        Process tokens through the plastic network with neuromodulation.
        
        Args:
            token_ids: List of token IDs to process
            learning: Whether to apply plasticity updates
            modulators: Dictionary of neuromodulator concentrations
            
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
            self._apply_plasticity_updates(activity_sequence, modulators)
        
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
    
    def _apply_plasticity_updates(self, activity_sequence: List[jnp.ndarray], modulators: Dict[str, float] = None):
        """Apply plasticity updates based on activity sequence with neuromodulation."""
        self.learning_step += 1
        
        # Update eligibility trace (decaying memory of recent activity)
        if len(activity_sequence) >= 2:
            pre_activity = activity_sequence[-2]
            post_activity = activity_sequence[-1]
            
            # Store old weights for change tracking
            old_weights = self.synaptic_weights.copy()
            
            # Phase 3: Update adenosine fatigue based on activity
            self.neuron_fatigue = self.plasticity.update_adenosine_fatigue(
                self.neuron_fatigue, post_activity
            )
            
            # Set adenosine modulator based on average fatigue
            if modulators is None:
                modulators = {}
            modulators['adenosine'] = float(jnp.mean(self.neuron_fatigue))
            
            # Phase 3: Update STP buffers (disabled for memory efficiency)
            # TODO: Implement sparse STP buffer updates
            # self.stp_facilitation, self.stp_depression, self.stp_buffer = self.plasticity.update_stp_buffers(
            #     self.stp_facilitation, self.stp_depression, self.stp_buffer,
            #     pre_activity, post_activity
            # )
            
            # Apply STP buffer to weights temporarily (disabled for memory efficiency)
            # TODO: Implement sparse STP buffer application
            effective_weights = self.synaptic_weights  # Use original weights for now
            
            # Update eligibility trace with decay
            self.eligibility_trace = self.eligibility_trace * 0.9 + jnp.outer(post_activity, pre_activity) * 0.1
            
            # Apply synaptic plasticity with neuromodulation (including adenosine)
            self.synaptic_weights = self.plasticity.update_weights(
                effective_weights,  # Use STP-modified weights
                pre_activity,
                post_activity, 
                self.eligibility_trace,
                modulators
            )
            
            # Log weight changes to verify plasticity is working
            weight_change = jnp.sum(jnp.abs(self.synaptic_weights - old_weights))
            if self.learning_step % 10 == 0:  # Log every 10 steps to avoid spam
                avg_fatigue = float(jnp.mean(self.neuron_fatigue))
                avg_stp = float(jnp.mean(jnp.abs(self.stp_buffer)))
                print(f"Step {self.learning_step}: Weight change = {float(weight_change):.6f}, "
                      f"Fatigue = {avg_fatigue:.3f}, STP = {avg_stp:.3f}")
            
            # Phase 4: Update synaptic tags based on experience importance
            experience_importance = self._compute_experience_importance(modulators)
            self.synaptic_tags = self.plasticity.update_synaptic_tags(
                self.synaptic_tags, pre_activity, post_activity, experience_importance
            )
            
            # Store important experiences for replay
            if experience_importance > self.plasticity.experience_importance_threshold:
                self._store_experience_for_replay(post_activity, experience_importance, modulators)
            
            # Apply connectivity cap to prevent saturation
            self._apply_connectivity_cap()
        
        # Structural plasticity every few steps
        if self.learning_step % 10 == 0:
            self._apply_structural_plasticity()
    
    def _apply_connectivity_cap(self, max_connectivity: float = 0.12):
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
            
            # Phase 5: Performance monitoring
            self.performance_stats['connectivity_caps'] += 1
            self.performance_stats['sleep_consolidations'] += 1
    
    def _sleep_consolidation(self):
        """Sleep-like consolidation: strengthen important connections, prune weak ones."""
        # Memory-optimized sleep mechanism for large networks
        
        # Use a more conservative percentage to avoid memory issues
        target_connectivity = 0.03  # 3% instead of 5% to reduce memory usage
        
        # Work with chunks to avoid creating large intermediate tensors
        chunk_size = 1000  # Process 1000 neurons at a time
        consolidated_weights = jnp.zeros_like(self.synaptic_weights)
        
        # Global threshold based on current connectivity
        current_weights = jnp.abs(self.synaptic_weights)
        # Sample a subset to estimate threshold quickly
        sample_size = min(10000, current_weights.size)
        sample_indices = jax.random.choice(
            jax.random.PRNGKey(42), 
            current_weights.size, 
            shape=(sample_size,), 
            replace=False
        )
        sample_weights = current_weights.flatten()[sample_indices]
        threshold = jnp.percentile(sample_weights, 97)  # Top 3% threshold
        
        # Apply threshold-based consolidation (more memory efficient)
        strong_connections = current_weights > threshold
        consolidated_weights = jnp.where(
            strong_connections,
            self.synaptic_weights * 1.05,  # Modest strengthening
            self.synaptic_weights * 0.1   # Weak pruning instead of complete removal
        )
        
        return consolidated_weights
    
    def _compute_experience_importance(self, modulators: Dict[str, float] = None) -> float:
        """
        Compute the importance of current experience for tagging and replay.
        
        Important experiences get tagged and stored for sleep replay.
        """
        if modulators is None:
            modulators = {}
        
        # Base importance factors
        dopamine_importance = abs(modulators.get('dopamine', 0.0))  # RPE magnitude
        attention_importance = modulators.get('acetylcholine', 0.0)  # Attention level
        novelty_importance = modulators.get('norepinephrine', 0.0)  # Novelty level
        
        # Combine importance factors
        combined_importance = (
            dopamine_importance * 0.4 +      # Strong RPE is very important
            attention_importance * 0.3 +     # High attention is important
            novelty_importance * 0.3         # Novel experiences are important
        )
        
        # Normalize to [0, 1] range
        importance = jnp.clip(combined_importance, 0.0, 1.0)
        
        return float(importance)
    
    def _store_experience_for_replay(self, activity: jnp.ndarray, importance: float, 
                                   modulators: Dict[str, float] = None):
        """
        Store important experience for sleep replay.
        
        Stores activity pattern and context for later replay during sleep.
        """
        experience = {
            'activity': activity.copy(),
            'importance': importance,
            'learning_step': self.learning_step,
            'modulators': modulators.copy() if modulators else {},
            'timestamp': time.time()
        }
        
        # Add to experience buffer
        self.experience_buffer.append(experience)
        
        # Keep only most recent and important experiences
        if len(self.experience_buffer) > 50:
            # Sort by importance and keep top experiences
            self.experience_buffer.sort(key=lambda x: x['importance'], reverse=True)
            self.experience_buffer = self.experience_buffer[:50]
        
        # Update replay traces with this experience (memory-optimized)
        self.replay_traces = self.replay_traces * 0.9 + activity * 0.1
    
    def _apply_structural_plasticity(self):
        """Apply structural plasticity with Phase 5 optimizations."""
        if len(self.activity_history) >= 2:
            # Phase 5: Adaptive frequency based on network activity
            activity_change = self._compute_activity_change()
            steps_since_last = self.learning_step - self.last_structural_update
            
            # Only update if significant activity change or enough time has passed
            if activity_change > 0.1 or steps_since_last >= 20:
                # Phase 5: Incremental correlation update
                self._update_correlation_matrix()
                
                # Apply structural plasticity with current correlation matrix
                self.synaptic_weights = self.plasticity.prune_and_form_synapses(
                    self.synaptic_weights, 
                    jnp.abs(self.correlation_matrix)
                )
                
                self.last_structural_update = self.learning_step
                self.performance_stats['structural_updates'] += 1
            else:
                self.performance_stats['structural_skips'] += 1
    
    def _compute_activity_change(self) -> float:
        """Compute change in activity pattern to determine if structural update is needed."""
        if len(self.activity_history) < 2:
            return 1.0  # Force update if insufficient history
        
        # Compare recent activity to older activity
        recent_activity = jnp.array(self.activity_history[-3:])  # Last 3 steps
        older_activity = jnp.array(self.activity_history[-6:-3])  # Previous 3 steps
        
        if len(older_activity) == 0:
            return 1.0
            
        # Compute cosine similarity between recent and older patterns
        recent_mean = jnp.mean(recent_activity, axis=0)
        older_mean = jnp.mean(older_activity, axis=0)
        
        # Normalize vectors
        recent_norm = jnp.linalg.norm(recent_mean)
        older_norm = jnp.linalg.norm(older_mean)
        
        if recent_norm == 0 or older_norm == 0:
            return 1.0
            
        similarity = jnp.dot(recent_mean, older_mean) / (recent_norm * older_norm)
        change = 1.0 - similarity  # Higher change means less similarity
        
        return float(change)
    
    def _update_correlation_matrix(self):
        """Incrementally update correlation matrix for efficiency."""
        if len(self.activity_history) < 2:
            return
            
        # Phase 5: Use exponential moving average for correlation updates
        alpha = 0.1  # Learning rate for correlation updates
        
        # Get current and previous activity
        current_activity = jnp.array(self.activity_history[-1])
        
        # Update correlation matrix incrementally using outer product
        activity_outer = jnp.outer(current_activity, current_activity)
        
        # Exponential moving average update
        self.correlation_matrix = (1 - alpha) * self.correlation_matrix + alpha * activity_outer
        
        # Handle numerical issues
        self.correlation_matrix = jnp.nan_to_num(self.correlation_matrix)
        
        self.correlation_update_count += 1
    
    def get_performance_stats(self) -> Dict:
        """Get Phase 5 performance optimization statistics."""
        total_structural_calls = self.performance_stats['structural_updates'] + self.performance_stats['structural_skips']
        structural_efficiency = (self.performance_stats['structural_skips'] / max(total_structural_calls, 1)) * 100
        
        return {
            'structural_updates': self.performance_stats['structural_updates'],
            'structural_skips': self.performance_stats['structural_skips'],
            'structural_efficiency': f"{structural_efficiency:.1f}%",
            'connectivity_caps': self.performance_stats['connectivity_caps'],
            'sleep_consolidations': self.performance_stats['sleep_consolidations'],
            'correlation_updates': self.correlation_update_count,
            'learning_step': self.learning_step,
            'last_structural_update': self.last_structural_update
        }
    
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
            
            # Filter out special tokens that shouldn't be generated
            special_tokens_to_avoid = [0, 1, 2, 3]  # PAD, UNK, BOS, EOS
            next_tokens = [t for t in next_tokens if t not in special_tokens_to_avoid]
            
            if next_tokens:
                # Create more balanced sampling weights that give PAUSE tokens a fair chance
                # Use a gentler decay that doesn't heavily penalize later positions
                weights = jnp.array([0.9 ** i for i in range(len(next_tokens))])
                
                # Give PAUSE tokens (ID 4) a significant boost if they're in the list
                # This is important for natural speech generation with proper pauses
                if 4 in next_tokens:
                    pause_idx = next_tokens.index(4)
                    # Give PAUSE tokens a much stronger boost to ensure they get selected
                    weights = weights.at[pause_idx].set(weights[pause_idx] * 3.0)
                
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
        
        # Filter out special tokens that shouldn't be generated
        special_tokens_to_avoid = [0, 1, 2, 3]  # PAD, UNK, BOS, EOS
        available_tokens = [t for t in available_tokens if t not in special_tokens_to_avoid]
        
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
        
        # Phase 3: Adenosine fatigue and STP stats
        fatigue_stats = {
            'mean_fatigue': float(jnp.mean(self.neuron_fatigue)),
            'max_fatigue': float(jnp.max(self.neuron_fatigue)),
            'n_tired_neurons': int(jnp.sum(self.neuron_fatigue > self.plasticity.fatigue_threshold))
        }
        
        stp_stats = {
            'mean_facilitation': float(jnp.mean(self.stp_facilitation)),
            'mean_depression': float(jnp.mean(self.stp_depression)),
            'mean_stp_buffer': float(jnp.mean(jnp.abs(self.stp_buffer))),
            'max_stp_buffer': float(jnp.max(jnp.abs(self.stp_buffer)))
        }
        
        # Phase 4: Sleep replay & Synaptic tagging stats
        tagging_stats = {
            'mean_synaptic_tags': float(jnp.mean(self.synaptic_tags)),
            'max_synaptic_tags': float(jnp.max(self.synaptic_tags)),
            'n_tagged_synapses': int(jnp.sum(self.synaptic_tags > 0.1)),
            'n_stored_experiences': len(self.experience_buffer)
        }
        
        replay_stats = {
            'mean_replay_traces': float(jnp.mean(jnp.abs(self.replay_traces))),
            'max_replay_traces': float(jnp.max(jnp.abs(self.replay_traces))),
            'experience_buffer_size': len(self.experience_buffer)
        }
        
        return {
            'learning_step': self.learning_step,
            'weights': weight_stats,
            'activity': activity_stats,
            'token_mappings': len(self.token_mapper.token_mappings),
            'tonic_strength': tonic_strength,
            'maturity_factor': maturity_factor,
            'is_mature': maturity_factor < 0.1,
            'fatigue': fatigue_stats,
            'stp': stp_stats,
            'tagging': tagging_stats,
            'replay': replay_stats
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
                # Phase 3 state
                'neuron_fatigue': np.array(self.neuron_fatigue),
                'stp_facilitation': np.array(self.stp_facilitation),
                'stp_depression': np.array(self.stp_depression),
                'stp_buffer': np.array(self.stp_buffer),
                # Phase 4 state
                'synaptic_tags': np.array(self.synaptic_tags),
                'replay_traces': np.array(self.replay_traces),
                'experience_buffer': self.experience_buffer,
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
            
            # Restore Phase 3 state (with defaults for backward compatibility)
            self.neuron_fatigue = jnp.array(data.get('neuron_fatigue', 
                                                   np.zeros(self.n_neurons, dtype=self.dtype)))
            self.stp_facilitation = jnp.array(data.get('stp_facilitation', 
                                                      np.ones((self.n_neurons, self.n_neurons), dtype=self.dtype)))
            self.stp_depression = jnp.array(data.get('stp_depression', 
                                                    np.ones((self.n_neurons, self.n_neurons), dtype=self.dtype)))
            self.stp_buffer = jnp.array(data.get('stp_buffer', 
                                               np.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)))
            
            # Restore Phase 4 state (with defaults for backward compatibility)
            self.synaptic_tags = jnp.array(data.get('synaptic_tags', 
                                                  np.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)))
            self.replay_traces = jnp.array(data.get('replay_traces', 
                                                  np.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype)))
            self.experience_buffer = data.get('experience_buffer', [])
            
            # Restore token mappings
            self.token_mapper.token_mappings = data['token_mappings'].item()
            self.token_mapper.neuron_usage = jnp.array(data['neuron_usage'])
            
            print(f"Network state loaded from {filepath}")
            print(f"Restored learning step: {self.learning_step}")
            
        except Exception as e:
            print(f"Failed to load network state: {e}")
            print("Continuing with fresh network state")