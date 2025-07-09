"""
Sequence-aware Columnar Hebbian SNN

Extends the base HebSNN to handle sequential language processing with:
- Columnar organization for tokens/concepts
- Temporal sequence memory
- Hierarchical processing layers
- Biological attention mechanisms
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import partial

from hebbianllm.core.network import HebSNN


class SequenceMemory:
    """Temporal sequence memory for maintaining context."""
    
    def __init__(self, max_length: int = 512, memory_decay: float = 0.95):
        self.max_length = max_length
        self.memory_decay = memory_decay
        self.sequence_buffer = []
        self.attention_weights = jnp.ones(max_length)
        
    def update(self, activity: jnp.ndarray, position: int):
        """Update sequence memory with new activity."""
        # Add to buffer
        if len(self.sequence_buffer) >= self.max_length:
            self.sequence_buffer.pop(0)
        self.sequence_buffer.append(activity)
        
        # Decay older memories
        self.attention_weights = self.attention_weights * self.memory_decay
        if position < len(self.attention_weights):
            self.attention_weights = self.attention_weights.at[position].set(1.0)
    
    def get_context(self, query_length: int = 10) -> jnp.ndarray:
        """Get relevant context from sequence memory."""
        if not self.sequence_buffer:
            return jnp.zeros((query_length, self.sequence_buffer[0].shape[0] if self.sequence_buffer else 100))
        
        # Get recent context with attention weighting
        context_length = min(query_length, len(self.sequence_buffer))
        context = jnp.array(self.sequence_buffer[-context_length:])
        
        # Apply attention weights
        weights = self.attention_weights[-context_length:]
        weighted_context = context * weights[:, None]
        
        return weighted_context


class ColumnarLayer:
    """A layer of columnar neural populations."""
    
    def __init__(self, n_columns: int, column_size: int = 100, dtype=jnp.float32):
        self.n_columns = n_columns
        self.column_size = column_size
        self.total_neurons = n_columns * column_size
        self.dtype = dtype
        
        # Column organization
        self.column_ranges = [(i * column_size, (i + 1) * column_size) 
                             for i in range(n_columns)]
        
        # Lateral inhibition for winner-take-all
        self.inhibition_strength = 2.0
        
    def activate_column(self, column_id: int, strength: float = 1.0) -> jnp.ndarray:
        """Activate a specific column."""
        activity = jnp.zeros(self.total_neurons, dtype=self.dtype)
        start, end = self.column_ranges[column_id]
        # Gaussian activity pattern within column
        column_activity = jnp.exp(-jnp.arange(self.column_size)**2 / (2 * (self.column_size/4)**2))
        activity = activity.at[start:end].set(column_activity * strength)
        return activity
    
    def apply_lateral_inhibition(self, activity: jnp.ndarray) -> jnp.ndarray:
        """Apply lateral inhibition between columns (biological softmax)."""
        # Compute column activities
        column_activities = jnp.array([
            jnp.sum(activity[start:end]) 
            for start, end in self.column_ranges
        ])
        
        # Winner-take-all competition
        max_activity = jnp.max(column_activities)
        inhibition = jnp.where(column_activities < max_activity * 0.5, 
                              -self.inhibition_strength, 0.0)
        
        # Apply inhibition to each column
        inhibited_activity = activity.copy()
        for i, (start, end) in enumerate(self.column_ranges):
            inhibited_activity = inhibited_activity.at[start:end].add(inhibition[i])
        
        return jnp.maximum(inhibited_activity, 0.0)


class SequenceHebSNN(HebSNN):
    """
    Sequence-aware Hebbian SNN for language processing.
    
    Features:
    - Columnar organization for vocabulary
    - Hierarchical processing layers
    - Temporal sequence memory
    - Biological attention mechanisms
    """
    
    def __init__(self,
                 vocab_size: int = 10000,
                 max_seq_length: int = 512,
                 column_size: int = 50,
                 n_layers: int = 3,
                 layer_sizes: Optional[List[int]] = None,
                 **kwargs):
        
        # Calculate network size based on columnar organization
        if layer_sizes is None:
            layer_sizes = [vocab_size, vocab_size // 2, vocab_size // 4]
        
        total_columns = sum(layer_sizes)
        total_neurons = total_columns * column_size
        
        # Initialize base network with calculated size
        super().__init__(
            n_sensory=total_neurons // 4,
            n_associative=total_neurons // 2,
            n_inhibitory=total_neurons // 8,
            n_output=total_neurons // 4,
            **kwargs
        )
        
        # Add current activity state for tracking
        self.current_activity = jnp.zeros(self.n_neurons, dtype=self.dtype)
        
        # Sequence-specific parameters
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.column_size = column_size
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        
        # Create columnar layers
        self.layers = []
        neuron_offset = 0
        
        for i, layer_size in enumerate(layer_sizes):
            layer = ColumnarLayer(layer_size, column_size)
            # Map layer to specific neurons in the network
            layer.neuron_offset = neuron_offset
            neuron_offset += layer.total_neurons
            self.layers.append(layer)
        
        # Sequence memory
        self.sequence_memory = SequenceMemory(max_seq_length)
        
        # Token to column mapping
        self.token_to_column = {}
        self.column_to_token = {}
        self._init_token_mapping()
        
        # Generation state
        self.generation_temperature = 1.0
        self.generation_context = []
        
        print(f"SequenceHebSNN initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Layers: {layer_sizes}")
        print(f"  Total columns: {total_columns}")
        print(f"  Total neurons: {self.n_neurons}")
    
    def _init_token_mapping(self):
        """Initialize mapping between tokens and columns."""
        # Map tokens to columns in first layer (vocabulary layer)
        vocab_layer = self.layers[0]
        for token_id in range(min(self.vocab_size, vocab_layer.n_columns)):
            self.token_to_column[token_id] = token_id
            self.column_to_token[token_id] = token_id
    
    def encode_token_sequence(self, token_ids: List[int]) -> jnp.ndarray:
        """Encode a sequence of tokens to neural activity patterns."""
        vocab_layer = self.layers[0]
        sequence_activity = jnp.zeros(self.n_neurons, dtype=self.dtype)
        
        # Encode each token in the vocabulary layer
        for i, token_id in enumerate(token_ids[-self.max_seq_length:]):
            if token_id in self.token_to_column:
                column_id = self.token_to_column[token_id]
                # Temporal encoding: recent tokens have higher activity
                temporal_weight = jnp.exp(-0.1 * (len(token_ids) - 1 - i))
                token_activity = vocab_layer.activate_column(column_id, temporal_weight)
                
                # Add to sequence activity
                start_idx = vocab_layer.neuron_offset
                end_idx = start_idx + vocab_layer.total_neurons
                sequence_activity = sequence_activity.at[start_idx:end_idx].add(token_activity)
        
        return sequence_activity
    
    def process_sequence(self, token_ids: List[int], n_steps: int = 10) -> Dict:
        """Process a sequence of tokens through the network."""
        # Encode sequence
        sequence_input = self.encode_token_sequence(token_ids)
        
        # Create batch with single sequence
        batch_input = sequence_input[None, :]  # Add batch dimension
        
        # Process through network
        results = self.batch_run(batch_input, n_steps=n_steps)
        
        # Update sequence memory
        final_activity = results['final_states']['v'][0]  # Remove batch dimension
        self.current_activity = final_activity  # Update current activity
        self.sequence_memory.update(final_activity, len(token_ids))
        
        return {
            'activity': final_activity,
            'spike_history': results['spike_history'][0],
            'sequence_encoding': sequence_input
        }
    
    def generate_next_token(self, context_tokens: List[int], temperature: float = 1.0) -> int:
        """Generate next token based on context."""
        # Process context
        context_result = self.process_sequence(context_tokens)
        
        # Get vocabulary layer activity
        vocab_layer = self.layers[0]
        start_idx = vocab_layer.neuron_offset
        end_idx = start_idx + vocab_layer.total_neurons
        vocab_activity = context_result['activity'][start_idx:end_idx]
        
        # Compute column activities
        column_activities = jnp.array([
            jnp.sum(vocab_activity[start:end])
            for start, end in vocab_layer.column_ranges
        ])
        
        # Apply temperature and softmax
        if temperature > 0:
            logits = column_activities / temperature
            probabilities = jax.nn.softmax(logits)
            
            # Sample from distribution
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
            next_column = jax.random.categorical(key, logits)
        else:
            # Greedy selection
            next_column = jnp.argmax(column_activities)
        
        # Convert column to token
        if int(next_column) in self.column_to_token:
            return self.column_to_token[int(next_column)]
        else:
            return 0  # Unknown token
    
    def generate_sequence(self, 
                         prompt_tokens: List[int], 
                         max_length: int = 50,
                         temperature: float = 1.0) -> List[int]:
        """Generate a sequence of tokens given a prompt."""
        generated = prompt_tokens.copy()
        
        for _ in range(max_length):
            # Use recent context for generation
            context = generated[-self.max_seq_length:]
            next_token = self.generate_next_token(context, temperature)
            
            # Stop on end token or repetition
            if next_token == 0 or (len(generated) > 1 and next_token == generated[-1]):
                break
                
            generated.append(next_token)
        
        return generated
    
    def get_layer_activities(self, activity: jnp.ndarray) -> Dict[int, jnp.ndarray]:
        """Get activity for each layer."""
        layer_activities = {}
        
        for i, layer in enumerate(self.layers):
            start_idx = layer.neuron_offset
            end_idx = start_idx + layer.total_neurons
            layer_activities[i] = activity[start_idx:end_idx]
        
        return layer_activities
    
    def get_column_activities(self, layer_id: int, activity: jnp.ndarray) -> jnp.ndarray:
        """Get column-wise activities for a specific layer."""
        if layer_id >= len(self.layers):
            return jnp.array([])
        
        layer = self.layers[layer_id]
        start_idx = layer.neuron_offset
        end_idx = start_idx + layer.total_neurons
        layer_activity = activity[start_idx:end_idx]
        
        # Compute column activities
        column_activities = jnp.array([
            jnp.sum(layer_activity[start:end])
            for start, end in layer.column_ranges
        ])
        
        return column_activities
    
    def get_top_active_tokens(self, activity: jnp.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get top-k most active tokens from vocabulary layer."""
        vocab_activities = self.get_column_activities(0, activity)
        
        # Get top-k indices and values
        top_indices = jnp.argsort(vocab_activities)[-top_k:][::-1]
        top_values = vocab_activities[top_indices]
        
        # Convert to token IDs
        top_tokens = []
        for idx, value in zip(top_indices, top_values):
            token_id = self.column_to_token.get(int(idx), int(idx))
            top_tokens.append((token_id, float(value)))
        
        return top_tokens
    
    def get_activity(self) -> jnp.ndarray:
        """Get current network activity."""
        return self.current_activity
    
    def reset_sequence_state(self):
        """Reset sequence-related state."""
        self.sequence_memory = SequenceMemory(self.max_seq_length)
        self.generation_context = []
        self.current_activity = jnp.zeros(self.n_neurons, dtype=self.dtype)
        self.reset()  # Reset base network state