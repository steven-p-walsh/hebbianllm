#!/usr/bin/env python3
"""
Learning demonstration of HebLLM.
Shows how the network learns patterns through STDP.
"""

import time
import jax
import jax.numpy as jnp
from hebbianllm import HebSNN


def create_test_patterns(n_patterns: int, n_neurons: int, sparsity: float = 0.05):
    """Create test patterns for learning."""
    key = jax.random.PRNGKey(42)
    patterns = jax.random.bernoulli(key, sparsity, shape=(n_patterns, n_neurons))
    return patterns


def measure_weight_changes(network, patterns, n_steps: int = 50):
    """Measure weight changes during learning."""
    # Record initial weights
    initial_weights = jnp.copy(network.weights)
    
    # Run learning
    results = network.batch_run(patterns, n_steps=n_steps)
    
    # Calculate weight changes
    weight_changes = network.weights - initial_weights
    
    return {
        'initial_weights': initial_weights,
        'final_weights': network.weights,
        'weight_changes': weight_changes,
        'results': results
    }


def main():
    """Run learning demonstration."""
    
    print("HebLLM Learning Demonstration")
    print("=" * 40)
    
    # Create network
    network = HebSNN(
        n_sensory=200,
        n_associative=800,
        n_inhibitory=200,
        n_output=200,
        batch_size=8,
        connectivity_density=0.05  # Sparser for better learning observation
    )
    
    print(f"Created network with {network.n_neurons} neurons")
    print(f"Connections: {len(network.pre_indices):,}")
    print(f"Using {network.n_devices} device(s)")
    print()
    
    # Create test patterns
    patterns = create_test_patterns(8, network.n_neurons, sparsity=0.1)
    print(f"Created {len(patterns)} test patterns")
    
    # Measure learning over multiple epochs
    print("\nLearning Progress:")
    print("-" * 30)
    
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Measure weight changes
        learning_results = measure_weight_changes(network, patterns, n_steps=20)
        
        # Calculate statistics
        weight_changes = learning_results['weight_changes']
        total_change = jnp.sum(jnp.abs(weight_changes))
        mean_change = jnp.mean(jnp.abs(weight_changes))
        strengthened = jnp.sum(weight_changes > 0)
        weakened = jnp.sum(weight_changes < 0)
        
        print(f"  Total weight change: {total_change:.4f}")
        print(f"  Mean weight change: {mean_change:.6f}")
        print(f"  Strengthened connections: {strengthened}")
        print(f"  Weakened connections: {weakened}")
        
        # Show network activity
        spike_history = learning_results['results']['spike_history']
        total_spikes = jnp.sum(spike_history)
        sparsity = total_spikes / (8 * 20 * network.n_neurons)
        
        print(f"  Network activity: {total_spikes} spikes ({sparsity:.4f} sparsity)")
        print()
    
    # Test pattern recognition
    print("Testing Pattern Recognition:")
    print("-" * 30)
    
    # Present original patterns and measure response
    for i in range(min(3, len(patterns))):
        pattern = patterns[i:i+1]  # Single pattern
        
        # Reset network state
        network.reset()
        
        # Present pattern
        start_time = time.time()
        response = network.batch_run(pattern, n_steps=10)
        processing_time = time.time() - start_time
        
        # Analyze response
        spikes = response['spike_history']
        total_response = jnp.sum(spikes)
        
        # Get output activity
        output_start = network.n_sensory + network.n_associative + network.n_inhibitory
        output_spikes = spikes[0, :, output_start:]
        output_activity = jnp.sum(output_spikes)
        
        print(f"Pattern {i+1}:")
        print(f"  Total response: {total_response} spikes")
        print(f"  Output activity: {output_activity} spikes")
        print(f"  Processing time: {processing_time:.4f}s")
        print()
    
    # Final network statistics
    print("Final Network Statistics:")
    print("-" * 30)
    
    stats = network.get_performance_stats()
    print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"Connectivity density: {stats['connectivity_density']:.4f}")
    
    # Weight distribution
    weights = network.weights
    print(f"Weight statistics:")
    print(f"  Mean: {jnp.mean(weights):.4f}")
    print(f"  Std: {jnp.std(weights):.4f}")
    print(f"  Min: {jnp.min(weights):.4f}")
    print(f"  Max: {jnp.max(weights):.4f}")
    
    print("\nLearning demonstration completed!")


if __name__ == "__main__":
    main()