#!/usr/bin/env python3
"""
Simple demonstration of HebLLM.
Shows basic usage of the high-performance Hebbian spiking neural network.
"""

import time
import jax
import jax.numpy as jnp
from hebbianllm import HebSNN


def main():
    """Run a simple demonstration of HebLLM."""
    
    print("HebLLM Demo")
    print("=" * 30)
    
    # Create network
    network = HebSNN(
        n_sensory=100,
        n_associative=500,
        n_inhibitory=100,
        n_output=100,
        batch_size=16
    )
    
    print(f"Created network with {network.n_neurons} neurons")
    print(f"- Sensory: {network.n_sensory}")
    print(f"- Associative: {network.n_associative}")
    print(f"- Inhibitory: {network.n_inhibitory}")
    print(f"- Output: {network.n_output}")
    print(f"Using {network.n_devices} device(s)")
    print()
    
    # Generate test patterns
    key = jax.random.PRNGKey(42)
    patterns = jax.random.bernoulli(key, 0.05, shape=(16, network.n_neurons))
    
    print("Processing batch of 16 patterns for 50 time steps...")
    
    # Run batch processing
    start_time = time.time()
    results = network.batch_run(patterns, n_steps=50)
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.3f}s")
    print(f"Performance: {16*50/processing_time:.1f} pattern-steps/sec")
    
    # Show results
    spike_history = results['spike_history']
    total_spikes = jnp.sum(spike_history)
    print(f"Total spikes generated: {total_spikes}")
    print(f"Average spikes per pattern: {total_spikes / 16:.1f}")
    print(f"Network sparsity: {total_spikes / (16 * 50 * network.n_neurons) * 100:.2f}%")
    
    # Get performance statistics
    stats = network.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"- Memory usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"- Connections: {stats['n_connections']:,}")
    print(f"- Connectivity density: {stats['connectivity_density']:.3f}")
    print(f"- Data type: {stats['dtype']}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()