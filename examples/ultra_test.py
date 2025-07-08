#!/usr/bin/env python3
"""
Quick test of ultra-optimized implementation.
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.ultra_jax_ops import UltraJAXHebSNN

def test_ultra_optimization():
    """Test ultra-optimized implementation."""
    print("Testing ultra-optimized implementation...")
    
    # Create a reasonably large network
    network = UltraJAXHebSNN(
        n_sensory=500,
        n_associative=2000,
        n_inhibitory=500,
        n_output=500,
        connectivity_density=0.1,
        mixed_precision=True,
        batch_size=16,
        seed=42
    )
    
    print(f"Network created with {network.n_neurons} neurons")
    print(f"Performance stats: {network.get_performance_stats()}")
    
    # Test single step
    print("\nTesting single step...")
    inputs = jnp.zeros(network.n_neurons)
    inputs = inputs.at[0:5].set(1.0)  # Activate first 5 neurons
    
    start_time = time.time()
    spikes, novelty = network.step(inputs)
    step_time = time.time() - start_time
    
    print(f"Single step time: {step_time:.4f}s")
    print(f"Spikes generated: {jnp.sum(spikes)}")
    print(f"Novelty signal: {novelty:.4f}")
    
    # Test multi-step simulation
    print("\nTesting multi-step simulation...")
    start_time = time.time()
    
    for i in range(50):
        if i < 10:
            inputs = jnp.zeros(network.n_neurons)
            inputs = inputs.at[i:i+5].set(1.0)
        else:
            inputs = jnp.zeros(network.n_neurons)
        
        spikes, novelty = network.step(inputs)
    
    multi_step_time = time.time() - start_time
    print(f"50 steps time: {multi_step_time:.4f}s")
    print(f"Time per step: {multi_step_time/50:.4f}s")
    
    # Test batch processing
    print("\nTesting batch processing...")
    batch_patterns = jnp.array([
        jnp.zeros(network.n_neurons).at[i*10:(i+1)*10].set(1.0)
        for i in range(8)
    ])
    
    start_time = time.time()
    batch_results = network.batch_run(batch_patterns, n_steps=20)
    batch_time = time.time() - start_time
    
    print(f"Batch processing time: {batch_time:.4f}s")
    print(f"Batch spike history shape: {batch_results['spike_history'].shape}")
    
    # Get output activity
    output_activity = network.get_output_activity()
    print(f"Output activity keys: {len(output_activity)}")
    
    print("\n✅ Ultra-optimized implementation test completed successfully!")
    
    return {
        'single_step_time': step_time,
        'multi_step_time': multi_step_time,
        'batch_time': batch_time,
        'neurons': network.n_neurons
    }

if __name__ == "__main__":
    test_ultra_optimization()