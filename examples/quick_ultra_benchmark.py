#!/usr/bin/env python3
"""
Quick benchmark comparing all implementations.
"""

import sys
import os
import numpy as np
import time
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN
from hebbianllm.core.jax_ops import JAXHebSNN
from hebbianllm.core.ultra_jax_ops import UltraJAXHebSNN

def benchmark_implementations():
    """Quick benchmark of all implementations."""
    
    # Test configuration
    n_sensory, n_associative, n_inhibitory, n_output = 200, 800, 200, 200
    duration = 25.0
    
    print("=" * 60)
    print(f"PERFORMANCE COMPARISON")
    print(f"Network: {n_sensory + n_associative + n_inhibitory + n_output} neurons")
    print("=" * 60)
    
    # Test patterns
    patterns = [
        np.random.choice(n_sensory, 5, replace=False),
        np.random.choice(n_sensory, 5, replace=False),
        np.random.choice(n_sensory, 5, replace=False)
    ]
    
    results = {}
    
    # Test Original implementation
    print("1. Testing Original CPU implementation...")
    try:
        start_time = time.time()
        
        network = HebSNN(
            n_sensory=n_sensory,
            n_associative=n_associative,
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=0.05,
            seed=42
        )
        
        for pattern in patterns:
            network.reset()
            network.inject_spikes(pattern.tolist())
            network.run(duration)
        
        original_time = time.time() - start_time
        results['original'] = original_time
        print(f"   Time: {original_time:.2f}s")
        
    except Exception as e:
        print(f"   Error: {e}")
        results['original'] = None
    
    # Test JAX Basic implementation
    print("\n2. Testing JAX Basic implementation...")
    try:
        start_time = time.time()
        
        network = JAXHebSNN(
            n_sensory=n_sensory,
            n_associative=n_associative,
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=0.05,
            seed=42
        )
        
        for pattern in patterns:
            network.reset()
            
            def input_fn(t):
                if t < 10.0:
                    spikes = jnp.zeros(network.n_neurons)
                    spikes = spikes.at[pattern].set(1.0)
                    return spikes
                return jnp.zeros(network.n_neurons)
            
            network.run(duration, input_fn=input_fn)
        
        jax_time = time.time() - start_time
        results['jax_basic'] = jax_time
        print(f"   Time: {jax_time:.2f}s")
        
    except Exception as e:
        print(f"   Error: {e}")
        results['jax_basic'] = None
    
    # Test Ultra-optimized implementation
    print("\n3. Testing Ultra-optimized implementation...")
    try:
        start_time = time.time()
        
        network = UltraJAXHebSNN(
            n_sensory=n_sensory,
            n_associative=n_associative,
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=0.05,
            mixed_precision=True,
            seed=42
        )
        
        for pattern in patterns:
            network.reset()
            
            n_steps = int(duration)
            for step in range(n_steps):
                if step < 10:
                    inputs = jnp.zeros(network.n_neurons)
                    inputs = inputs.at[pattern].set(1.0)
                else:
                    inputs = jnp.zeros(network.n_neurons)
                
                network.step(inputs)
        
        ultra_time = time.time() - start_time
        results['ultra'] = ultra_time
        print(f"   Time: {ultra_time:.2f}s")
        
        # Get performance stats
        stats = network.get_performance_stats()
        print(f"   Memory usage: {stats['memory_usage_mb']:.1f}MB")
        print(f"   Connections: {stats['n_connections']:,}")
        
    except Exception as e:
        print(f"   Error: {e}")
        results['ultra'] = None
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"{'Implementation':<20} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 45)
    
    baseline = results.get('original', results.get('jax_basic', 1.0))
    
    for name, time_val in results.items():
        if time_val is not None:
            speedup = baseline / time_val if baseline and time_val else 1.0
            print(f"{name.capitalize():<20} {time_val:<12.2f} {speedup:<10.1f}x")
    
    # Find best performance
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1])
        print(f"\nFastest: {fastest[0]} ({fastest[1]:.2f}s)")
        
        if results['original'] and results['ultra']:
            total_speedup = results['original'] / results['ultra']
            print(f"Total speedup: {total_speedup:.1f}x")
    
    print("\nOptimizations implemented:")
    print("- Event-driven sparse simulation")
    print("- Memory pooling and pre-allocation")
    print("- Mixed precision (float16)")
    print("- Vectorized operations")
    print("- JIT compilation")
    print("- Efficient connectivity representation")

if __name__ == "__main__":
    benchmark_implementations()