#!/usr/bin/env python3
"""
Metal/GPU performance test for Hebbian SNN.

This script benchmarks the JAX-accelerated implementation against
the original CPU-based implementation.
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import gc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN
from hebbianllm.core.jax_ops import JAXHebSNN

def create_input_patterns(n_sensory: int, n_patterns: int = 5, pattern_size: int = 10):
    """Create test input patterns."""
    patterns = []
    for i in range(n_patterns):
        pattern = np.random.choice(n_sensory, pattern_size, replace=False)
        patterns.append(pattern)
    return patterns

def benchmark_original_network(network_sizes, patterns_list, duration=100.0):
    """Benchmark original CPU-based network."""
    print("Benchmarking original CPU-based network...")
    results = []
    
    for i, (n_sensory, n_associative, n_inhibitory, n_output) in enumerate(network_sizes):
        print(f"\nTesting size: {n_sensory}+{n_associative}+{n_inhibitory}+{n_output}")
        
        # Create network
        network = HebSNN(
            n_sensory=n_sensory,
            n_associative=n_associative,
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=0.1,
            seed=42
        )
        
        patterns = patterns_list[i]
        
        # Benchmark training
        start_time = time.time()
        
        for pattern in patterns:
            network.reset()
            network.inject_spikes(pattern.tolist())
            network.run(duration)
        
        elapsed = time.time() - start_time
        
        results.append({
            'size': f"{n_sensory}+{n_associative}+{n_inhibitory}+{n_output}",
            'time': elapsed,
            'neurons': n_sensory + n_associative + n_inhibitory + n_output
        })
        
        print(f"Time: {elapsed:.2f}s")
        
        # Cleanup
        del network
        gc.collect()
    
    return results

def benchmark_jax_network(network_sizes, patterns_list, duration=100.0):
    """Benchmark JAX-accelerated network."""
    print("\nBenchmarking JAX-accelerated network...")
    results = []
    
    for i, (n_sensory, n_associative, n_inhibitory, n_output) in enumerate(network_sizes):
        print(f"\nTesting size: {n_sensory}+{n_associative}+{n_inhibitory}+{n_output}")
        
        # Create network
        network = JAXHebSNN(
            n_sensory=n_sensory,
            n_associative=n_associative,
            n_inhibitory=n_inhibitory,
            n_output=n_output,
            connectivity_density=0.1,
            seed=42
        )
        
        patterns = patterns_list[i]
        
        # Benchmark training
        start_time = time.time()
        
        for pattern in patterns:
            network.reset()
            
            # Create input function
            def input_fn(t):
                if t < 10.0:  # Inject spikes at beginning
                    spikes = jnp.zeros(network.n_neurons)
                    spikes = spikes.at[pattern].set(1.0)
                    return spikes
                return jnp.zeros(network.n_neurons)
            
            network.run(duration, input_fn=input_fn)
        
        elapsed = time.time() - start_time
        
        results.append({
            'size': f"{n_sensory}+{n_associative}+{n_inhibitory}+{n_output}",
            'time': elapsed,
            'neurons': n_sensory + n_associative + n_inhibitory + n_output
        })
        
        print(f"Time: {elapsed:.2f}s")
        
        # Cleanup
        del network
        gc.collect()
    
    return results

def test_learning_quality():
    """Test learning quality of JAX implementation."""
    print("\nTesting learning quality...")
    
    # Small network for detailed analysis
    n_sensory, n_associative, n_inhibitory, n_output = 50, 200, 50, 50
    
    # Create both networks
    original_net = HebSNN(
        n_sensory=n_sensory,
        n_associative=n_associative,
        n_inhibitory=n_inhibitory,
        n_output=n_output,
        connectivity_density=0.1,
        seed=42
    )
    
    jax_net = JAXHebSNN(
        n_sensory=n_sensory,
        n_associative=n_associative,
        n_inhibitory=n_inhibitory,
        n_output=n_output,
        connectivity_density=0.1,
        seed=42
    )
    
    # Create test patterns
    patterns = create_input_patterns(n_sensory, n_patterns=3, pattern_size=5)
    
    # Train both networks
    print("Training original network...")
    for pattern in patterns:
        original_net.reset()
        original_net.inject_spikes(pattern.tolist())
        original_net.run(100.0)
    
    print("Training JAX network...")
    for pattern in patterns:
        jax_net.reset()
        
        def input_fn(t):
            if t < 10.0:
                spikes = jnp.zeros(jax_net.n_neurons)
                spikes = spikes.at[pattern].set(1.0)
                return spikes
            return jnp.zeros(jax_net.n_neurons)
        
        jax_net.run(100.0, input_fn=input_fn)
    
    # Test responses
    print("Testing responses...")
    
    original_responses = []
    jax_responses = []
    
    for pattern in patterns:
        # Original network
        original_net.reset()
        original_net.inject_spikes(pattern.tolist())
        original_net.run(50.0)
        orig_activity = original_net.get_output_activity()
        original_responses.append(list(orig_activity.values()))
        
        # JAX network
        jax_net.reset()
        
        def input_fn(t):
            if t < 10.0:
                spikes = jnp.zeros(jax_net.n_neurons)
                spikes = spikes.at[pattern].set(1.0)
                return spikes
            return jnp.zeros(jax_net.n_neurons)
        
        jax_net.run(50.0, input_fn=input_fn)
        jax_activity = jax_net.get_output_activity()
        jax_responses.append(list(jax_activity.values()))
    
    # Calculate pattern separation
    def calculate_separation(responses):
        if len(responses) <= 1:
            return 0.0
        
        separations = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                r1 = np.array(responses[i])
                r2 = np.array(responses[j])
                
                if np.linalg.norm(r1) > 0 and np.linalg.norm(r2) > 0:
                    cos_sim = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                    separations.append(1.0 - cos_sim)
        
        return np.mean(separations) if separations else 0.0
    
    orig_separation = calculate_separation(original_responses)
    jax_separation = calculate_separation(jax_responses)
    
    print(f"Original network pattern separation: {orig_separation:.4f}")
    print(f"JAX network pattern separation: {jax_separation:.4f}")
    
    return orig_separation, jax_separation

def plot_results(original_results, jax_results):
    """Plot benchmark results."""
    sizes = [r['size'] for r in original_results]
    original_times = [r['time'] for r in original_results]
    jax_times = [r['time'] for r in jax_results]
    
    # Calculate speedup
    speedups = [orig / jax for orig, jax in zip(original_times, jax_times)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Execution times
    x = np.arange(len(sizes))
    width = 0.35
    
    ax1.bar(x - width/2, original_times, width, label='Original CPU', alpha=0.8)
    ax1.bar(x + width/2, jax_times, width, label='JAX Accelerated', alpha=0.8)
    ax1.set_xlabel('Network Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax2.bar(x, speedups, alpha=0.8, color='green')
    ax2.set_xlabel('Network Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('JAX Speedup vs Original')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPerformance comparison saved to 'performance_comparison.png'")

def main():
    """Run performance benchmarks."""
    print("=" * 60)
    print("Hebbian SNN Performance Benchmark")
    print("=" * 60)
    
    # Show JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Test network sizes
    network_sizes = [
        (50, 200, 50, 50),      # Small: 350 neurons
        (100, 500, 100, 100),   # Medium: 800 neurons  
        (200, 1000, 200, 200),  # Large: 1600 neurons
        (500, 2000, 500, 500),  # XL: 3500 neurons
    ]
    
    # Create patterns for each size
    patterns_list = []
    for n_sensory, _, _, _ in network_sizes:
        patterns = create_input_patterns(n_sensory, n_patterns=3, pattern_size=5)
        patterns_list.append(patterns)
    
    # Run benchmarks
    print("\nRunning performance benchmarks...")
    print("This may take several minutes...")
    
    try:
        # Benchmark original implementation
        original_results = benchmark_original_network(network_sizes, patterns_list, duration=50.0)
        
        # Benchmark JAX implementation
        jax_results = benchmark_jax_network(network_sizes, patterns_list, duration=50.0)
        
        # Test learning quality
        orig_sep, jax_sep = test_learning_quality()
        
        # Display results
        print("\n" + "=" * 60)
        print("PERFORMANCE RESULTS")
        print("=" * 60)
        
        print(f"{'Network Size':<20} {'Original (s)':<15} {'JAX (s)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        for orig, jax_res in zip(original_results, jax_results):
            speedup = orig['time'] / jax_res['time']
            print(f"{orig['size']:<20} {orig['time']:<15.2f} {jax_res['time']:<15.2f} {speedup:<10.2f}x")
        
        print(f"\nLearning Quality:")
        print(f"Original network separation: {orig_sep:.4f}")
        print(f"JAX network separation: {jax_sep:.4f}")
        
        # Plot results
        plot_results(original_results, jax_results)
        
        # Summary
        avg_speedup = np.mean([orig['time'] / jax_res['time'] for orig, jax_res in zip(original_results, jax_results)])
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 2.0:
            print("✅ Significant performance improvement achieved!")
        elif avg_speedup > 1.2:
            print("✅ Moderate performance improvement achieved!")
        else:
            print("⚠️ Limited performance improvement - may need further optimization")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()