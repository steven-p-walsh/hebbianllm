#!/usr/bin/env python3
"""
Performance demonstration of HebLLM.
Shows performance scaling with different network sizes and batch sizes.
"""

import time
import jax
import jax.numpy as jnp
from hebbianllm import HebSNN


def benchmark_configuration(config: dict, n_steps: int = 100):
    """Benchmark a specific network configuration."""
    print(f"Testing {config['name']}...")
    
    # Create network
    network = HebSNN(
        n_sensory=config['n_sensory'],
        n_associative=config['n_associative'],
        n_inhibitory=config['n_inhibitory'],
        n_output=config['n_output'],
        batch_size=config['batch_size']
    )
    
    # Generate test patterns
    key = jax.random.PRNGKey(42)
    patterns = jax.random.bernoulli(
        key, 0.05, shape=(config['batch_size'], network.n_neurons)
    )
    
    # Benchmark
    start_time = time.time()
    results = network.batch_run(patterns, n_steps=n_steps)
    processing_time = time.time() - start_time
    
    # Calculate metrics
    total_operations = config['batch_size'] * n_steps
    performance = total_operations / processing_time
    memory_usage = network._estimate_memory_usage()
    
    return {
        'name': config['name'],
        'n_neurons': network.n_neurons,
        'batch_size': config['batch_size'],
        'performance': performance,
        'memory_usage': memory_usage,
        'processing_time': processing_time,
        'n_devices': network.n_devices
    }


def main():
    """Run performance benchmarks."""
    
    print("HebLLM Performance Benchmark")
    print("=" * 40)
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print()
    
    # Test configurations
    configurations = [
        {
            'name': 'Small Network',
            'n_sensory': 500,
            'n_associative': 2000,
            'n_inhibitory': 500,
            'n_output': 500,
            'batch_size': 32
        },
        {
            'name': 'Medium Network',
            'n_sensory': 1000,
            'n_associative': 4000,
            'n_inhibitory': 1000,
            'n_output': 1000,
            'batch_size': 64
        },
        {
            'name': 'Large Network',
            'n_sensory': 2000,
            'n_associative': 8000,
            'n_inhibitory': 2000,
            'n_output': 2000,
            'batch_size': 128
        }
    ]
    
    results = []
    
    for config in configurations:
        try:
            result = benchmark_configuration(config)
            results.append(result)
            print(f"✓ {result['name']}: {result['performance']:.1f} pattern-steps/sec")
        except Exception as e:
            print(f"✗ {config['name']}: Failed - {e}")
    
    # Summary
    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"{'Configuration':<15} {'Neurons':<8} {'Batch':<6} {'Performance':<12} {'Memory':<10} {'Devices':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<15} {result['n_neurons']:<8} {result['batch_size']:<6} "
              f"{result['performance']:<12.1f} {result['memory_usage']:<10.1f} {result['n_devices']:<8}")
    
    # Show GPU utilization if available
    if jax.devices('gpu'):
        print("\n" + "=" * 40)
        print("GPU UTILIZATION")
        print("=" * 40)
        
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'], 
                capture_output=True, text=True
            )
            for i, line in enumerate(result.stdout.strip().split('\n')):
                used, total, util = line.split(', ')
                print(f"GPU {i}: {used} MB / {total} MB ({int(used)/int(total)*100:.1f}%) - {util}% util")
        except:
            print("GPU status unavailable")
    
    print(f"\nBenchmark completed!")


if __name__ == "__main__":
    main()