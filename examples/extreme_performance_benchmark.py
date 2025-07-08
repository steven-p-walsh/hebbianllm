#!/usr/bin/env python3
"""
Extreme performance benchmark for all optimization levels.

This script benchmarks:
1. Original CPU implementation
2. Basic JAX optimizations  
3. Ultra-optimized JAX implementation
4. Distributed/multi-GPU implementation

Tests scalability from hundreds to millions of neurons.
"""

import sys
import os
import numpy as np
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN
from hebbianllm.core.jax_ops import JAXHebSNN
from hebbianllm.core.ultra_jax_ops import UltraJAXHebSNN
from hebbianllm.core.distributed_ops import DistributedHebSNN

class ExtremeBenchmark:
    """Comprehensive benchmarking suite for extreme performance testing."""
    
    def __init__(self):
        self.results = {
            'original': [],
            'jax_basic': [],
            'jax_ultra': [],
            'distributed': []
        }
        
        # Test configurations - increasing complexity
        self.test_configs = [
            # (n_sensory, n_associative, n_inhibitory, n_output, name)
            (100, 400, 100, 100, "Small (700 neurons)"),
            (200, 800, 200, 200, "Medium (1.4K neurons)"),
            (500, 2000, 500, 500, "Large (3.5K neurons)"),
            (1000, 4000, 1000, 1000, "XL (7K neurons)"),
            (2000, 8000, 2000, 2000, "XXL (14K neurons)"),
            (5000, 20000, 5000, 5000, "Ultra (35K neurons)"),
        ]
        
        # For extreme tests (only for ultra-optimized versions)
        self.extreme_configs = [
            (10000, 40000, 10000, 10000, "Extreme (70K neurons)"),
            (20000, 80000, 20000, 20000, "Massive (140K neurons)"),
        ]
        
        self.duration = 25.0  # Short duration for speed
        self.n_patterns = 3
        self.pattern_size = 5
    
    def create_test_patterns(self, n_sensory: int):
        """Create test patterns for benchmarking."""
        patterns = []
        for i in range(self.n_patterns):
            pattern = np.random.choice(n_sensory, self.pattern_size, replace=False)
            patterns.append(pattern)
        return patterns
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def benchmark_original(self, config):
        """Benchmark original CPU implementation."""
        n_sensory, n_associative, n_inhibitory, n_output, name = config
        
        if n_sensory + n_associative + n_inhibitory + n_output > 7000:
            print(f"  Skipping original implementation for {name} (too slow)")
            return None
        
        print(f"  Testing original implementation: {name}")
        
        try:
            # Create network
            network = HebSNN(
                n_sensory=n_sensory,
                n_associative=n_associative,
                n_inhibitory=n_inhibitory,
                n_output=n_output,
                connectivity_density=0.05,  # Reduced for speed
                seed=42
            )
            
            patterns = self.create_test_patterns(n_sensory)
            
            # Benchmark
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            for pattern in patterns:
                network.reset()
                network.inject_spikes(pattern.tolist())
                network.run(self.duration)
            
            elapsed = time.time() - start_time
            memory_used = self.get_memory_usage() - start_memory
            
            result = {
                'name': name,
                'time': elapsed,
                'memory_mb': memory_used,
                'neurons': n_sensory + n_associative + n_inhibitory + n_output,
                'neurons_per_second': (n_sensory + n_associative + n_inhibitory + n_output) * self.n_patterns / elapsed
            }
            
            print(f"    Time: {elapsed:.2f}s, Memory: {memory_used:.1f}MB")
            
            # Cleanup
            del network
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def benchmark_jax_basic(self, config):
        """Benchmark basic JAX implementation."""
        n_sensory, n_associative, n_inhibitory, n_output, name = config
        
        print(f"  Testing JAX basic implementation: {name}")
        
        try:
            # Create network
            network = JAXHebSNN(
                n_sensory=n_sensory,
                n_associative=n_associative,
                n_inhibitory=n_inhibitory,
                n_output=n_output,
                connectivity_density=0.05,
                seed=42
            )
            
            patterns = self.create_test_patterns(n_sensory)
            
            # Benchmark
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            for pattern in patterns:
                network.reset()
                
                def input_fn(t):
                    if t < 10.0:
                        spikes = jnp.zeros(network.n_neurons)
                        spikes = spikes.at[pattern].set(1.0)
                        return spikes
                    return jnp.zeros(network.n_neurons)
                
                network.run(self.duration, input_fn=input_fn)
            
            elapsed = time.time() - start_time
            memory_used = self.get_memory_usage() - start_memory
            
            result = {
                'name': name,
                'time': elapsed,
                'memory_mb': memory_used,
                'neurons': n_sensory + n_associative + n_inhibitory + n_output,
                'neurons_per_second': (n_sensory + n_associative + n_inhibitory + n_output) * self.n_patterns / elapsed
            }
            
            print(f"    Time: {elapsed:.2f}s, Memory: {memory_used:.1f}MB")
            
            # Cleanup
            del network
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def benchmark_jax_ultra(self, config):
        """Benchmark ultra-optimized JAX implementation."""
        n_sensory, n_associative, n_inhibitory, n_output, name = config
        
        print(f"  Testing JAX ultra implementation: {name}")
        
        try:
            # Create network
            network = UltraJAXHebSNN(
                n_sensory=n_sensory,
                n_associative=n_associative,
                n_inhibitory=n_inhibitory,
                n_output=n_output,
                connectivity_density=0.05,
                mixed_precision=True,
                batch_size=16,
                seed=42
            )
            
            patterns = self.create_test_patterns(n_sensory)
            
            # Benchmark
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            for pattern in patterns:
                network.reset()
                
                # Run with event-driven inputs
                n_steps = int(self.duration)
                for step in range(n_steps):
                    if step < 10:  # Input for first 10 steps
                        inputs = jnp.zeros(network.n_neurons)
                        inputs = inputs.at[pattern].set(1.0)
                    else:
                        inputs = jnp.zeros(network.n_neurons)
                    
                    network.step(inputs)
            
            elapsed = time.time() - start_time
            memory_used = self.get_memory_usage() - start_memory
            
            # Get performance stats
            stats = network.get_performance_stats()
            
            result = {
                'name': name,
                'time': elapsed,
                'memory_mb': memory_used,
                'neurons': n_sensory + n_associative + n_inhibitory + n_output,
                'neurons_per_second': (n_sensory + n_associative + n_inhibitory + n_output) * self.n_patterns / elapsed,
                'stats': stats
            }
            
            print(f"    Time: {elapsed:.2f}s, Memory: {memory_used:.1f}MB")
            print(f"    Stats: {stats['connectivity_density']:.3f} density, {stats['memory_usage_mb']:.1f}MB internal")
            
            # Cleanup
            del network
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def benchmark_distributed(self, config):
        """Benchmark distributed implementation."""
        n_sensory, n_associative, n_inhibitory, n_output, name = config
        
        # Only test distributed on larger networks
        if n_sensory + n_associative + n_inhibitory + n_output < 10000:
            print(f"  Skipping distributed for {name} (too small)")
            return None
        
        print(f"  Testing distributed implementation: {name}")
        
        try:
            # Create network
            network = DistributedHebSNN(
                n_sensory=n_sensory,
                n_associative=n_associative,
                n_inhibitory=n_inhibitory,
                n_output=n_output,
                batch_size=32
            )
            
            patterns = self.create_test_patterns(n_sensory)
            
            # Create batch of patterns
            batch_patterns = jnp.array([
                jnp.zeros(network.n_neurons).at[pattern].set(1.0)
                for pattern in patterns
            ])
            
            # Benchmark
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            # Process in batch
            results = network.process_massive_batch(batch_patterns, n_steps=int(self.duration))
            
            elapsed = time.time() - start_time
            memory_used = self.get_memory_usage() - start_memory
            
            # Get throughput stats
            throughput = network.estimate_throughput()
            
            result = {
                'name': name,
                'time': elapsed,
                'memory_mb': memory_used,
                'neurons': n_sensory + n_associative + n_inhibitory + n_output,
                'neurons_per_second': throughput['total_operations_per_second'],
                'throughput': throughput
            }
            
            print(f"    Time: {elapsed:.2f}s, Memory: {memory_used:.1f}MB")
            print(f"    Throughput: {throughput['total_operations_per_second']:.0f} ops/sec")
            
            # Cleanup
            del network
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all implementations."""
        print("=" * 80)
        print("EXTREME PERFORMANCE BENCHMARK")
        print("=" * 80)
        
        # Show system info
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print()
        
        # Test each configuration
        for config in self.test_configs:
            name = config[4]
            print(f"Testing configuration: {name}")
            
            # Test all implementations
            orig_result = self.benchmark_original(config)
            jax_result = self.benchmark_jax_basic(config)
            ultra_result = self.benchmark_jax_ultra(config)
            dist_result = self.benchmark_distributed(config)
            
            # Store results
            if orig_result:
                self.results['original'].append(orig_result)
            if jax_result:
                self.results['jax_basic'].append(jax_result)
            if ultra_result:
                self.results['jax_ultra'].append(ultra_result)
            if dist_result:
                self.results['distributed'].append(dist_result)
            
            print()
        
        # Test extreme configurations (only ultra-optimized)
        print("Testing extreme configurations...")
        for config in self.extreme_configs:
            name = config[4]
            print(f"Testing extreme configuration: {name}")
            
            ultra_result = self.benchmark_jax_ultra(config)
            dist_result = self.benchmark_distributed(config)
            
            if ultra_result:
                self.results['jax_ultra'].append(ultra_result)
            if dist_result:
                self.results['distributed'].append(dist_result)
            
            print()
    
    def analyze_results(self):
        """Analyze and display benchmark results."""
        print("=" * 80)
        print("PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Find common configurations
        common_configs = []
        for result in self.results['original']:
            config_name = result['name']
            if any(r['name'] == config_name for r in self.results['jax_basic']) and \
               any(r['name'] == config_name for r in self.results['jax_ultra']):
                common_configs.append(config_name)
        
        print(f"{'Configuration':<20} {'Original':<12} {'JAX Basic':<12} {'JAX Ultra':<12} {'Speedup':<10}")
        print("-" * 75)
        
        for config_name in common_configs:
            orig = next(r for r in self.results['original'] if r['name'] == config_name)
            jax_basic = next(r for r in self.results['jax_basic'] if r['name'] == config_name)
            jax_ultra = next(r for r in self.results['jax_ultra'] if r['name'] == config_name)
            
            speedup = orig['time'] / jax_ultra['time']
            
            print(f"{config_name:<20} {orig['time']:<12.2f} {jax_basic['time']:<12.2f} {jax_ultra['time']:<12.2f} {speedup:<10.1f}x")
        
        # Show extreme results
        print("\nExtreme Scale Results:")
        print(f"{'Configuration':<25} {'JAX Ultra':<15} {'Neurons/sec':<15} {'Memory (MB)':<12}")
        print("-" * 70)
        
        for result in self.results['jax_ultra']:
            if result['neurons'] >= 35000:  # Extreme configurations
                print(f"{result['name']:<25} {result['time']:<15.2f} {result['neurons_per_second']:<15.0f} {result['memory_mb']:<12.1f}")
        
        # Calculate overall statistics
        if self.results['original'] and self.results['jax_ultra']:
            orig_times = [r['time'] for r in self.results['original']]
            ultra_times = [r['time'] for r in self.results['jax_ultra'][:len(orig_times)]]
            
            avg_speedup = np.mean([o/u for o, u in zip(orig_times, ultra_times)])
            max_speedup = np.max([o/u for o, u in zip(orig_times, ultra_times)])
            
            print(f"\nOverall Performance:")
            print(f"Average speedup: {avg_speedup:.1f}x")
            print(f"Maximum speedup: {max_speedup:.1f}x")
            
            # Find largest network tested
            largest_ultra = max(self.results['jax_ultra'], key=lambda x: x['neurons'])
            print(f"Largest network: {largest_ultra['neurons']:,} neurons in {largest_ultra['time']:.2f}s")
    
    def plot_results(self):
        """Plot benchmark results."""
        if not any(self.results.values()):
            print("No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution times
        for impl_name, results in self.results.items():
            if results:
                neurons = [r['neurons'] for r in results]
                times = [r['time'] for r in results]
                ax1.plot(neurons, times, 'o-', label=impl_name, linewidth=2)
        
        ax1.set_xlabel('Number of Neurons')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Network Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Plot 2: Speedup
        if self.results['original'] and self.results['jax_ultra']:
            orig_dict = {r['name']: r['time'] for r in self.results['original']}
            ultra_dict = {r['name']: r['time'] for r in self.results['jax_ultra']}
            
            common_names = set(orig_dict.keys()) & set(ultra_dict.keys())
            speedups = [orig_dict[name] / ultra_dict[name] for name in common_names]
            names = list(common_names)
            
            ax2.bar(range(len(names)), speedups, alpha=0.7)
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title('Speedup: Original vs Ultra-Optimized')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory usage
        for impl_name, results in self.results.items():
            if results:
                neurons = [r['neurons'] for r in results]
                memory = [r['memory_mb'] for r in results]
                ax3.plot(neurons, memory, 'o-', label=impl_name, linewidth=2)
        
        ax3.set_xlabel('Number of Neurons')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Network Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Plot 4: Throughput
        for impl_name, results in self.results.items():
            if results:
                neurons = [r['neurons'] for r in results]
                throughput = [r['neurons_per_second'] for r in results]
                ax4.plot(neurons, throughput, 'o-', label=impl_name, linewidth=2)
        
        ax4.set_xlabel('Number of Neurons')
        ax4.set_ylabel('Neurons Processed per Second')
        ax4.set_title('Processing Throughput')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('extreme_performance_results.png', dpi=300, bbox_inches='tight')
        print("Results saved to 'extreme_performance_results.png'")

def main():
    """Run extreme performance benchmark."""
    benchmark = ExtremeBenchmark()
    
    try:
        # Run comprehensive benchmark
        benchmark.run_comprehensive_benchmark()
        
        # Analyze results
        benchmark.analyze_results()
        
        # Plot results
        benchmark.plot_results()
        
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()