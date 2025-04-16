#!/usr/bin/env python3
"""
Scalability demo for Hebbian SNN.

This script tests the network's learning capabilities at multiple scales
with JAX acceleration on CPU.
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import gc

# Configure JAX to use CPU only
jax.config.update('jax_platform_name', 'cpu')
print("JAX configured to use CPU backend")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN

class ScalabilityTest:
    """Test suite for scalability experiments."""
    
    def __init__(self):
        # Define network sizes to test
        self.network_sizes = [
            (10, 50, 10, 10),     # Small network
            (50, 200, 30, 50),    # Medium network
            (100, 500, 50, 100),  # Large network
            # Uncomment for extreme tests with sufficient memory:
            # (200, 1000, 100, 200),  # XL network
            # (500, 2000, 200, 500),  # XXL network
        ]
        
        # Number of patterns to learn
        self.n_patterns = 5
        self.pattern_size = 3
        
        # Training parameters
        self.training_repetitions = 50
        self.run_duration = 50.0
        
        # Results storage
        self.results = []
    
    def generate_patterns(self, n_sensory, n_patterns, pattern_size):
        """Generate random non-overlapping patterns."""
        patterns = []
        all_indices = set(range(n_sensory))
        
        for _ in range(n_patterns):
            # Pick random neurons that aren't in previous patterns
            available = list(all_indices - set(sum(patterns, [])))
            
            if len(available) < pattern_size:
                break
                
            pattern = sorted(np.random.choice(
                available, pattern_size, replace=False))
            patterns.append(pattern)
        
        return patterns
    
    def reinforce_pattern(self, pattern, repetitions=5):
        """Reinforce pattern with repetition for stronger stimulation."""
        reinforced = []
        for idx in pattern:
            reinforced.extend([idx] * repetitions)
        return reinforced
    
    def test_network(self, network, patterns, reinforced_patterns):
        """Test network responses to each pattern."""
        response_matrix = np.zeros((len(patterns), network.n_output))
        
        for i, pattern in enumerate(reinforced_patterns):
            network.reset()
            network.inject_spikes(pattern)
            network.run(duration=self.run_duration)
            
            # Get output activity
            activity = network.get_output_activity()
            for token_id, rate in activity.items():
                token_id = int(token_id)
                response_matrix[i, token_id] = rate
        
        return response_matrix
    
    def calculate_separation(self, responses):
        """Calculate average pattern separation metric."""
        if len(responses) <= 1:
            return 0.0
        
        separation_scores = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                # Calculate cosine similarity between responses
                cosine_sim = np.dot(responses[i], responses[j]) / (
                    np.linalg.norm(responses[i]) * np.linalg.norm(responses[j]))
                # Convert to distance (1-similarity)
                distance = 1.0 - cosine_sim
                separation_scores.append(distance)
        
        return np.mean(separation_scores)
    
    def estimate_memory_usage(self, network):
        """Estimate memory usage of the network in MB."""
        # Calculate approximate size of weights matrices
        n_sensory = network.n_sensory
        n_associative = network.n_associative
        n_inhibitory = network.n_inhibitory
        n_output = network.n_output
        
        total_params = 0
        # Sensory to associative
        total_params += n_sensory * n_associative
        # Associative to associative 
        total_params += n_associative * n_associative
        # Associative to inhibitory
        total_params += n_associative * n_inhibitory
        # Inhibitory to associative
        total_params += n_inhibitory * n_associative
        # Associative to output
        total_params += n_associative * n_output
        
        # Assume float32 (4 bytes per parameter)
        memory_usage = total_params * 4 / (1024 * 1024)  # Convert to MB
        return memory_usage
    
    def plot_results(self):
        """Plot and save performance metrics."""
        network_labels = ["Small", "Medium", "Large", "XL", "XXL"][:len(self.results)]
        network_sizes = [f"{r['n_sensory']}+{r['n_associative']}+{r['n_output']}" 
                        for r in self.results]
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Hebbian SNN Scalability Results', fontsize=16)
        
        # Plot 1: Improvement in pattern separation
        improvements = [r['after_separation'] - r['before_separation'] for r in self.results]
        axs[0, 0].bar(network_labels, improvements)
        axs[0, 0].set_title('Improvement in Pattern Separation')
        axs[0, 0].set_ylabel('Separation Improvement')
        
        # Plot 2: Training time
        train_times = [r['avg_training_time'] for r in self.results]
        axs[0, 1].bar(network_labels, train_times)
        axs[0, 1].set_title('Average Training Time per Pattern')
        axs[0, 1].set_ylabel('Time (seconds)')
        
        # Plot 3: Memory usage
        memory_usage = [r['memory_usage'] for r in self.results]
        axs[1, 0].bar(network_labels, memory_usage)
        axs[1, 0].set_title('Estimated Memory Usage')
        axs[1, 0].set_ylabel('Memory (MB)')
        
        # Plot 4: Number of neurons
        neuron_counts = [r['n_sensory'] + r['n_associative'] + r['n_inhibitory'] + r['n_output'] 
                         for r in self.results]
        axs[1, 1].bar(network_labels, neuron_counts)
        axs[1, 1].set_title('Total Neuron Count')
        axs[1, 1].set_ylabel('Number of Neurons')
        
        plt.tight_layout()
        plt.savefig('scalability_results.png')
        print("Results saved to 'scalability_results.png'")

    def run_tests(self):
        """Run all scalability tests."""
        for size_idx, (n_sensory, n_associative, n_inhibitory, n_output) in enumerate(self.network_sizes):
            network_name = ["Small", "Medium", "Large", "XL", "XXL"][size_idx]
            print(f"\n=== Testing {network_name} Network ===")
            print(f"Config: {n_sensory} sensory, {n_associative} associative, {n_inhibitory} inhibitory, {n_output} output")
            
            # Initialize network
            network = HebSNN(
                n_sensory=n_sensory,
                n_associative=n_associative,
                n_inhibitory=n_inhibitory,
                n_output=n_output,
                connectivity_density=0.2,  # Lower density for larger networks
                seed=42
            )
            
            # Generate patterns
            patterns = self.generate_patterns(
                n_sensory, self.n_patterns, self.pattern_size)
            print(f"Generated {len(patterns)} unique patterns")
            
            # Reinforce patterns
            reinforced_patterns = [self.reinforce_pattern(p) for p in patterns]
            
            # Test before training
            print("Testing response patterns before training...")
            before_responses = self.test_network(
                network, patterns, reinforced_patterns)
            before_separation = self.calculate_separation(before_responses)
            print(f"Pattern separation before training: {before_separation:.4f}")
            
            # Benchmark training time
            print("Training network...")
            training_durations = []
            total_start = time.time()
            
            with tqdm(total=self.training_repetitions) as pbar:
                for _ in range(self.training_repetitions):
                    # Train on each pattern in sequence
                    for pattern in reinforced_patterns:
                        network.reset()
                        network.neuromodulation.novelty_signal = 0.8  # Boost learning
                        
                        # Time this specific pattern's training
                        start_time = time.time()
                        network.inject_spikes(pattern)
                        network.run(duration=self.run_duration)
                        elapsed = time.time() - start_time
                        
                        training_durations.append(elapsed)
                    pbar.update(1)
            
            total_time = time.time() - total_start
            avg_duration = np.mean(training_durations)
            print(f"Average training time per pattern: {avg_duration:.4f}s")
            print(f"Total training time: {total_time:.2f}s")
            
            # Test after training
            print("Testing response patterns after training...")
            after_responses = self.test_network(
                network, patterns, reinforced_patterns)
            after_separation = self.calculate_separation(after_responses)
            print(f"Pattern separation after training: {after_separation:.4f}")
            
            # Calculate improvement
            improvement = after_separation - before_separation
            percent_improvement = (improvement / max(0.0001, before_separation)) * 100
            print(f"Separation improvement: {improvement:.4f} ({percent_improvement:.1f}%)")
            
            # Record memory usage
            memory_usage = self.estimate_memory_usage(network)
            print(f"Estimated memory usage: {memory_usage:.2f} MB")
            
            # Store results
            self.results.append({
                'n_sensory': n_sensory,
                'n_associative': n_associative,
                'n_inhibitory': n_inhibitory,
                'n_output': n_output,
                'before_separation': before_separation,
                'after_separation': after_separation,
                'avg_training_time': avg_duration,
                'memory_usage': memory_usage
            })
            
            # Give some time for garbage collection before the next test
            gc.collect()
            time.sleep(1)
            
        # Plot the results
        self.plot_results()

def main():
    """Run scalability tests."""
    print("Starting Hebbian SNN Scalability Tests with JAX optimization")
    
    # Create and run test suite
    tester = ScalabilityTest()
    tester.run_tests()

if __name__ == "__main__":
    main() 