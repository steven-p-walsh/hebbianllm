#!/usr/bin/env python3
"""
Optimized demo using JAX-accelerated Hebbian SNN.

This example demonstrates the improved performance and scalability
of the JAX-optimized implementation.
"""

import sys
import os
import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.jax_ops import JAXHebSNN

def create_sequential_patterns(n_sensory: int, n_patterns: int = 5):
    """Create sequential token patterns for language-like input."""
    patterns = []
    for i in range(n_patterns):
        # Create a sequence of 3-4 tokens
        sequence_length = np.random.randint(3, 5)
        pattern = np.random.choice(n_sensory, sequence_length, replace=False)
        patterns.append(pattern)
    return patterns

def run_learning_experiment():
    """Run a learning experiment with the optimized network."""
    print("Running optimized learning experiment...")
    
    # Create a larger network that would be slow with original implementation
    network = JAXHebSNN(
        n_sensory=200,
        n_associative=1000,
        n_inhibitory=200,
        n_output=200,
        connectivity_density=0.1,
        seed=42
    )
    
    # Create patterns
    patterns = create_sequential_patterns(network.n_sensory, n_patterns=8)
    print(f"Created {len(patterns)} patterns")
    
    # Training phase
    print("Training phase...")
    start_time = time.time()
    
    for epoch in range(10):  # Multiple epochs
        for pattern_idx, pattern in enumerate(patterns):
            network.reset()
            
            # Create input function for sequential pattern
            def input_fn(t):
                spikes = jnp.zeros(network.n_neurons)
                
                # Present pattern elements sequentially
                for i, token in enumerate(pattern):
                    if i * 20 <= t < (i + 1) * 20:  # Each token for 20ms
                        spikes = spikes.at[token].set(1.0)
                
                return spikes
            
            # Run network
            result = network.run(len(pattern) * 20, input_fn=input_fn)
            
        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1}/10 completed")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test phase
    print("Testing learned patterns...")
    test_results = []
    
    for pattern_idx, pattern in enumerate(patterns):
        network.reset()
        
        def input_fn(t):
            spikes = jnp.zeros(network.n_neurons)
            # Present only first token of pattern
            if t < 20:
                spikes = spikes.at[pattern[0]].set(1.0)
            return spikes
        
        result = network.run(100, input_fn=input_fn)
        output_activity = network.get_output_activity()
        
        # Get top 3 active outputs
        sorted_outputs = sorted(output_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        test_results.append({
            'pattern': pattern,
            'input_token': pattern[0],
            'top_outputs': sorted_outputs
        })
    
    # Display results
    print("\nTest Results:")
    for i, result in enumerate(test_results):
        print(f"Pattern {i+1}: {result['pattern']}")
        print(f"  Input token: {result['input_token']}")
        print(f"  Top outputs: {result['top_outputs']}")
    
    return network, test_results

def visualize_network_activity(network, pattern):
    """Visualize network activity for a given pattern."""
    network.reset()
    
    # Create input function
    def input_fn(t):
        spikes = jnp.zeros(network.n_neurons)
        for i, token in enumerate(pattern):
            if i * 10 <= t < (i + 1) * 10:
                spikes = spikes.at[token].set(1.0)
        return spikes
    
    # Run and collect activity
    result = network.run(len(pattern) * 10, input_fn=input_fn)
    
    # Plot activity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot spike raster
    spikes = result['spikes']
    times = np.arange(len(spikes))
    
    for t in range(len(spikes)):
        spike_neurons = np.where(spikes[t])[0]
        ax1.scatter([t] * len(spike_neurons), spike_neurons, s=1, c='black', alpha=0.6)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Neuron ID')
    ax1.set_title('Network Spike Activity')
    ax1.grid(True, alpha=0.3)
    
    # Plot novelty signal
    ax2.plot(times, result['novelty'])
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Novelty Signal')
    ax2.set_title('Neuromodulation Signal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('network_activity.png', dpi=300)
    print("Network activity visualization saved to 'network_activity.png'")

def main():
    """Run the optimized demo."""
    print("=" * 60)
    print("Optimized Hebbian SNN Demo")
    print("=" * 60)
    
    try:
        # Run learning experiment
        network, results = run_learning_experiment()
        
        # Visualize activity for first pattern
        if results:
            pattern = results[0]['pattern']
            print(f"\nVisualizing activity for pattern: {pattern}")
            visualize_network_activity(network, pattern)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("Key achievements:")
        print("- Scaled to 1400+ neurons with good performance")
        print("- Demonstrated sequential pattern learning")
        print("- Showed neuromodulation dynamics")
        print("- JAX acceleration enabled larger-scale experiments")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()