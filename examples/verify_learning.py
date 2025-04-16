#!/usr/bin/env python3
"""
Script to verify the core learning mechanism works properly.

A focused test that directly verifies synaptic weights are changing during training.
"""

import sys
import os
import numpy as np
import time
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN

def main():
    """Run the verification experiment."""
    print("Creating minimal network for weight verification...")
    
    # Create small network
    network = HebSNN(
        n_sensory=5,
        n_associative=10,
        n_inhibitory=3,
        n_output=5,
        connectivity_density=0.5,  # Higher density to ensure connections exist
        seed=42
    )
    
    # Define specific neurons to test
    source_neuron = 0  # First sensory neuron
    target_neurons = [network.n_sensory + 1, network.n_neurons - 2]  # One associative, one output
    
    # Get initial weights
    initial_weights = {}
    for target in target_neurons:
        weight = network.connectivity.get_weight(source_neuron, target)
        if weight is not None:
            initial_weights[target] = weight
            print(f"Initial weight from {source_neuron} to {target}: {weight:.4f}")
    
    if not initial_weights:
        print("No direct connections found. Creating them...")
        for target in target_neurons:
            network.connectivity.add_synapse(source_neuron, target, weight=0.2)
            initial_weights[target] = 0.2
            print(f"Created connection from {source_neuron} to {target} with weight 0.2")
    
    # Now repeatedly stimulate the source neuron
    print("\nStimulating source neuron repeatedly...")
    weight_history = {target: [weight] for target, weight in initial_weights.items()}
    modulation_history = []
    
    # Run 10 training repetitions
    for i in tqdm(range(10)):
        # Force high modulation to amplify learning
        network.neuromodulation.novelty_signal = 0.8
        network.neuromodulation.reward_signal = 0.5
        modulation = network.neuromodulation.get_modulation()['total']
        modulation_history.append(modulation)
        
        # Inject 5 spikes to ensure strong activation
        network.inject_spikes([source_neuron] * 5)
        
        # Run a short simulation
        network.run(duration=10.0)
        
        # Record weights after this repetition
        for target in initial_weights.keys():
            current_weight = network.connectivity.get_weight(source_neuron, target)
            weight_history[target].append(current_weight)
    
    # Print results
    print("\nWeight changes over training:")
    for target, weights in weight_history.items():
        neuron_type = "Associative" if target < network.n_sensory + network.n_associative else "Output"
        print(f"\nConnection to {neuron_type} neuron {target}:")
        print("  Weights:", [f"{w:.4f}" for w in weights])
        
        if len(weights) > 1:
            total_change = weights[-1] - weights[0]
            percent_change = (total_change / weights[0]) * 100 if weights[0] > 0 else float('inf')
            print(f"  Total change: {total_change:.4f} ({percent_change:.1f}%)")
    
    print(f"\nAverage modulation level: {np.mean(modulation_history):.4f}")
    
    # Simple visualization of weight changes
    if weight_history:
        print("\nWeight change visualization:")
        for target, weights in weight_history.items():
            neuron_type = "Associative" if target < network.n_sensory + network.n_associative else "Output"
            print(f"Neuron {target} ({neuron_type}):")
            initial = weights[0]
            for i, w in enumerate(weights):
                bar_length = int(w * 40)  # Scale for display
                change = w - initial
                direction = "+" if change >= 0 else ""
                print(f"  Step {i}: {'█' * bar_length} {w:.4f} ({direction}{change:.4f})")

if __name__ == "__main__":
    main() 