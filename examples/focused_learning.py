#!/usr/bin/env python3
"""
Focused Learning Demo - Shows clear association between input and output

This example demonstrates how the network can learn to associate
a specific input pattern with a consistent output response.
"""

import sys
import os
import numpy as np
import time
from tqdm import tqdm
import operator
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN

def get_top_neurons(firing_rates, top_n=5):
    """Get the indices of the top N firing neurons."""
    indices = np.argsort(firing_rates)[-top_n:]
    return [(i, firing_rates[i]) for i in reversed(indices) if firing_rates[i] > 0]

def main():
    """Demonstrate focused input pattern learning."""
    print("Creating network for focused pattern learning...")
    
    # Create a compact network
    network = HebSNN(
        n_sensory=10,
        n_associative=30,
        n_inhibitory=5,
        n_output=10,
        connectivity_density=0.3,  # Higher connectivity for faster learning
        seed=42
    )
    print(f"Network created with {network.n_neurons} neurons")
    
    # Define two different input patterns
    pattern_A = [0, 1, 2]  # First pattern - first 3 sensory neurons
    pattern_B = [7, 8, 9]  # Second pattern - last 3 sensory neurons
    
    print(f"Pattern A: Sensory neurons {pattern_A}")
    print(f"Pattern B: Sensory neurons {pattern_B}")
    
    # Strengthen the patterns for better signal propagation
    reinforced_A = []
    reinforced_B = []
    for idx in pattern_A:
        reinforced_A.extend([idx] * 5)  # Repeat each 5 times
    for idx in pattern_B:
        reinforced_B.extend([idx] * 5)  # Repeat each 5 times
    
    # Create arrays to track activity before/after training
    num_tests = 5
    pattern_A_responses_before = np.zeros((num_tests, network.n_output))
    pattern_B_responses_before = np.zeros((num_tests, network.n_output))
    pattern_A_responses_after = np.zeros((num_tests, network.n_output))
    pattern_B_responses_after = np.zeros((num_tests, network.n_output))
    
    # === Phase 1: Test responses before training ===
    print("\n--- Phase 1: Testing responses BEFORE training ---")
    output_start_idx = network.n_neurons - network.n_output
    
    for test in range(num_tests):
        # Test Pattern A
        network.reset()
        network.inject_spikes(reinforced_A)
        network.run(duration=50.0)
        
        # Get output activity
        activity = network.get_output_activity()
        for token_id, rate in activity.items():
            # token_id is the same as the index in our output array
            # (since output neurons were created with token_id = their index)
            token_id = int(token_id)  # Convert to int to use as index
            pattern_A_responses_before[test, token_id] = rate
        
        # Test Pattern B
        network.reset()
        network.inject_spikes(reinforced_B)
        network.run(duration=50.0)
        
        # Get output activity
        activity = network.get_output_activity()
        for token_id, rate in activity.items():
            # token_id is the same as the index in our output array
            token_id = int(token_id)  # Convert to int to use as index
            pattern_B_responses_before[test, token_id] = rate
    
    # Print initial response statistics
    print("\nPattern A initial responses:")
    avg_A_before = np.mean(pattern_A_responses_before, axis=0)
    top_A_before = get_top_neurons(avg_A_before, 3)
    for idx, rate in top_A_before:
        print(f"  Output neuron {idx}: {rate:.4f}")
    
    print("\nPattern B initial responses:")
    avg_B_before = np.mean(pattern_B_responses_before, axis=0)
    top_B_before = get_top_neurons(avg_B_before, 3)
    for idx, rate in top_B_before:
        print(f"  Output neuron {idx}: {rate:.4f}")
    
    # === Phase 2: Alternating training on both patterns ===
    print("\n--- Phase 2: Training on alternating patterns ---")
    training_repetitions = 40  # 20 for each pattern
    
    network.reset()
    print("Training on both patterns alternately...")
    
    with tqdm(total=training_repetitions, desc="Training Progress") as pbar:
        for rep in range(training_repetitions):
            # Alternate between patterns
            if rep % 2 == 0:
                # Force high modulation for active learning
                network.neuromodulation.novelty_signal = 0.8
                network.inject_spikes(reinforced_A)
            else:
                network.neuromodulation.novelty_signal = 0.8
                network.inject_spikes(reinforced_B)
                
            network.run(duration=50.0)
            pbar.update(1)
    
    print("Training completed.")
    
    # === Phase 3: Test responses after training ===
    print("\n--- Phase 3: Testing responses AFTER training ---")
    
    for test in range(num_tests):
        # Test Pattern A
        network.reset()
        network.inject_spikes(reinforced_A)
        network.run(duration=50.0)
        
        # Get output activity
        activity = network.get_output_activity()
        for token_id, rate in activity.items():
            token_id = int(token_id)  # Convert to int to use as index
            pattern_A_responses_after[test, token_id] = rate
        
        # Test Pattern B
        network.reset()
        network.inject_spikes(reinforced_B)
        network.run(duration=50.0)
        
        # Get output activity
        activity = network.get_output_activity()
        for token_id, rate in activity.items():
            token_id = int(token_id)  # Convert to int to use as index
            pattern_B_responses_after[test, token_id] = rate
    
    # Print response statistics after training
    print("\nPattern A responses after training:")
    avg_A_after = np.mean(pattern_A_responses_after, axis=0)
    top_A_after = get_top_neurons(avg_A_after, 3)
    for idx, rate in top_A_after:
        print(f"  Output neuron {idx}: {rate:.4f}")
    
    print("\nPattern B responses after training:")
    avg_B_after = np.mean(pattern_B_responses_after, axis=0)
    top_B_after = get_top_neurons(avg_B_after, 3)
    for idx, rate in top_B_after:
        print(f"  Output neuron {idx}: {rate:.4f}")
    
    # === Analysis: Check if patterns learned separate representations ===
    print("\n--- Learning Analysis ---")
    
    # Look at how much the outputs changed
    A_change = avg_A_after - avg_A_before
    B_change = avg_B_after - avg_B_before
    
    # Check if top neurons changed
    top_A_before_idx = [idx for idx, _ in top_A_before]
    top_A_after_idx = [idx for idx, _ in top_A_after]
    top_B_before_idx = [idx for idx, _ in top_B_before]
    top_B_after_idx = [idx for idx, _ in top_B_after]
    
    print(f"Pattern A top neurons before: {top_A_before_idx}")
    print(f"Pattern A top neurons after:  {top_A_after_idx}")
    print(f"Pattern B top neurons before: {top_B_before_idx}")
    print(f"Pattern B top neurons after:  {top_B_after_idx}")
    
    # Check if patterns are more easily distinguished after training
    A_selectivity_before = []
    B_selectivity_before = []
    A_selectivity_after = []
    B_selectivity_after = []
    
    for i in range(network.n_output):
        # A selectivity: how much more it responds to A than B
        A_selectivity_before.append(avg_A_before[i] - avg_B_before[i])
        A_selectivity_after.append(avg_A_after[i] - avg_B_after[i])
        
        # B selectivity: how much more it responds to B than A
        B_selectivity_before.append(avg_B_before[i] - avg_A_before[i])
        B_selectivity_after.append(avg_B_after[i] - avg_A_after[i])
    
    # Find neurons that became more selective for each pattern
    A_selective_before = np.where(np.array(A_selectivity_before) > 0.1)[0]
    A_selective_after = np.where(np.array(A_selectivity_after) > 0.1)[0]
    B_selective_before = np.where(np.array(B_selectivity_before) > 0.1)[0]
    B_selective_after = np.where(np.array(B_selectivity_after) > 0.1)[0]
    
    print(f"\nA-selective neurons before: {A_selective_before}")
    print(f"A-selective neurons after:  {A_selective_after}")
    print(f"B-selective neurons before: {B_selective_before}")
    print(f"B-selective neurons after:  {B_selective_after}")
    
    # Check for exclusivity - neurons that changed from one pattern to the other
    switched_to_A = np.setdiff1d(A_selective_after, A_selective_before)
    switched_to_B = np.setdiff1d(B_selective_after, B_selective_before)
    
    print(f"\nNeurons that became A-selective: {switched_to_A}")
    print(f"Neurons that became B-selective: {switched_to_B}")
    
    # Compute pattern similarity before and after
    dot_product_before = np.dot(avg_A_before, avg_B_before)
    dot_product_after = np.dot(avg_A_after, avg_B_after)
    
    norm_A_before = np.linalg.norm(avg_A_before)
    norm_B_before = np.linalg.norm(avg_B_before)
    norm_A_after = np.linalg.norm(avg_A_after)
    norm_B_after = np.linalg.norm(avg_B_after)
    
    if norm_A_before > 0 and norm_B_before > 0:
        similarity_before = dot_product_before / (norm_A_before * norm_B_before)
    else:
        similarity_before = 0
        
    if norm_A_after > 0 and norm_B_after > 0:
        similarity_after = dot_product_after / (norm_A_after * norm_B_after)
    else:
        similarity_after = 0
    
    print(f"\nPattern similarity before training: {similarity_before:.4f}")
    print(f"Pattern similarity after training: {similarity_after:.4f}")
    
    if similarity_after < similarity_before:
        print("\nThe network has learned more distinct representations for the two patterns!")
        percent_improvement = ((similarity_before - similarity_after) / similarity_before) * 100
        print(f"Pattern separation improved by {percent_improvement:.1f}%")
    else:
        print("\nPatterns became more similar - the network might need parameter tuning or more training.")

if __name__ == "__main__":
    main() 