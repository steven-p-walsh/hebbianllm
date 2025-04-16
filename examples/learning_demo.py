#!/usr/bin/env python3
"""
Learning demonstration for the Hebbian SNN.

This script demonstrates the network's ability to learn associations
between an input pattern and output neurons through STDP.
"""

import sys
import os
import numpy as np
import time
from tqdm import tqdm
import operator
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN

# --- Helper Functions --- 
def get_top_output_activity(network: HebSNN, top_n: int = 5) -> List[Tuple[int, float]]:
    """Gets the top N active output neurons and their rates."""
    output_activity = network.get_output_activity()
    if not output_activity:
        return []
    
    # Sort by firing rate (descending)
    sorted_activity = sorted(output_activity.items(), key=operator.itemgetter(1), reverse=True)
    
    # Return top N
    return [(int(token_id), float(rate)) for token_id, rate in sorted_activity[:top_n] if rate > 0.001]

def get_pathway_weights(network: HebSNN, sensory_indices: List[int], output_indices: List[int]) -> List[float]:
    """Get average weights of pathways from sensory to output neurons."""
    # This is a simple approximation - just looks at direct connections
    # In reality, the pathways involve associative neurons as well
    pathway_weights = []
    
    for s_idx in sensory_indices:
        for o_idx in output_indices:
            weight = network.connectivity.get_weight(s_idx, o_idx)
            if weight is not None:
                pathway_weights.append(weight)
    
    # Also check for key associative neuron connections
    # (simplistic approach: neurons in the middle layer of indices)
    assoc_start = network.n_sensory
    assoc_end = assoc_start + network.n_associative
    assoc_neurons = list(range(assoc_start, assoc_end, 5))[:5]  # Sample a few
    
    # Sensory -> Associative connections
    for s_idx in sensory_indices:
        for a_idx in assoc_neurons:
            weight = network.connectivity.get_weight(s_idx, a_idx)
            if weight is not None:
                pathway_weights.append(weight)
    
    # Associative -> Output connections
    for a_idx in assoc_neurons:
        for o_idx in output_indices:
            weight = network.connectivity.get_weight(a_idx, o_idx)
            if weight is not None:
                pathway_weights.append(weight)
    
    return pathway_weights

# --- Main Demo --- 
def main():
    """Run learning demonstration."""
    print("Creating Hebbian SNN for Learning Demo...")
    
    # Network parameters - relatively small for speed
    network_params = {
        'n_sensory': 20,
        'n_associative': 50,
        'n_inhibitory': 10,
        'n_output': 20,
        'connectivity_density': 0.25, # Slightly higher density might help learning
        'seed': 42
    }
    network = HebSNN(**network_params)
    
    print(f"Network created with {network.n_neurons} neurons.")

    # --- Define Input Pattern --- 
    pattern_a_indices = [2, 3, 4]  # Sensory neurons representing 'Pattern A'
    print(f"\nInput Pattern A: Stimulating sensory neurons {pattern_a_indices}")
    
    # Define neurons to track in our pathway analysis
    output_neurons_to_track = list(range(70, 90))  # All output neurons (indices 70-89)
    
    # Reinforce the pattern for stronger initial stimulation
    reinforced_pattern_a = []
    for idx in pattern_a_indices:
        reinforced_pattern_a.extend([idx] * 5) # Repeat each 5 times

    # --- Phase 1: Test Before Training --- 
    print("\n--- Phase 1: Testing response BEFORE training ---")
    network.reset() # Start from a clean slate
    network.inject_spikes(reinforced_pattern_a)
    network.run(duration=50.0) # Run for a short period to see initial response
    
    activity_before = get_top_output_activity(network)
    print("Output Neuron Activity BEFORE Training:")
    if activity_before:
        for token_id, rate in activity_before:
            print(f"  Output Neuron {token_id}: Rate {rate:.4f}")
    else:
        print("  No significant output activity.")
    
    # Get baseline weights
    pathway_weights_before = get_pathway_weights(network, pattern_a_indices, [n[0] for n in activity_before[:3]])
    print(f"\nPathway Weights BEFORE Training:")
    print(f"  Average: {np.mean(pathway_weights_before):.4f}")
    print(f"  Count: {len(pathway_weights_before)} connections")
    print(f"  Min: {min(pathway_weights_before):.4f}, Max: {max(pathway_weights_before):.4f}")

    # --- Phase 2: Training --- 
    print("\n--- Phase 2: Training on Pattern A ---")
    network.reset() # Reset again to start training fairly
    training_repetitions = 50  # Increased from 20 to 50
    run_duration_per_rep = 50.0 # ms
    total_training_duration = training_repetitions * run_duration_per_rep
    
    # Track neuromodulation during training
    modulation_values = []
    
    start_time = time.time()
    with tqdm(total=training_repetitions, desc="Training Progress") as pbar:
        for rep in range(training_repetitions):
            # Force higher neuromodulation for learning during training
            if rep < 10:
                # Boost early learning with higher novelty
                network.neuromodulation.novelty_signal = 0.5
                network.neuromodulation.reward_signal = 0.3
            
            network.inject_spikes(reinforced_pattern_a)
            
            # Capture modulation at this step
            mod_info = network.neuromodulation.get_modulation()
            modulation_values.append(mod_info['total'])
            
            network.run(duration=run_duration_per_rep)
            pbar.update(1)
            
    end_time = time.time()
    print(f"Training completed. ({training_repetitions} reps, {total_training_duration}ms sim time) in {end_time - start_time:.2f}s")
    print(f"Average modulation during training: {np.mean(modulation_values):.4f}")

    # --- Phase 3: Test After Training --- 
    print("\n--- Phase 3: Testing response AFTER training ---")
    # DO NOT reset the network here - we want to test the trained state
    network.inject_spikes(reinforced_pattern_a)
    network.run(duration=50.0) # Run for the same short period as before training
    
    activity_after = get_top_output_activity(network)
    print("Output Neuron Activity AFTER Training:")
    if activity_after:
        for token_id, rate in activity_after:
            print(f"  Output Neuron {token_id}: Rate {rate:.4f}")
    else:
        print("  No significant output activity.")
    
    # Get final weights
    pathway_weights_after = get_pathway_weights(network, pattern_a_indices, [n[0] for n in activity_after[:3]])
    print(f"\nPathway Weights AFTER Training:")
    print(f"  Average: {np.mean(pathway_weights_after):.4f}")
    print(f"  Count: {len(pathway_weights_after)} connections")
    print(f"  Min: {min(pathway_weights_after):.4f}, Max: {max(pathway_weights_after):.4f}")
    
    # Calculate weight change
    if pathway_weights_before and pathway_weights_after:
        weight_change = np.mean(pathway_weights_after) - np.mean(pathway_weights_before)
        print(f"\nAverage weight change: {weight_change:.4f} ({weight_change/np.mean(pathway_weights_before)*100:.2f}%)")

    # --- Comparison --- 
    print("\n--- Comparison --- ")
    print("Activity Before Training:", activity_before)
    print("Activity After Training:", activity_after)
    
    # Look for neuron-specific increases
    common_neurons = []
    for neuron_after, rate_after in activity_after:
        for neuron_before, rate_before in activity_before:
            if neuron_after == neuron_before:
                common_neurons.append((neuron_after, rate_before, rate_after))
                break
    
    if common_neurons:
        print("\nCommon neurons (before -> after):")
        for neuron, rate_before, rate_after in common_neurons:
            change = rate_after - rate_before
            print(f"  Neuron {neuron}: {rate_before:.4f} -> {rate_after:.4f} (Change: {change:.4f}, {change/rate_before*100:.1f}%)")
    
    # Basic check for learning
    if activity_after and (not activity_before or activity_after[0][1] > activity_before[0][1] * 1.2):
         # Check if top neuron after is significantly stronger than top neuron before
        print("\nObservation: Network appears to have learned/strengthened an association for Pattern A.")
        if activity_before and activity_after[0][0] == activity_before[0][0]:
             print(f"  (Top output neuron {activity_after[0][0]} remained the same but strengthened)")
        elif activity_before:
             print(f"  (Top output neuron changed from {activity_before[0][0]} to {activity_after[0][0]})")
    elif activity_after and activity_before and activity_after[0][1] <= activity_before[0][1]:
        print("\nObservation: Output activity did not significantly increase. Learning may be weak or parameters need tuning.")
    else:
        print("\nObservation: No significant output activity detected either before or after training.")

if __name__ == "__main__":
    main() 