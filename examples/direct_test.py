#!/usr/bin/env python3
"""
Direct test script for debugging.

This script directly tests spike propagation from input to output.
"""

import sys
import os
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN

def main():
    """Run a direct test of the network."""
    print("Creating minimal test network...")
    
    # Create a very small network
    network = HebSNN(
        n_sensory=5,
        n_associative=10,
        n_inhibitory=3,
        n_output=5,
        connectivity_density=0.5,  # High connectivity for testing
        seed=42
    )
    
    # Index ranges
    sensory_range = (0, network.n_sensory)
    assoc_range = (network.n_sensory, network.n_sensory + network.n_associative)
    inhib_range = (assoc_range[1], assoc_range[1] + network.n_inhibitory) 
    output_range = (inhib_range[1], inhib_range[1] + network.n_output)
    
    print(f"Network created with {network.n_neurons} neurons")
    print(f"Neuron ranges:")
    print(f"  Sensory:      {sensory_range[0]}-{sensory_range[1]-1}")
    print(f"  Associative:  {assoc_range[0]}-{assoc_range[1]-1}")
    print(f"  Inhibitory:   {inhib_range[0]}-{inhib_range[1]-1}")
    print(f"  Output:       {output_range[0]}-{output_range[1]-1}")
    
    # Print some connectivity statistics
    print("\nConnectivity checks:")
    
    # Check for direct sensory -> output connections
    direct_connections = []
    for s_idx in range(sensory_range[0], sensory_range[1]):
        for o_idx in range(output_range[0], output_range[1]):
            weight = network.connectivity.get_weight(s_idx, o_idx)
            if weight is not None:
                direct_connections.append((s_idx, o_idx, weight))
    
    print(f"Direct sensory->output connections: {len(direct_connections)}")
    for s, o, w in direct_connections[:5]:  # Show first 5
        print(f"  Sensory {s} → Output {o}: Weight = {w:.3f}")
    
    # Print connection counts
    connection_counts = {
        'sensory→associative': 0,
        'associative→associative': 0,
        'associative→output': 0,
        'inhibitory→output': 0
    }
    
    # Sample some neurons and count connections
    for s_idx in range(sensory_range[0], sensory_range[1]):
        for a_idx in range(assoc_range[0], assoc_range[1]):
            if network.connectivity.get_weight(s_idx, a_idx) is not None:
                connection_counts['sensory→associative'] += 1
    
    for a1_idx in range(assoc_range[0], assoc_range[1]):
        for a2_idx in range(assoc_range[0], assoc_range[1]):
            if a1_idx != a2_idx and network.connectivity.get_weight(a1_idx, a2_idx) is not None:
                connection_counts['associative→associative'] += 1
    
    for a_idx in range(assoc_range[0], assoc_range[1]):
        for o_idx in range(output_range[0], output_range[1]):
            if network.connectivity.get_weight(a_idx, o_idx) is not None:
                connection_counts['associative→output'] += 1
    
    for i_idx in range(inhib_range[0], inhib_range[1]):
        for o_idx in range(output_range[0], output_range[1]):
            if network.connectivity.get_weight(i_idx, o_idx) is not None:
                connection_counts['inhibitory→output'] += 1
    
    print("\nConnection counts:")
    for path, count in connection_counts.items():
        print(f"  {path}: {count}")
    
    # Test simple spike propagation
    print("\nTesting spike propagation:")
    
    # Inject a strong stimulus to the first sensory neuron
    stimulated_neuron = 0
    strong_stim = [stimulated_neuron] * 20  # Repeated 20 times
    
    print(f"Injecting 20 spikes to sensory neuron {stimulated_neuron}...")
    network.reset()
    network.inject_spikes(strong_stim)
    
    # Run for short duration with debugging
    print("Running simulation...")
    
    # Inject neuromodulation to enhance activity
    network.neuromodulation.novelty_signal = 1.0
    
    # Record activity at multiple timepoints
    timepoints = [10, 20, 50, 100]
    for duration in timepoints:
        network.run(duration - network.current_time)  # Run until this timepoint
        
        # Count active neurons
        active_sensory = []
        active_assoc = []
        active_inhib = [] 
        active_output = []
        
        # Get membrane potentials to see neurons close to firing
        near_threshold = []
        
        for i, neuron in enumerate(network.neurons):
            if neuron.last_spike_time >= 0 and neuron.last_spike_time >= network.current_time - 10:
                # This neuron has spiked in the last 10ms
                if i < sensory_range[1]:
                    active_sensory.append(i)
                elif i < assoc_range[1]:
                    active_assoc.append(i)
                elif i < inhib_range[1]:
                    active_inhib.append(i)
                else:
                    active_output.append(i)
            
            # Check neurons that are close to threshold
            threshold_ratio = neuron.membrane_potential / neuron.threshold
            if threshold_ratio > 0.5 and not neuron.is_refractory:
                near_threshold.append((i, threshold_ratio, neuron.membrane_potential, neuron.threshold))
        
        print(f"\nAt t={duration}ms:")
        print(f"  Active sensory neurons: {active_sensory}")
        print(f"  Active associative neurons: {active_assoc}")
        print(f"  Active inhibitory neurons: {active_inhib}")
        print(f"  Active output neurons: {active_output}")
        
        # Print detailed output neuron activity
        output_activity = network.get_output_activity()
        if output_activity:
            print("\n  Output neuron activity:")
            for token_id, rate in sorted(output_activity.items(), key=lambda x: x[1], reverse=True):
                print(f"    Output neuron {token_id}: {rate:.4f}")
        else:
            print("\n  No output activity detected in get_output_activity()")
        
        # Print near-threshold neurons
        if near_threshold:
            print("\n  Neurons close to threshold:")
            for idx, ratio, potential, threshold in sorted(near_threshold, key=lambda x: x[1], reverse=True)[:5]:
                neuron_type = "Unknown"
                if idx < sensory_range[1]:
                    neuron_type = "Sensory"
                elif idx < assoc_range[1]:
                    neuron_type = "Associative"
                elif idx < inhib_range[1]:
                    neuron_type = "Inhibitory"
                else:
                    neuron_type = "Output"
                    
                print(f"    {neuron_type} {idx}: potential={potential:.3f}, threshold={threshold:.3f}, ratio={ratio:.2f}")

if __name__ == "__main__":
    main() 