#!/usr/bin/env python3
"""
Minimal demonstration of the Hebbian SNN without visualization.

This script provides a basic demonstration of the Hebbian SNN core
functionality without the overhead of real-time visualization.
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
    """Run minimal demonstration of Hebbian SNN."""
    print("Creating Hebbian SNN...")
    
    # Create a small network for demo purposes
    network = HebSNN(
        n_sensory=20,      # Smaller network for faster simulation
        n_associative=50,
        n_inhibitory=10,
        n_output=20,
        connectivity_density=0.2,  # Higher connectivity for better propagation
        seed=42
    )
    
    print(f"Network created with {network.n_neurons} neurons")
    print(f"- Sensory: {network.n_sensory}")
    print(f"- Associative: {network.n_associative}")
    print(f"- Inhibitory: {network.n_inhibitory}")
    print(f"- Output: {network.n_output}")
    
    # Inject a strong initial stimulus
    print("\nInjecting initial stimulus...")
    
    # Strong stimulus to the first 5 sensory neurons, each repeated 5 times
    neurons_to_stimulate = list(range(5))
    reinforced_stimulus = []
    for n in neurons_to_stimulate:
        reinforced_stimulus.extend([n] * 5)  # Repeat each 5 times
    
    network.inject_spikes(reinforced_stimulus)
    
    # Run for 100ms
    duration = 100.0
    
    print(f"\nRunning simulation for {duration}ms...")
    start_time = time.time()
    
    # Use debug=True to see detailed neuron activity
    debug = False
    
    # Enable debug output directly in this script
    if debug:
        # Uncomment debug printing in network.py
        for method_name, attr in HebSNN.__dict__.items():
            if hasattr(attr, '__code__') and method_name == '_update_neurons':
                # This is a method with a code object
                code = attr.__code__
                if '# Debug info' in str(code):
                    print("Debug mode enabled in _update_neurons")
    
    # Run simulation with progress bar
    spike_counts = {
        'sensory': 0,
        'associative': 0,
        'inhibitory': 0,
        'output': 0
    }
    
    with tqdm(total=int(duration), desc="Simulation Progress", unit="ms") as pbar:
        while network.current_time < duration:
            # Run for 1 ms
            current_time_before = network.current_time
            
            # Get neurons that spiked in this update
            spiking_neurons = network.step()
            
            # Count spikes by neuron type
            for n_idx in spiking_neurons:
                if n_idx < network.n_sensory:
                    spike_counts['sensory'] += 1
                elif n_idx < network.n_sensory + network.n_associative:
                    spike_counts['associative'] += 1
                elif n_idx < network.n_sensory + network.n_associative + network.n_inhibitory:
                    spike_counts['inhibitory'] += 1
                else:
                    spike_counts['output'] += 1
            
            # Update progress bar
            time_elapsed = network.current_time - current_time_before
            pbar.update(int(time_elapsed))
            
            # Print debug info about firing activity
            if debug and spiking_neurons:
                n_types = []
                for n in spiking_neurons:
                    if n < network.n_sensory:
                        n_types.append("S")
                    elif n < network.n_sensory + network.n_associative:
                        n_types.append("A")
                    elif n < network.n_sensory + network.n_associative + network.n_inhibitory:
                        n_types.append("I")
                    else:
                        n_types.append("O")
                        
                print(f"Time {network.current_time:.1f}ms: {len(spiking_neurons)} neurons fired - {n_types}")
    
    end_time = time.time()
    
    # Print final statistics
    total_spikes = sum(spike_counts.values())
    
    print(f"\nSimulation completed - {duration}ms in {end_time - start_time:.2f} seconds")
    print(f"Total spikes: {total_spikes}")
    print(f"Spike breakdown:")
    for pop, count in spike_counts.items():
        print(f"- {pop}: {count} spikes ({count/total_spikes*100:.1f}% of total)")
    
    # Print firing rates
    print(f"\nAverage firing rates:")
    ms_duration = duration
    
    sensory_rate = spike_counts['sensory'] / (network.n_sensory * ms_duration / 1000)
    assoc_rate = spike_counts['associative'] / (network.n_associative * ms_duration / 1000)
    inhib_rate = spike_counts['inhibitory'] / (network.n_inhibitory * ms_duration / 1000)
    output_rate = spike_counts['output'] / (network.n_output * ms_duration / 1000)
    
    print(f"- Sensory: {sensory_rate:.2f} Hz")
    print(f"- Associative: {assoc_rate:.2f} Hz")
    print(f"- Inhibitory: {inhib_rate:.2f} Hz")
    print(f"- Output: {output_rate:.2f} Hz")
    print(f"- Overall: {total_spikes / (network.n_neurons * ms_duration / 1000):.2f} Hz")


if __name__ == "__main__":
    main() 