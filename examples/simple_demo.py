#!/usr/bin/env python3
"""
Simple demonstration of the Hebbian SNN.

This script provides a basic demonstration of the Hebbian SNN 
with visualization of network activity.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbianllm.core.network import HebSNN
from hebbianllm.visualization.activity_monitor import ActivityMonitor


def main():
    """Run simple demonstration of Hebbian SNN."""
    print("Creating Hebbian SNN...")
    
    # Create a small network for demo purposes
    network = HebSNN(
        n_sensory=50,
        n_associative=200,
        n_inhibitory=50,
        n_output=50,
        connectivity_density=0.1,
        seed=42
    )
    
    # Create activity monitor
    monitor = ActivityMonitor(network)
    
    print(f"Network created with {network.n_neurons} neurons")
    print(f"- Sensory: {network.n_sensory}")
    print(f"- Associative: {network.n_associative}")
    print(f"- Inhibitory: {network.n_inhibitory}")
    print(f"- Output: {network.n_output}")
    
    # Define a simple input pattern
    def input_pattern(t):
        """Generate input pattern based on time."""
        # Cycle through first 10 sensory neurons
        if int(t) % 20 == 0:
            # Create a larger cluster of activity
            center = (int(t) // 20) % 10
            # Stimulate 5 neighboring neurons instead of 3
            neuron_ids = [(center + i) % network.n_sensory for i in range(-2, 3)]
            
            # Repeat each neuron 5 times to give stronger stimulation
            # This simulates multiple spikes arriving close together
            reinforced_ids = []
            for neuron_id in neuron_ids:
                reinforced_ids.extend([neuron_id] * 5)
            
            return reinforced_ids
        return None
    
    # Run the network with visualization
    print("Running network simulation...")
    
    # Setup plotting
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax_raster = fig.add_subplot(gs[0, :])
    ax_rates = fig.add_subplot(gs[1, 0])
    ax_output = fig.add_subplot(gs[1, 1])
    
    # Run for 500 ms
    duration = 100.0
    update_interval = 50.0  # Update visualization every 50 ms
    next_update = update_interval
    
    start_time = time.time()
    
    # Create a tqdm progress bar for the simulation
    with tqdm(total=int(duration), desc="Simulation Progress", unit="ms") as pbar:
        while network.current_time < duration:
            # Get input
            input_data = input_pattern(network.current_time)
            if input_data:
                network.inject_spikes(input_data)
            
            # Run for 1 ms
            current_time_before = network.current_time
            network.run(1.0)
            
            # Update progress bar with actual time elapsed in simulation
            time_elapsed = network.current_time - current_time_before
            pbar.update(int(time_elapsed))
            
            # Update monitor
            monitor.update(network.current_time)
            
            # Update plots periodically
            if network.current_time >= next_update:
                # Clear plots
                ax_raster.clear()
                ax_rates.clear()
                ax_output.clear()
                
                # Update plots
                monitor.plot_raster(ax=ax_raster)
                monitor.plot_firing_rates(ax=ax_rates)
                
                # Get output activity
                output_activities = monitor.get_active_output_neurons()
                token_ids = [str(x[0]) for x in output_activities[:10]]  # Top 10
                activities = [x[1] for x in output_activities[:10]]
                
                if token_ids and activities:
                    ax_output.bar(token_ids, activities)
                ax_output.set_title('Top Active Output Neurons')
                ax_output.set_xlabel('Token ID')
                ax_output.set_ylabel('Firing Rate')
                
                # Update title with time
                fig.suptitle(f"Hebbian SNN Simulation - Time: {network.current_time:.1f} ms")
                
                # Display
                fig.tight_layout()
                plt.pause(0.01)
                
                next_update = network.current_time + update_interval
    
    end_time = time.time()
    
    # Print final status
    print(f"\nSimulation completed - {duration} ms in {end_time - start_time:.2f} seconds")
    print(f"Population firing rates:")
    rates = monitor.get_population_rates()
    for pop, rate in rates.items():
        print(f"- {pop}: {rate*100:.2f}%")
    
    # Save final state as image
    plt.savefig("hebbian_snn_demo.png")
    
    # Let the user close the window
    print("Press any key to exit...")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main() 