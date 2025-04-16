"""
Activity monitoring and visualization tools.

This module provides tools for monitoring and visualizing 
the activity of the Hebbian SNN during simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional, Any, Union
import time

from hebbianllm.core.network import HebSNN


class ActivityMonitor:
    """Monitor and visualize SNN activity."""
    
    def __init__(self, network: HebSNN, buffer_size: int = 1000):
        """
        Initialize the activity monitor.
        
        Args:
            network: The SNN to monitor
            buffer_size: Size of the activity buffer to maintain
        """
        self.network = network
        self.buffer_size = buffer_size
        
        # Neuron populations
        self.n_sensory = network.n_sensory
        self.n_associative = network.n_associative
        self.n_inhibitory = network.n_inhibitory
        self.n_output = network.n_output
        
        # Activity buffer
        self.activity_buffer = np.zeros((network.n_neurons, buffer_size), dtype=bool)
        self.time_buffer = np.zeros(buffer_size)
        self.current_idx = 0
        
        # Statistics
        self.firing_rates = np.zeros(network.n_neurons)
        self.last_update_time = time.time()
    
    def update(self, current_time: float):
        """
        Update the monitor with the current network state.
        
        Args:
            current_time: Current simulation time
        """
        # Get current activity
        current_activity = np.zeros(self.network.n_neurons, dtype=bool)
        
        # Check if any events in the queue are scheduled for now
        while (not self.network.spike_queue.empty() and 
               self.network.spike_queue.queue[0].time <= current_time):
            event = self.network.spike_queue.queue[0]
            current_activity[event.neuron_idx] = True
        
        # Update activity buffer
        buffer_idx = self.current_idx % self.buffer_size
        self.activity_buffer[:, buffer_idx] = current_activity
        self.time_buffer[buffer_idx] = current_time
        self.current_idx += 1
        
        # Update statistics
        if time.time() - self.last_update_time > 0.1:  # Update stats every 100ms
            self._update_statistics()
            self.last_update_time = time.time()
    
    def _update_statistics(self):
        """Update network statistics."""
        # Calculate firing rates over recent window
        window_size = min(100, self.buffer_size, self.current_idx)
        start_idx = max(0, self.current_idx - window_size)
        
        # Extract activity in window
        if self.current_idx > start_idx:
            # Convert indices to buffer positions
            start_pos = start_idx % self.buffer_size
            current_pos = self.current_idx % self.buffer_size
            
            if current_pos > start_pos:
                window_activity = self.activity_buffer[:, start_pos:current_pos]
            else:
                # Handle wrap-around
                activity1 = self.activity_buffer[:, start_pos:]
                activity2 = self.activity_buffer[:, :current_pos]
                window_activity = np.concatenate([activity1, activity2], axis=1)
            
            # Calculate firing rates (spikes per ms)
            self.firing_rates = np.mean(window_activity, axis=1)
    
    def get_population_rates(self) -> Dict[str, float]:
        """
        Get average firing rates for each population.
        
        Returns:
            Dictionary with population rates
        """
        sensory_rate = np.mean(self.firing_rates[:self.n_sensory])
        
        associative_start = self.n_sensory
        associative_end = associative_start + self.n_associative
        associative_rate = np.mean(self.firing_rates[associative_start:associative_end])
        
        inhibitory_start = associative_end
        inhibitory_end = inhibitory_start + self.n_inhibitory
        inhibitory_rate = np.mean(self.firing_rates[inhibitory_start:inhibitory_end])
        
        output_start = inhibitory_end
        output_rate = np.mean(self.firing_rates[output_start:])
        
        return {
            'sensory': sensory_rate,
            'associative': associative_rate,
            'inhibitory': inhibitory_rate,
            'output': output_rate,
            'overall': np.mean(self.firing_rates)
        }
    
    def get_active_output_neurons(self, threshold: float = 0.01) -> List[Tuple[int, float]]:
        """
        Get list of active output neurons above threshold.
        
        Args:
            threshold: Activity threshold
            
        Returns:
            List of (token_id, firing_rate) tuples
        """
        output_start = self.n_sensory + self.n_associative + self.n_inhibitory
        output_activity = []
        
        for i in range(output_start, self.network.n_neurons):
            rate = self.firing_rates[i]
            if rate > threshold:
                token_id = self.network.neurons[i].token_id
                output_activity.append((token_id, rate))
        
        # Sort by activity (highest first)
        output_activity.sort(key=lambda x: x[1], reverse=True)
        
        return output_activity
    
    def plot_firing_rates(self, ax=None):
        """
        Plot firing rates of different neuron populations.
        
        Args:
            ax: Matplotlib axis to plot on (creates new if None)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        populations = ['Sensory', 'Associative', 'Inhibitory', 'Output']
        
        start_indices = [
            0,
            self.n_sensory,
            self.n_sensory + self.n_associative,
            self.n_sensory + self.n_associative + self.n_inhibitory
        ]
        
        end_indices = [
            self.n_sensory,
            self.n_sensory + self.n_associative,
            self.n_sensory + self.n_associative + self.n_inhibitory,
            self.network.n_neurons
        ]
        
        pop_rates = []
        for start, end in zip(start_indices, end_indices):
            pop_rates.append(np.mean(self.firing_rates[start:end]))
        
        # Create bar plot
        ax.bar(populations, pop_rates)
        ax.set_ylabel('Firing Rate (spikes/ms)')
        ax.set_title('Population Firing Rates')
        
        # Add overall average line
        overall_rate = np.mean(self.firing_rates)
        ax.axhline(overall_rate, color='r', linestyle='--', label=f'Overall: {overall_rate:.4f}')
        ax.legend()
        
        return ax
    
    def plot_raster(self, duration: float = 100.0, ax=None):
        """
        Plot spike raster for recent activity.
        
        Args:
            duration: Time duration to plot in ms
            ax: Matplotlib axis to plot on (creates new if None)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Current time and window
        current_time = self.network.current_time
        start_time = max(0, current_time - duration)
        
        # Extract relevant spikes from history
        spike_times = []
        spike_neurons = []
        
        for spikes, t in self.network.spike_history:
            if start_time <= t <= current_time:
                for n_idx in spikes:
                    spike_neurons.append(n_idx)
                    spike_times.append(t)
        
        # Plot raster
        if spike_times and spike_neurons:
            # Create array of neuron indices and times
            spike_neurons = np.array(spike_neurons)
            spike_times = np.array(spike_times)
            
            # Create color array
            color_list = []
            for n_idx in spike_neurons:
                if n_idx < self.n_sensory:
                    color_list.append("green")
                elif n_idx >= self.n_sensory + self.n_associative + self.n_inhibitory:
                    color_list.append("red")
                elif n_idx >= self.n_sensory + self.n_associative:
                    color_list.append("purple")
                else:
                    color_list.append("blue")
            
            # Plot scatter points
            ax.scatter(spike_times, spike_neurons, s=2, c=color_list, alpha=0.8)
            
            # Add population labels
            ax.axhline(self.n_sensory, color='k', linestyle='--', alpha=0.3)
            ax.axhline(self.n_sensory + self.n_associative, color='k', linestyle='--', alpha=0.3)
            ax.axhline(self.n_sensory + self.n_associative + self.n_inhibitory, color='k', linestyle='--', alpha=0.3)
            
            # Add text labels for populations
            ax.text(start_time, self.n_sensory / 2, 'Sensory', fontsize=10, ha='left', va='center')
            ax.text(start_time, self.n_sensory + self.n_associative / 2, 'Associative', fontsize=10, ha='left', va='center')
            ax.text(start_time, self.n_sensory + self.n_associative + self.n_inhibitory / 2, 'Inhibitory', fontsize=10, ha='left', va='center')
            ax.text(start_time, self.n_sensory + self.n_associative + self.n_inhibitory + self.n_output / 2, 'Output', fontsize=10, ha='left', va='center')
        
        ax.set_xlim(start_time, current_time)
        ax.set_ylim(0, self.network.n_neurons)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Spike Raster Plot')
        
        return ax
    
    def create_live_visualization(self, update_interval: int = 100):
        """
        Create a live visualization of network activity.
        
        Args:
            update_interval: Update interval in ms
            
        Returns:
            Matplotlib animation
        """
        # Create figure and axes
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax_raster = fig.add_subplot(gs[0, :])
        ax_rates = fig.add_subplot(gs[1, 0])
        ax_output = fig.add_subplot(gs[1, 1])
        
        # Initial plots
        self.plot_raster(ax=ax_raster)
        self.plot_firing_rates(ax=ax_rates)
        
        # Output neuron activity bars
        output_activities = self.get_active_output_neurons()
        token_ids = [str(x[0]) for x in output_activities[:10]]  # Top 10
        activities = [x[1] for x in output_activities[:10]]
        
        bars = ax_output.bar(token_ids, activities)
        ax_output.set_title('Top Active Output Neurons')
        ax_output.set_xlabel('Token ID')
        ax_output.set_ylabel('Firing Rate')
        
        # Update function for animation
        def update(frame):
            # Update statistics
            self._update_statistics()
            
            # Clear axes
            ax_raster.clear()
            ax_rates.clear()
            ax_output.clear()
            
            # Redraw plots
            self.plot_raster(ax=ax_raster)
            self.plot_firing_rates(ax=ax_rates)
            
            # Update output activity
            output_activities = self.get_active_output_neurons()
            token_ids = [str(x[0]) for x in output_activities[:10]]  # Top 10
            activities = [x[1] for x in output_activities[:10]]
            
            ax_output.bar(token_ids, activities)
            ax_output.set_title('Top Active Output Neurons')
            ax_output.set_xlabel('Token ID')
            ax_output.set_ylabel('Firing Rate')
            
            # Adjust layout
            fig.tight_layout()
            
            return []
        
        # Create animation
        animation = FuncAnimation(fig, update, interval=update_interval, blit=True)
        
        return animation 