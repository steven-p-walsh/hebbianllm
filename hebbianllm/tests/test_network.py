"""
Tests for the Hebbian SNN network.

This module contains tests for the core SNN functionality.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hebbianllm.core.network import HebSNN
from hebbianllm.core.neurons import SensoryNeuron, AssociativeNeuron
from hebbianllm.visualization.activity_monitor import ActivityMonitor


class TestHebSNN(unittest.TestCase):
    """Tests for the Hebbian SNN network."""
    
    def setUp(self):
        """Set up a small test network."""
        # Create a small test network
        self.network = HebSNN(
            n_sensory=20,
            n_associative=50,
            n_inhibitory=10,
            n_output=20,
            connectivity_density=0.2,
            seed=42
        )
    
    def test_network_initialization(self):
        """Test network initialization."""
        # Check neuron counts
        self.assertEqual(len(self.network.neurons), 100)
        
        # Check neuron types
        self.assertTrue(isinstance(self.network.neurons[0], SensoryNeuron))
        self.assertTrue(isinstance(self.network.neurons[25], AssociativeNeuron))
        
        # Check connectivity
        connections = self.network.connectivity.get_connections(0)
        self.assertGreater(len(connections), 0)
    
    def test_stimulation_and_propagation(self):
        """Test spike stimulation and propagation."""
        # Stimulate a sensory neuron
        self.network.inject_spikes([0])
        
        # Check that the spike is in the queue
        self.assertFalse(self.network.spike_queue.empty())
        
        # Run for 10 ms
        self.network.run(10.0)
        
        # Check that spikes propagated
        self.assertGreater(len(self.network.spike_history), 0)
    
    def test_stdp_learning(self):
        """Test STDP learning."""
        # Record initial weight
        pre_idx = 0
        post_idx = 25  # An associative neuron
        
        # Make sure there's a connection
        initial_weight = self.network.connectivity.get_weight(pre_idx, post_idx)
        if initial_weight is None:
            # Create connection if it doesn't exist
            self.network.connectivity.add_synapse(pre_idx, post_idx, weight=0.5)
            initial_weight = 0.5
        
        # Stimulate pre-synaptic neuron
        self.network.inject_spikes([pre_idx])
        
        # Run for 5 ms
        self.network.run(5.0)
        
        # Stimulate post-synaptic neuron (should result in LTP)
        self.network.inject_spikes([post_idx])
        
        # Run for 10 more ms
        self.network.run(10.0)
        
        # Check if weight increased
        final_weight = self.network.connectivity.get_weight(pre_idx, post_idx)
        self.assertIsNotNone(final_weight)
        
        # Should have increased due to LTP (or at least not decreased)
        self.assertGreaterEqual(final_weight, initial_weight)
    
    def test_token_stimulation(self):
        """Test stimulation of neurons by token IDs."""
        # Since sensory neurons have token_id = index by default,
        # this should stimulate the first sensory neuron
        self.network.stimulate_by_token([0, 5, 10])
        
        # Check that the spikes are in the queue
        self.assertFalse(self.network.spike_queue.empty())
        
        # Run for a short duration
        self.network.run(5.0)
        
        # Check that something was recorded in history
        self.assertGreater(len(self.network.spike_history), 0)
    
    def test_sleep_consolidation(self):
        """Test sleep consolidation functionality."""
        # Inject a bunch of spikes to trigger consolidation
        for i in range(100):  # Need >consolidation_threshold spikes
            self.network.inject_spikes([i % 20])  # Cycle through first 20 neurons
            self.network.run(2.0)  # Short run for each
            
        # Manually trigger sleep
        self.network.time_since_last_sleep = self.network.sleep_interval
        self.network._sleep_consolidation()
        
        # Verify replay buffer is cleared after consolidation
        self.assertEqual(len(self.network.replay_buffer), 0)
    
    def test_neuromodulation(self):
        """Test neuromodulation functionality."""
        # Record baseline modulation
        baseline = self.network.neuromodulation.baseline_modulation
        
        # Set a high novelty signal
        self.network.neuromodulation.novelty_signal = 0.8
        
        # Get modulation
        modulation = self.network.neuromodulation.get_modulation()['total']
        
        # Should be higher than baseline
        self.assertGreater(modulation, baseline)
    
    def test_activity_monitor(self):
        """Test activity monitoring."""
        # Create monitor
        monitor = ActivityMonitor(self.network)
        
        # Run network with some activity
        for i in range(10):
            self.network.inject_spikes([i])
            self.network.run(10.0)
            monitor.update(self.network.current_time)
        
        # Check population rates
        rates = monitor.get_population_rates()
        self.assertIn('sensory', rates)
        self.assertIn('overall', rates)
        
        # Test plotting - just check that it doesn't error
        try:
            ax = monitor.plot_firing_rates()
            plt.close()
            
            ax = monitor.plot_raster()
            plt.close()
        except Exception as e:
            self.fail(f"Plotting raised an exception: {e}")


if __name__ == '__main__':
    unittest.main() 