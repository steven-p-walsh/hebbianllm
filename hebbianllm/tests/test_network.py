"""
Tests for HebLLM network.

This module contains comprehensive tests for the high-performance
Hebbian SNN implementation.
"""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import time
from hebbianllm import HebSNN


class TestHebSNN(unittest.TestCase):
    """Tests for the HebSNN network."""
    
    def setUp(self):
        """Set up test network."""
        self.network = HebSNN(
            n_sensory=20,
            n_associative=50,
            n_inhibitory=10,
            n_output=20,
            batch_size=4,
            connectivity_density=0.2,
            seed=42
        )
    
    def test_network_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.n_neurons, 100)
        self.assertEqual(self.network.n_sensory, 20)
        self.assertEqual(self.network.n_associative, 50)
        self.assertEqual(self.network.n_inhibitory, 10)
        self.assertEqual(self.network.n_output, 20)
        self.assertEqual(self.network.batch_size, 4)
        self.assertGreater(self.network.n_devices, 0)
    
    def test_step_function(self):
        """Test single step function."""
        # Create test input
        inputs = jnp.zeros(self.network.n_neurons)
        inputs = inputs.at[0].set(1.0)  # Stimulate first neuron
        
        # Run step
        spikes, novelty = self.network.step(inputs)
        
        # Check outputs
        self.assertEqual(spikes.shape, (self.network.n_neurons,))
        self.assertIsInstance(novelty, (float, np.floating, jnp.floating))
        self.assertGreaterEqual(novelty, 0.0)
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        # Create batch of patterns
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.1, shape=(4, self.network.n_neurons))
        
        # Run batch processing
        results = self.network.batch_run(patterns, n_steps=10)
        
        # Check results structure
        self.assertIn('final_states', results)
        self.assertIn('spike_history', results)
        
        # Check shapes
        spike_history = results['spike_history']
        self.assertEqual(spike_history.shape, (4, 10, self.network.n_neurons))
        
        # Check final states
        final_states = results['final_states']
        for key in ['v', 'pre_traces', 'post_traces', 'refractory_until']:
            self.assertIn(key, final_states)
            self.assertEqual(final_states[key].shape, (4, self.network.n_neurons))
    
    def test_memory_usage(self):
        """Test memory usage estimation."""
        memory_usage = self.network._estimate_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreater(memory_usage, 0)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.network.get_performance_stats()
        
        # Check required fields
        required_fields = ['n_neurons', 'n_connections', 'dtype', 'memory_usage_mb', 
                          'connectivity_density', 'batch_size']
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Check values
        self.assertEqual(stats['n_neurons'], self.network.n_neurons)
        self.assertEqual(stats['batch_size'], self.network.batch_size)
        self.assertGreater(stats['n_connections'], 0)
        self.assertGreater(stats['memory_usage_mb'], 0)
        self.assertGreater(stats['connectivity_density'], 0)
    
    def test_reset_functionality(self):
        """Test network reset functionality."""
        # Run some processing
        key = jax.random.PRNGKey(42)
        inputs = jax.random.bernoulli(key, 0.1, shape=(self.network.n_neurons,))
        
        # Take several steps
        for _ in range(5):
            self.network.step(inputs)
        
        # Check that state has changed
        self.assertGreater(self.network.current_time, 0)
        
        # Reset
        self.network.reset()
        
        # Check that state is reset
        self.assertEqual(self.network.current_time, 0)
        self.assertEqual(len(self.network.spike_history), 0)
    
    def test_connectivity_initialization(self):
        """Test connectivity initialization."""
        # Check connectivity structure
        self.assertGreater(len(self.network.pre_indices), 0)
        self.assertGreater(len(self.network.weights), 0)
        self.assertEqual(len(self.network.pre_indices), len(self.network.post_indices))
        self.assertEqual(len(self.network.pre_indices), len(self.network.weights))
        
        # Check weight bounds
        self.assertTrue(jnp.all(self.network.weights >= 0))
        self.assertTrue(jnp.all(self.network.weights <= 1))
        
        # Check connectivity density
        expected_connections = int(self.network.n_neurons ** 2 * 0.2)
        actual_connections = len(self.network.pre_indices)
        # Allow some tolerance due to self-connection filtering
        self.assertLess(abs(actual_connections - expected_connections), expected_connections * 0.1)
    
    def test_learning_functionality(self):
        """Test STDP learning functionality."""
        # Record initial weights
        initial_weights = jnp.copy(self.network.weights)
        
        # Create patterns that should trigger learning
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.2, shape=(4, self.network.n_neurons))
        
        # Run learning
        results = self.network.batch_run(patterns, n_steps=20)
        
        # Check that weights have changed
        weight_changes = self.network.weights - initial_weights
        total_change = jnp.sum(jnp.abs(weight_changes))
        
        # Should have some weight changes due to STDP
        self.assertGreater(total_change, 0)
        
        # Weights should still be in bounds
        self.assertTrue(jnp.all(self.network.weights >= 0))
        self.assertTrue(jnp.all(self.network.weights <= 1))
    
    def test_output_activity(self):
        """Test output activity retrieval."""
        # Run some processing first
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.1, shape=(4, self.network.n_neurons))
        
        # Process patterns
        results = self.network.batch_run(patterns, n_steps=10)
        
        # Store spike history for output activity calculation
        spike_history = results['spike_history']
        self.network.spike_history = [spike_history[0, i, :] for i in range(spike_history.shape[1])]
        
        # Get output activity
        output_activity = self.network.get_output_activity()
        
        # Check structure
        self.assertIsInstance(output_activity, dict)
        self.assertEqual(len(output_activity), self.network.n_output)
        
        # Check values
        for i, activity in output_activity.items():
            self.assertIsInstance(activity, float)
            self.assertGreaterEqual(activity, 0)
    
    def test_gpu_utilization(self):
        """Test GPU utilization if available."""
        # Skip if no GPU available
        if not jax.devices('gpu'):
            self.skipTest("No GPU available for GPU utilization test")
        
        # Create larger network for GPU test
        gpu_network = HebSNN(
            n_sensory=100,
            n_associative=400,
            n_inhibitory=100,
            n_output=100,
            batch_size=16
        )
        
        # Generate batch
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.05, shape=(16, 700))
        
        # Time batch processing
        start_time = time.time()
        results = gpu_network.batch_run(patterns, n_steps=50)
        processing_time = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(processing_time, 10.0)  # Should take less than 10 seconds
        
        # Check performance
        performance = (16 * 50) / processing_time
        self.assertGreater(performance, 100)  # Should get >100 pattern-steps/sec
    
    def test_multi_gpu_scaling(self):
        """Test multi-GPU scaling if multiple GPUs available."""
        gpu_devices = jax.devices('gpu')
        if len(gpu_devices) < 2:
            self.skipTest("Multiple GPUs not available for scaling test")
        
        # Create network that can utilize multiple GPUs
        multi_gpu_network = HebSNN(
            n_sensory=500,
            n_associative=2000,
            n_inhibitory=500,
            n_output=500,
            batch_size=64  # Large batch for multi-GPU
        )
        
        # Should detect multiple devices
        self.assertGreater(multi_gpu_network.n_devices, 1)
        
        # Test performance with large batch
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.05, shape=(64, 3500))
        
        start_time = time.time()
        results = multi_gpu_network.batch_run(patterns, n_steps=25)
        processing_time = time.time() - start_time
        
        # Should handle large batch efficiently
        performance = (64 * 25) / processing_time
        self.assertGreater(performance, 500)  # Should get >500 pattern-steps/sec
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large networks."""
        # Create network with reasonable size
        large_network = HebSNN(
            n_sensory=1000,
            n_associative=4000,
            n_inhibitory=1000,
            n_output=1000,
            batch_size=32,
            connectivity_density=0.05  # Keep sparse for memory efficiency
        )
        
        # Check memory usage is reasonable
        memory_usage = large_network._estimate_memory_usage()
        self.assertLess(memory_usage, 5000)  # Should be less than 5GB
        
        # Test that it can process batches without memory issues
        key = jax.random.PRNGKey(42)
        patterns = jax.random.bernoulli(key, 0.03, shape=(32, 7000))
        
        # Should not raise memory errors
        results = large_network.batch_run(patterns, n_steps=10)
        
        # Check results are valid
        self.assertIn('spike_history', results)
        self.assertEqual(results['spike_history'].shape, (32, 10, 7000))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)