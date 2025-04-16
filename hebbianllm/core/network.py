"""
Core SNN network implementation.

This module implements the main Spiking Neural Network with
Hebbian learning for language processing.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import time
from queue import PriorityQueue

from hebbianllm.core.neurons import Neuron, SensoryNeuron, AssociativeNeuron, InhibitoryNeuron, OutputNeuron
from hebbianllm.core.synapses import Synapse, SparseConnectivity
from hebbianllm.core.neuromodulation import NeuromodulationSystem


class SpikeEvent:
    """Spike event in the network."""
    
    def __init__(self, neuron_idx: int, time: float):
        """
        Initialize a spike event.
        
        Args:
            neuron_idx: Index of the neuron that spiked
            time: Time of the event (delivery time)
        """
        self.neuron_idx = neuron_idx
        self.time = time
    
    def __lt__(self, other):
        """Compare events based on time for priority queue."""
        return self.time < other.time


class HebSNN:
    """
    Hebbian Spiking Neural Network.
    
    This is the main SNN class implementing the Hebbian learning
    mechanisms for language processing.
    """
    
    def __init__(self, 
                 n_sensory: int = 1000,
                 n_associative: int = 5000,
                 n_inhibitory: int = 1000,
                 n_output: int = 1000,
                 connectivity_density: float = 0.1,
                 seed: int = 42):
        """
        Initialize the Hebbian SNN.
        
        Args:
            n_sensory: Number of sensory neurons
            n_associative: Number of associative neurons
            n_inhibitory: Number of inhibitory neurons
            n_output: Number of output neurons
            connectivity_density: Connection density (0-1)
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        
        # Set network sizes
        self.n_sensory = n_sensory
        self.n_associative = n_associative
        self.n_inhibitory = n_inhibitory
        self.n_output = n_output
        self.n_neurons = n_sensory + n_associative + n_inhibitory + n_output
        
        # Create neurons
        self._create_neurons()
        
        # Create connectivity
        self.connectivity = SparseConnectivity(
            n_neurons=self.n_neurons,
            connectivity_density=connectivity_density,
            seed=seed
        )
        
        # Initialize neuromodulation system
        self.neuromodulation = NeuromodulationSystem()
        
        # Simulation state
        self.current_time = 0.0
        self.dt = 1.0  # 1ms time steps
        self.spike_queue = PriorityQueue()
        
        # Activity tracking
        self.spike_history = []
        self.activity_record = np.zeros((self.n_neurons, 1000), dtype=bool)  # Rolling buffer
        self.activity_idx = 0
        
        # Sleep consolidation
        self.replay_buffer = []
        self.time_since_last_sleep = 0.0
        self.sleep_interval = 1000.0  # Time between sleep phases
        self.consolidation_threshold = 100  # Min spikes before consolidation
    
    def _create_neurons(self):
        """Create all neurons in the network."""
        self.neurons = []
        
        # Create sensory neurons
        for i in range(self.n_sensory):
            self.neurons.append(SensoryNeuron(token_id=i))
        
        # Create associative neurons
        for i in range(self.n_associative):
            self.neurons.append(AssociativeNeuron())
        
        # Create inhibitory neurons
        for i in range(self.n_inhibitory):
            self.neurons.append(InhibitoryNeuron())
        
        # Create output neurons
        for i in range(self.n_output):
            self.neurons.append(OutputNeuron(token_id=i))
    
    def reset(self):
        """Reset the network state."""
        # Reset all neurons
        for neuron in self.neurons:
            neuron.reset()
        
        # Clear event queue
        self.spike_queue = PriorityQueue()
        
        # Reset simulation time
        self.current_time = 0.0
        
        # Clear tracking
        self.spike_history = []
        self.activity_record = np.zeros((self.n_neurons, 1000), dtype=bool)
        self.activity_idx = 0
        
        # Reset sleep state
        self.replay_buffer = []
        self.time_since_last_sleep = 0.0
    
    def inject_spikes(self, neuron_indices: List[int]):
        """
        Inject spikes into specific neurons.
        
        Args:
            neuron_indices: Indices of neurons to activate
        """
        for idx in neuron_indices:
            if idx < 0 or idx >= self.n_neurons:
                continue
                
            # Schedule immediate spike
            self.spike_queue.put(SpikeEvent(idx, self.current_time))
    
    def stimulate_by_token(self, token_ids: List[int]):
        """
        Stimulate sensory neurons corresponding to specific tokens.
        
        Args:
            token_ids: List of token IDs to stimulate
        """
        neurons_to_spike = []
        
        # Find sensory neurons with matching token IDs
        for idx in range(self.n_sensory):
            if isinstance(self.neurons[idx], SensoryNeuron) and self.neurons[idx].token_id in token_ids:
                neurons_to_spike.append(idx)
        
        # Inject spikes
        self.inject_spikes(neurons_to_spike)
    
    def step(self):
        """Execute a single timestep of the simulation."""
        # Process all spikes scheduled for current time
        self._process_spikes()
        
        # Update all neurons
        spiking_neurons = self._update_neurons()
        
        # Check for sleep phase
        self.time_since_last_sleep += self.dt
        if self.time_since_last_sleep >= self.sleep_interval:
            self._sleep_consolidation()
            self.time_since_last_sleep = 0.0
        
        # Advance time
        self.current_time += self.dt
        
        return spiking_neurons
    
    def _process_spikes(self):
        """Process all spikes scheduled for current time."""
        # Collect spikes for this timestep
        current_spikes = []
        
        while not self.spike_queue.empty():
            event = self.spike_queue.queue[0]
            
            # If event is in the future, stop processing
            if event.time > self.current_time:
                break
            
            # Remove event from queue
            event = self.spike_queue.get()
            current_spikes.append(event.neuron_idx)
        
        # If no spikes, return
        if not current_spikes:
            return
        
        # Record activity
        self._record_activity(current_spikes)
        
        # Process each spike
        for neuron_idx in current_spikes:
            # Get all outgoing connections
            connections = self.connectivity.get_connections(neuron_idx)
            
            # Spike transmission
            for post_idx, weight, delay in connections:
                # Inhibitory neurons provide negative input
                if isinstance(self.neurons[neuron_idx], InhibitoryNeuron):
                    effective_weight = -weight * self.neurons[neuron_idx].inhibition_strength
                else:
                    effective_weight = weight
                
                # Schedule input to post-synaptic neuron
                delivery_time = self.current_time + delay
                
                # Apply input and check for new spikes
                self.neurons[post_idx].receive_input(effective_weight, delivery_time)
                
                # # If post-synaptic neuron spikes, add to queue
                # if self.neurons[post_idx].membrane_potential >= self.neurons[post_idx].threshold:
                #     self.spike_queue.put(SpikeEvent(post_idx, delivery_time))
                #     self.neurons[post_idx].membrane_potential = self.neurons[post_idx].resting_potential
        
        # Update modulation based on current activity
        activity = np.zeros(self.n_neurons, dtype=bool)
        activity[current_spikes] = True
        modulation = self.neuromodulation.update(activity)
        
        # Apply STDP learning
        self._apply_stdp(current_spikes, modulation)
    
    def _update_neurons(self):
        """Update all neurons for current timestep."""
        
        # Debug info (optional)
        # Check if there are any neurons with potential close to threshold
        # near_threshold = []
        # for i, neuron in enumerate(self.neurons):
        #     if not neuron.is_refractory and neuron.membrane_potential > neuron.threshold * 0.8:
        #         near_threshold.append((i, neuron.membrane_potential, neuron.threshold))
        # if near_threshold:
        #     print(f"Time {self.current_time}: Neurons close to threshold: {near_threshold[:5]}")
        
        spiking_neurons = []
        
        for i, neuron in enumerate(self.neurons):
            # Update neuron state
            spiked = neuron.update(self.dt, self.current_time)
            
            # If neuron spiked, add to queue
            if spiked:
                self.spike_queue.put(SpikeEvent(i, self.current_time))
                spiking_neurons.append(i)
                
                # Debug: Print info about spiking neuron
                # neuron_type = "Unknown"
                # if i < self.n_sensory:
                #     neuron_type = "Sensory"
                # elif i < self.n_sensory + self.n_associative:
                #     neuron_type = "Associative"
                # elif i < self.n_sensory + self.n_associative + self.n_inhibitory:
                #     neuron_type = "Inhibitory"
                # else:
                #     neuron_type = "Output"
                # print(f"Neuron {i} ({neuron_type}) spiked at time {self.current_time}")
        
        # Debug info (optional)
        # if spiking_neurons:
        #     print(f"Time {self.current_time}: Neurons that spiked: {spiking_neurons}") 
            
        return spiking_neurons
    
    def _apply_stdp(self, spiking_neurons: List[int], modulation: float):
        """
        Apply STDP learning to synapses connected to spiking neurons.
        
        Args:
            spiking_neurons: Indices of neurons that spiked
            modulation: Neuromodulatory signal
        """
        # Simple, stable, direct Hebbian learning approach
        for neuron_idx in spiking_neurons:
            # 1. Strengthen incoming connections - core Hebbian principle
            # "Neurons that fire together, wire together"
            incoming_connections = self._get_incoming_connections(neuron_idx)
            for pre_idx, _ in incoming_connections:
                pre_idx = int(pre_idx)  # Ensure integer
                
                # Skip inhibitory neurons (their connections handled differently)
                if pre_idx >= self.n_sensory + self.n_associative and pre_idx < self.n_sensory + self.n_associative + self.n_inhibitory:
                    continue
                
                # Strengthen connection (the pre-neuron helped this neuron fire)
                current_weight = self.connectivity.get_weight(pre_idx, neuron_idx)
                if current_weight is not None:
                    # Apply scaled learning rate based on whether pre-neuron recently fired
                    if self.neurons[pre_idx].pre_trace > 0.1:  # Evidence it recently fired
                        # Stronger potentiation with high modulation
                        potentiation = 0.05 * modulation
                        # Ensure early neurons get stable enhancement
                        if pre_idx < self.n_sensory and neuron_idx >= self.n_neurons - self.n_output:
                            # Direct sensory->output connections get extra boost
                            potentiation *= 2.0
                        new_weight = np.clip(current_weight + potentiation, 0.0, 1.0)
                        self.connectivity.update_weight(pre_idx, neuron_idx, new_weight)
            
            # 2. Handle inhibitory neurons specially - they should learn the opposite way
            is_inhibitory = (neuron_idx >= self.n_sensory + self.n_associative and 
                             neuron_idx < self.n_sensory + self.n_associative + self.n_inhibitory)
            
            if is_inhibitory:
                # Inhibitory neurons strengthen when post-neurons don't fire
                # We look at highly active neurons and strengthen inhibition to them
                for post_idx in range(self.n_neurons):
                    if self.neurons[post_idx].membrane_potential > 0.5 * self.neurons[post_idx].threshold:
                        # This neuron is getting close to firing, strengthen inhibition
                        current_weight = self.connectivity.get_weight(neuron_idx, post_idx)
                        if current_weight is not None:
                            inhibition_boost = 0.02 * modulation
                            new_weight = np.clip(current_weight + inhibition_boost, 0.0, 1.0) 
                            self.connectivity.update_weight(neuron_idx, post_idx, new_weight)
            
            # 3. Optional: implement trace-based LTD for outgoing connections
            # Note: For stable learning initially, we can focus just on potentiation
            # and let competition between neurons handle the specificity
    
    def _get_incoming_connections(self, post_idx: int) -> List[Tuple[int, float]]:
        """
        Get all incoming connections to a neuron.
        
        Args:
            post_idx: Index of the post-synaptic neuron
            
        Returns:
            List of (pre_idx, weight) tuples
        """
        connections = []
        for i, post_i in enumerate(self.connectivity.post_indices):
            if post_i == post_idx:
                pre_idx = self.connectivity.pre_indices[i]
                weight = self.connectivity.weights[i]
                connections.append((pre_idx, weight))
        
        return connections
    
    def _record_activity(self, spiking_neurons: List[int]):
        """
        Record neural activity for analysis.
        
        Args:
            spiking_neurons: Indices of neurons that spiked
        """
        # Update activity record (rolling buffer)
        self.activity_record[:, self.activity_idx % self.activity_record.shape[1]] = False
        self.activity_record[spiking_neurons, self.activity_idx % self.activity_record.shape[1]] = True
        self.activity_idx += 1
        
        # Record for sleep replay
        self.replay_buffer.append(spiking_neurons)
        if len(self.replay_buffer) > 100:  # Limit buffer size
            self.replay_buffer.pop(0)
        
        # Record spike history
        self.spike_history.append((spiking_neurons, self.current_time))
    
    def _sleep_consolidation(self):
        """Perform sleep consolidation phase."""
        if len(self.replay_buffer) < self.consolidation_threshold:
            return  # Not enough activity to consolidate
        
        # Save original state
        original_time = self.current_time
        
        # Reduced plasticity during sleep
        modulation_scale = 0.3
        
        # Replay recent patterns
        for spikes in self.replay_buffer:
            # Inject spikes from replay
            self.inject_spikes(spikes)
            
            # Process with reduced plasticity
            activity = np.zeros(self.n_neurons, dtype=bool)
            activity[spikes] = True
            
            # Force low modulation
            self.neuromodulation.novelty_signal = 0.0
            self.neuromodulation.surprise_signal = 0.0
            modulation = self.neuromodulation.update(activity) * modulation_scale
            
            # Apply STDP with low modulation
            self._apply_stdp(spikes, modulation)
        
        # Restore state
        self.current_time = original_time
        
        # Prune weak synapses
        self._prune_synapses()
        
        # Check for consolidation
        self._check_synapse_consolidation()
        
        # Clear replay buffer after consolidation
        self.replay_buffer = []
    
    def _prune_synapses(self, prune_threshold: float = 0.01):
        """
        Prune weak synapses.
        
        Args:
            prune_threshold: Weight threshold below which to prune
        """
        # Find weak synapses
        pruning_mask = self.connectivity.weights < prune_threshold
        
        # Remove pruned synapses
        for i in range(len(pruning_mask)):
            if pruning_mask[i]:
                pre_idx = self.connectivity.pre_indices[i]
                post_idx = self.connectivity.post_indices[i]
                self.connectivity.remove_synapse(pre_idx, post_idx)
    
    def _check_synapse_consolidation(self, min_age: float = 1000.0, threshold: int = 50):
        """
        Check for synapse consolidation.
        
        Args:
            min_age: Minimum age in ms before consolidation
            threshold: Activation count threshold for consolidation
        """
        # Check each synapse
        for i in range(len(self.connectivity.activation_count)):
            if (self.connectivity.plasticity_regime[i] == 'fast' and
                self.connectivity.activation_count[i] > threshold and 
                (self.current_time - self.connectivity.creation_time[i]) > min_age):
                
                # Consolidate synapse
                self.connectivity.plasticity_regime[i] = 'slow'
    
    def get_output_activity(self) -> Dict[int, float]:
        """
        Get activity of output neurons.
        
        Returns:
            Dictionary mapping token IDs to activity levels
        """
        output_activity = {}
        
        # Calculate recent firing rates for output neurons
        window_size = min(100, self.activity_record.shape[1])
        
        for i in range(self.n_neurons - self.n_output, self.n_neurons):
            if isinstance(self.neurons[i], OutputNeuron) and self.neurons[i].token_id is not None:
                # Extract recent activity for this neuron
                start_idx = max(0, self.activity_idx - window_size)
                end_idx = self.activity_idx
                
                if end_idx > start_idx:
                    start_mod = start_idx % self.activity_record.shape[1]
                    end_mod = end_idx % self.activity_record.shape[1]
                    
                    if end_mod > start_mod:
                        activity = self.activity_record[i, start_mod:end_mod]
                    else:
                        # Handle wrap-around
                        activity1 = self.activity_record[i, start_mod:]
                        activity2 = self.activity_record[i, :end_mod]
                        activity = np.concatenate([activity1, activity2])
                    
                    # Calculate firing rate
                    firing_rate = np.mean(activity)
                    
                    # Store in output
                    output_activity[self.neurons[i].token_id] = firing_rate
                    
        return output_activity
    
    def run(self, duration: float, input_fn: Optional[Callable] = None):
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Duration to run in ms
            input_fn: Optional function that provides input at each timestep
        """
        end_time = self.current_time + duration
        
        while self.current_time < end_time:
            # Get optional input
            if input_fn is not None:
                input_data = input_fn(self.current_time)
                if input_data is not None:
                    self.inject_spikes(input_data)
            
            # Step simulation
            self.step()
            
            # Optionally yield for real-time visualization
            # yield self.get_state() 