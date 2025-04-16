# Technical Specification: Phase 1 - Hebbian SNN Core Framework

## 1. System Overview

The Phase 1 implementation establishes the foundational spiking neural network (SNN) simulator optimized for NVIDIA GPUs, focusing on biological plausibility through Hebbian learning mechanisms.

### 1.1 Core Objectives
- Create a highly scalable SNN simulator using JAX and custom CUDA kernels
- Implement biologically-plausible neuron dynamics and STDP learning
- Develop sparse connectivity representations optimized for GPU
- Build visualization tools for neuronal activity
- Support future multimodal extensions

## 2. Core Architecture

### 2.1 System Components
```
HebLLM Framework
├── Network Core
│   ├── Neuron Management
│   ├── Synapse Management
│   ├── STDP Learning
│   └── Neuromodulation System
├── Computational Engine
│   ├── JAX-based Computation
│   ├── Custom CUDA Kernels
│   └── Sparse Matrix Operations
├── Memory Management
│   ├── Spike Buffer
│   ├── Episodic Memory Prototype
│   └── Synaptic Weight Storage
├── Simulation Control
│   ├── Time-step Controller
│   ├── Event Scheduler
│   └── Sleep Phase Manager
└── Visualization & Analytics
    ├── Real-time Activity Monitor
    ├── Network Statistics
    └── Connectivity Visualizer
```

### 2.2 Component Interactions
- Neurons emit spikes based on membrane potential
- Spikes propagate through sparse synaptic connections
- STDP modifies synaptic weights based on timing relationships
- Neuromodulatory signals globally modulate plasticity
- Sleep phases trigger specialized consolidation processes

## 3. Neuron Implementation

### 3.1 Neuron Types

#### 3.1.1 Sensory Neurons
```python
class SensoryNeuron:
    # Parameters
    threshold = 0.5          # Firing threshold
    resting_potential = 0.0  # Resting membrane potential
    refractory_period = 2.0  # ms
    
    # Implementation focus
    - Sparse encoding of input tokens
    - Tunable receptive fields
    - Direct input-to-spike conversion
    - Minimal adaptation properties
```

#### 3.1.2 Associative Neurons
```python
class AssociativeNeuron:
    # Parameters
    threshold = 0.6                # Firing threshold (adaptive)
    resting_potential = -0.1       # Resting membrane potential
    refractory_period = 4.0        # ms
    adaptation_time_constant = 500 # ms
    
    # Implementation focus
    - Adaptive threshold
    - Spike-frequency adaptation
    - Rich recurrent connectivity
    - Variable conduction delays
```

#### 3.1.3 Inhibitory Interneurons
```python
class InhibitoryNeuron:
    # Parameters
    threshold = 0.4          # Lower threshold for quick activation
    resting_potential = -0.2 # Lower resting potential
    refractory_period = 3.0  # ms
    inhibition_strength = 2.0 # Relative inhibition strength
    
    # Implementation focus
    - Fast spiking dynamics
    - Local regulatory functions
    - Lateral inhibition patterns
    - Homeostatic regulation
```

#### 3.1.4 Output Neurons
```python
class OutputNeuron:
    # Parameters
    threshold = 0.8          # Higher threshold for accumulating evidence
    resting_potential = -0.1 # Resting membrane potential
    refractory_period = 5.0  # ms
    
    # Implementation focus
    - Integration of associative evidence
    - Winner-take-all competition
    - Direct mapping to vocabulary tokens
    - Readout mechanisms
```

### 3.2 Neuronal Dynamics

#### 3.2.1 Neuron Update Equations
```
# Leaky Integrate-and-Fire with Adaptive Threshold
dv/dt = -(v - v_rest)/tau_m + I(t)/C
dtheta/dt = -theta/tau_adapt + alpha * spike_count

# Spike Generation
if v >= (threshold + theta):
    emit spike
    v = v_reset
    theta += theta_increment
```

#### 3.2.2 Synaptic Current Model
```
# Post-synaptic current contribution
I_syn(t) = sum_j(w_ij * a_j(t))

# Synaptic activation dynamic
da_j/dt = -a_j/tau_syn + sum_s delta(t - t_j^s)
```

## 4. Synaptic Learning

### 4.1 STDP Implementation

#### 4.1.1 Weight Update Rule
```
# Basic STDP
dw_ij = 
    A_+ * exp(-|t_post - t_pre|/tau_+) if t_post > t_pre
    -A_- * exp(-|t_post - t_pre|/tau_-) if t_post < t_pre

# With Neuromodulation
dw_ij = m(t) * [STDP formula above]

Where:
- A_+ = 0.05  # Potentiation amplitude
- A_- = 0.0525  # Depression amplitude (slightly larger)
- tau_+ = 20ms  # Potentiation time constant
- tau_- = 20ms  # Depression time constant
- m(t) = neuromodulatory signal (ranges from 0.1 to 5.0)
```

#### 4.1.2 Trace-Based Efficient Implementation
```
# Pre-synaptic trace
dx_pre/dt = -x_pre/tau_+ + delta(t - t_pre)

# Post-synaptic trace
dx_post/dt = -x_post/tau_- + delta(t - t_post)

# Weight updates at each spike
If pre-neuron spikes:
    w_ij += -A_- * x_post * m(t)
If post-neuron spikes:
    w_ij += A_+ * x_pre * m(t)
```

### 4.2 Dual Plasticity Regimes

```python
# Fast plasticity (early learning)
fast_plasticity_config = {
    'A_plus': 0.05,
    'A_minus': 0.0525,
    'tau_plus': 20.0,
    'tau_minus': 20.0
}

# Slow plasticity (consolidated synapses)
slow_plasticity_config = {
    'A_plus': 0.01,
    'A_minus': 0.0105,
    'tau_plus': 40.0,
    'tau_minus': 40.0
}

# Transition conditions
def check_consolidation(synapse):
    if synapse.activation_frequency > threshold and synapse.age > min_age:
        synapse.plasticity_regime = 'slow'
```

### 4.3 Homeostatic Regulation

```python
# Synaptic scaling
w_ij_scaled = w_ij * (target_activity / actual_activity)^gamma

# Intrinsic plasticity
threshold_adapt = baseline_threshold + alpha * (actual_activity - target_activity)

Where:
- target_activity = 0.02  # Target spike rate (spikes/ms)
- gamma = 0.1  # Scaling factor
- alpha = 0.05  # Threshold adaptation rate
```

## 5. Sparse Connectivity

### 5.1 Data Structures

#### 5.1.1 Sparse Matrix Formats
```python
# Coordinate Format (COO) for rapid updates
coo_synapses = {
    'row_indices': jnp.array([...]),  # Pre-synaptic neuron indices
    'col_indices': jnp.array([...]),  # Post-synaptic neuron indices
    'weights': jnp.array([...]),      # Synaptic weights
}

# Compressed Sparse Row (CSR) for efficient propagation
csr_synapses = {
    'indptr': jnp.array([...]),      # Row pointers
    'indices': jnp.array([...]),     # Column indices
    'weights': jnp.array([...]),     # Synaptic weights
}
```

#### 5.1.2 Synaptic Properties
```python
# Extended synaptic information
synapse_properties = {
    'delays': jnp.array([...]),       # Conduction delays (ms)
    'plasticity_regime': jnp.array([...]),  # 'fast' or 'slow'
    'last_update': jnp.array([...]),  # Timestamp of last update
    'creation_time': jnp.array([...]),  # When synapse was created
    'activation_count': jnp.array([...])  # Activity counter
}
```

### 5.2 Connectivity Initialization

```python
# Initial connectivity
def initialize_connectivity(n_neurons, connectivity_density=0.1):
    # Random sparse connectivity
    n_connections = int(n_neurons * n_neurons * connectivity_density)
    
    # Random pre/post pairs with minimal self-connections
    pre_indices = jnp.random.randint(0, n_neurons, (n_connections,))
    post_indices = jnp.random.randint(0, n_neurons, (n_connections,))
    
    # Filter out self-connections
    mask = pre_indices != post_indices
    pre_indices = pre_indices[mask]
    post_indices = post_indices[mask]
    
    # Initialize weights with log-normal distribution
    weights = jnp.exp(jnp.random.normal(-2, 0.5, (len(pre_indices),)))
    
    return {
        'row_indices': pre_indices,
        'col_indices': post_indices,
        'weights': weights,
    }
```

### 5.3 Dynamic Connectivity

```python
# Synapse creation (structural plasticity)
def create_synapse(network, pre_idx, post_idx):
    # Add new connection with small initial weight
    network.add_synapse(pre_idx, post_idx, weight=0.01)
    
# Synapse pruning
def prune_synapse(network, pre_idx, post_idx):
    # Remove weak/unused connections
    if network.get_weight(pre_idx, post_idx) < prune_threshold:
        network.remove_synapse(pre_idx, post_idx)
```

## 6. Spike Propagation

### 6.1 Event-Driven Update

```python
# Event queue structure
spike_queue = PriorityQueue()  # Ordered by delivery time

# Spike propagation
def propagate_spikes(network, current_time):
    # Process all spikes scheduled for current time
    while spike_queue and spike_queue.peek().time <= current_time:
        # Get next spike event
        event = spike_queue.pop()
        
        # Update post-synaptic neurons
        for post_idx, weight, delay in network.get_connections(event.neuron_idx):
            # Calculate delivery time
            delivery_time = current_time + delay
            
            # Schedule current contribution
            target_neuron = network.neurons[post_idx]
            target_neuron.schedule_input(weight, delivery_time)
            
            # Trigger STDP updates
            update_synapse(network, event.neuron_idx, post_idx, event.time)
```

### 6.2 Parallel Spike Processing

```python
# GPU-optimized spike processing using JAX
def process_spikes_parallel(network, spikes, time):
    # Get all connections for spiking neurons
    pre_indices = jnp.array([spike.neuron_idx for spike in spikes])
    
    # Gather outgoing connections (using sparse ops)
    post_indices, weights, delays = network.get_outgoing_connections(pre_indices)
    
    # Calculate delivery times
    delivery_times = time + delays
    
    # Update post-synaptic neuron inputs (via atomic adds)
    update_inputs_kernel(post_indices, weights, delivery_times)
    
    # Trigger STDP
    update_synapses_stdp(pre_indices, post_indices, time)
```

## 7. Neuromodulation System

### 7.1 Neuromodulatory Signals

```python
class NeuromodulationSystem:
    # Baseline modulation
    baseline_modulation = 1.0
    
    # Modulatory signals
    novelty_signal = 0.0
    surprise_signal = 0.0
    reward_signal = 0.0
    
    def compute_modulation(self, network_state):
        # Calculate novelty (based on unfamiliar patterns)
        self.novelty_signal = calculate_novelty(network_state)
        
        # Calculate surprise (prediction error)
        self.surprise_signal = calculate_surprise(network_state)
        
        # Combine signals
        modulation = self.baseline_modulation + \
                    5.0 * self.novelty_signal + \
                    3.0 * self.surprise_signal + \
                    2.0 * self.reward_signal
        
        return jnp.clip(modulation, 0.1, 5.0)
```

### 7.2 Neuromodulation Application

```python
# Apply modulation to STDP
def modulated_stdp_update(pre_idx, post_idx, delta_t, modulation):
    if delta_t > 0:
        # Post-before-pre: LTD
        dw = -A_minus * jnp.exp(-delta_t/tau_minus) * modulation
    else:
        # Pre-before-post: LTP
        dw = A_plus * jnp.exp(delta_t/tau_plus) * modulation
    
    return dw
```

## 8. Sleep Consolidation Prototype

### 8.1 Basic Implementation

```python
class SleepConsolidation:
    def __init__(self):
        self.replay_buffer = []
        self.consolidation_threshold = 100  # min spikes before consolidation
    
    def record_activity(self, active_neurons, time):
        # Store spike patterns for replay
        self.replay_buffer.append((active_neurons, time))
    
    def trigger_consolidation(self, network):
        if len(self.replay_buffer) >= self.consolidation_threshold:
            # Replay spike patterns with reduced plasticity
            modulation_scale = 0.3  # Reduced plasticity during sleep
            
            # Replay recent patterns
            for neurons, time in self.replay_buffer:
                # Stimulate neurons that were active
                network.force_spikes(neurons)
                
                # Process with reduced plasticity
                network.process_with_modulation(modulation_scale)
            
            # Clear buffer after consolidation
            self.replay_buffer = []
```

## 9. GPU Optimization

### 9.1 JAX-Based Implementation

```python
# Core neuron update using JAX
@jax.jit
def update_neurons(states, inputs, dt):
    # Update membrane potentials vectorized
    new_v = states['v'] + dt * (-(states['v'] - states['v_rest'])/states['tau_m'] + inputs)
    
    # Apply threshold conditions
    spiking = new_v >= states['thresholds']
    
    # Reset spiking neurons
    new_v = jnp.where(spiking, states['v_reset'], new_v)
    
    # Update adaptive thresholds
    new_thresholds = states['thresholds'] + states['theta_increment'] * spiking
    
    # Update traces for STDP
    new_x_pre = states['x_pre'] * jnp.exp(-dt/states['tau_plus']) + spiking
    new_x_post = states['x_post'] * jnp.exp(-dt/states['tau_minus'])
    
    # Return updated states
    return {
        'v': new_v,
        'thresholds': new_thresholds,
        'x_pre': new_x_pre,
        'x_post': new_x_post,
        'spiking': spiking
    }
```

### 9.2 Custom CUDA Kernels

```python
# Example CUDA kernel for sparse spike propagation
SPARSE_SPIKE_PROPAGATION_KERNEL = """
extern "C" __global__ void propagate_spikes(
    const int* __restrict__ spikes,
    const int num_spikes,
    const int* __restrict__ csr_indptr,
    const int* __restrict__ csr_indices,
    const float* __restrict__ csr_weights,
    const float* __restrict__ csr_delays,
    float* __restrict__ neuron_inputs,
    float* __restrict__ event_times,
    const float current_time
) {
    int spike_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (spike_idx < num_spikes) {
        int pre_neuron = spikes[spike_idx];
        
        // Get connection range for this pre-synaptic neuron
        int start_idx = csr_indptr[pre_neuron];
        int end_idx = csr_indptr[pre_neuron + 1];
        
        // Process all post-synaptic connections
        for (int conn_idx = start_idx; conn_idx < end_idx; conn_idx++) {
            int post_neuron = csr_indices[conn_idx];
            float weight = csr_weights[conn_idx];
            float delay = csr_delays[conn_idx];
            
            // Calculate delivery time
            float delivery_time = current_time + delay;
            
            // Atomic add to target neuron's input buffer at appropriate time slot
            int time_slot = ((int)(delivery_time * 1000)) % MAX_DELAY_SLOTS;
            atomicAdd(&neuron_inputs[post_neuron * MAX_DELAY_SLOTS + time_slot], weight);
            atomicMin(&event_times[post_neuron], delivery_time);
        }
    }
}
"""
```

### 9.3 Memory Optimization

```python
# Optimize memory usage
def optimize_memory(network_size, sparsity=0.1):
    # Estimate number of connections
    num_connections = int(network_size * network_size * sparsity)
    
    # Memory requirements
    memory_per_connection = 12  # bytes (4 for indices, 4 for weight, 4 for delay)
    total_connection_memory = num_connections * memory_per_connection
    
    # Neuron state memory
    neuron_state_size = 28  # bytes per neuron
    total_neuron_memory = network_size * neuron_state_size
    
    # Spike buffer memory (assume 1% active)
    spike_buffer_size = int(network_size * 0.01) * 8  # bytes
    
    # Total memory
    total_memory = total_connection_memory + total_neuron_memory + spike_buffer_size
    
    # Memory sharding strategy
    if total_memory > AVAILABLE_GPU_MEMORY:
        # Split network by neuron groups
        num_shards = math.ceil(total_memory / AVAILABLE_GPU_MEMORY)
        neurons_per_shard = network_size // num_shards
        
        # Define sharding strategy
        return {
            'sharded': True,
            'num_shards': num_shards,
            'neurons_per_shard': neurons_per_shard
        }
    else:
        return {'sharded': False}
```

## 10. Visualization System

### 10.1 Activity Monitoring

```python
class ActivityMonitor:
    def __init__(self, network, buffer_size=1000):
        self.network = network
        self.spike_history = np.zeros((network.size, buffer_size), dtype=bool)
        self.current_idx = 0
        
    def record(self, spikes, timestep):
        # Record spikes at current timestep
        self.spike_history[:, self.current_idx % self.spike_history.shape[1]] = spikes
        self.current_idx += 1
        
    def get_firing_rates(self, window=100):
        # Calculate firing rates over window
        window_size = min(window, self.spike_history.shape[1])
        start_idx = max(0, self.current_idx - window_size)
        end_idx = self.current_idx
        
        if end_idx > start_idx:
            window_spikes = self.spike_history[:, start_idx % self.spike_history.shape[1]:end_idx % self.spike_history.shape[1]]
            return np.mean(window_spikes, axis=1)
        else:
            return np.zeros(self.network.size)
```

### 10.2 Network Connectivity Visualization

```python
def visualize_connectivity(network, max_neurons=1000):
    # Sample neurons if network is large
    if network.size > max_neurons:
        sampled_indices = np.random.choice(network.size, max_neurons, replace=False)
        weights = network.get_connectivity_submatrix(sampled_indices, sampled_indices)
    else:
        weights = network.get_connectivity_matrix()
    
    # Create connectivity visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(weights, cmap='viridis', vmin=0, vmax=np.percentile(weights[weights > 0], 95))
    plt.colorbar(label='Synaptic Weight')
    plt.title('Network Connectivity')
    plt.xlabel('Post-synaptic Neuron ID')
    plt.ylabel('Pre-synaptic Neuron ID')
    
    return plt.gcf()
```

## 11. Implementation Schedule

### 11.1 Development Timeline

1. **Week 1-2**: Core JAX-based SNN framework
   - Neuron dynamics implementation
   - Basic STDP mechanism
   - Initial tests

2. **Week 3-4**: Sparse connectivity implementation
   - COO/CSR representations
   - Memory-efficient storage
   - Connection management

3. **Week 5-6**: CUDA optimization
   - Custom kernels for critical operations
   - Performance benchmarking
   - Optimization

4. **Week 7-8**: Neuromodulation and sleep
   - Neuromodulatory system
   - Basic sleep consolidation
   - Integration testing

5. **Week 9-10**: Visualization and monitoring
   - Activity visualization tools
   - Network statistics
   - Performance analysis

6. **Week 11-12**: Testing and documentation
   - Comprehensive testing
   - Performance benchmarking
   - Documentation and API finalization

## 12. Testing Framework

### 12.1 Unit Tests

```python
def test_neuron_dynamics():
    # Test various neuron dynamics
    neuron = LIFNeuron()
    
    # Test sub-threshold behavior
    neuron.receive_input(0.4)
    assert not neuron.has_spiked()
    
    # Test spiking
    neuron.receive_input(0.6)
    assert neuron.has_spiked()
    
    # Test refractory period
    neuron.receive_input(1.0)
    assert not neuron.has_spiked()

def test_stdp_learning():
    # Test STDP updates
    pre_neuron = LIFNeuron()
    post_neuron = LIFNeuron()
    synapse = Synapse(pre_neuron, post_neuron, weight=0.5)
    
    # Test LTP (pre before post)
    pre_neuron.force_spike(time=10)
    post_neuron.force_spike(time=15)
    synapse.apply_stdp()
    
    assert synapse.weight > 0.5  # Weight should increase
    
    # Test LTD (post before pre)
    synapse.weight = 0.5  # Reset
    post_neuron.force_spike(time=20)
    pre_neuron.force_spike(time=25)
    synapse.apply_stdp()
    
    assert synapse.weight < 0.5  # Weight should decrease
```

### 12.2 Performance Benchmarks

```python
def benchmark_spike_propagation(network_sizes, sparsity=0.1):
    results = []
    
    for size in network_sizes:
        # Create network of given size
        network = SNN(size, sparsity)
        
        # Benchmark with 1% of neurons spiking
        active_neurons = np.random.choice(size, size//100)
        
        # Time propagation
        start_time = time.time()
        for _ in range(100):  # 100 timesteps
            network.inject_spikes(active_neurons)
            network.step()
        elapsed = time.time() - start_time
        
        results.append({
            'network_size': size,
            'propagation_time': elapsed / 100,  # Average per step
            'million_syn_ops': size * size * sparsity * 0.01 / 1e6  # MSOps
        })
    
    return results
```

## 13. Development Environment

### 13.1 Package Requirements

```
# Core dependencies
jax==0.4.2
jaxlib==0.4.2
cuda-version>=11.8
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0

# CUDA development
cupy>=12.0.0
pycuda>=2022.1

# Visualization
plotly>=5.14.0
dash>=2.9.0
networkx>=3.1

# Testing
pytest>=7.3.1
hypothesis>=6.75.3
```

### 13.2 GPU Configuration

```python
# JAX GPU configuration
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

# Memory configuration
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
```