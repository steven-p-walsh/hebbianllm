# HebLLM: High-Performance Hebbian Spiking Neural Network

A high-performance implementation of biologically-inspired Spiking Neural Networks (SNNs) with Hebbian learning, featuring GPU acceleration and multi-device support.

## Features

- **GPU Acceleration**: Automatic GPU detection with multi-GPU support
- **High Performance**: Up to 30,000+ pattern-steps/second on modern GPUs
- **Biological Plausibility**: Spike-Timing-Dependent Plasticity (STDP) learning
- **Batch Processing**: Efficient batch processing for training and inference
- **Memory Efficient**: Sparse connectivity and optimized memory usage
- **Easy-to-Use**: Simple, clean API focused on performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hebbianllm.git
cd hebbianllm

# Install in development mode
pip install -e .

# For GPU support (recommended)
pip install -e ".[cuda]"
```

## Requirements

- Python 3.8+
- JAX 0.4.2+ (with GPU support for acceleration)
- NumPy 1.24.0+

For GPU acceleration:
- CUDA 11.8+ or ROCm 5.0+
- Compatible GPU (tested on RTX 3090, A100)

## Quick Start

### Basic Usage

```python
from hebbianllm import HebSNN
import jax
import jax.numpy as jnp

# Create network
network = HebSNN(
    n_sensory=1000,
    n_associative=4000,
    n_inhibitory=1000,
    n_output=1000,
    batch_size=64  # Process multiple patterns simultaneously
)

# Generate batch of input patterns
key = jax.random.PRNGKey(42)
patterns = jax.random.bernoulli(key, 0.05, shape=(64, 7000))

# Process batch efficiently
results = network.batch_run(patterns, n_steps=100)

# Get performance statistics
stats = network.get_performance_stats()
print(f"Performance: {stats['n_neurons']} neurons, {stats['memory_usage_mb']:.1f} MB")
```

### Single Step Processing

```python
from hebbianllm import HebSNN
import jax.numpy as jnp

# Create network
network = HebSNN(
    n_sensory=100,
    n_associative=500,
    n_inhibitory=100,
    n_output=100
)

# Create input pattern
inputs = jnp.zeros(700)
inputs = inputs.at[0:10].set(1.0)  # Stimulate first 10 neurons

# Process single step
spikes, novelty = network.step(inputs)
print(f"Spikes: {jnp.sum(spikes)}, Novelty: {novelty:.4f}")
```

### Learning Demonstration

```python
from hebbianllm import HebSNN
import jax
import jax.numpy as jnp

# Create network
network = HebSNN(
    n_sensory=200,
    n_associative=800,
    n_inhibitory=200,
    n_output=200,
    batch_size=8
)

# Create training patterns
key = jax.random.PRNGKey(42)
patterns = jax.random.bernoulli(key, 0.1, shape=(8, 1400))

# Train network (STDP learning happens automatically)
for epoch in range(10):
    results = network.batch_run(patterns, n_steps=20)
    print(f"Epoch {epoch+1}: {jnp.sum(results['spike_history'])} total spikes")
```

## API Reference

### HebSNN Class

The main class for creating and running high-performance Hebbian SNNs.

#### Constructor Parameters

- `n_sensory` (int): Number of input neurons (default: 1000)
- `n_associative` (int): Number of recurrent processing neurons (default: 5000)
- `n_inhibitory` (int): Number of inhibitory neurons (default: 1000)
- `n_output` (int): Number of output neurons (default: 1000)
- `batch_size` (int): Number of patterns to process simultaneously (default: 32)
- `connectivity_density` (float): Sparsity of connections (default: 0.1)
- `mixed_precision` (bool): Use mixed precision for memory efficiency (default: True)
- `seed` (int): Random seed for reproducibility (default: 42)

#### Methods

- `step(inputs, dt=1.0)`: Process single time step
- `batch_run(patterns, n_steps)`: Process batch of patterns
- `reset()`: Reset network state
- `get_performance_stats()`: Get performance statistics
- `get_output_activity(window_size=100)`: Get output neuron activity

#### Properties

- `n_neurons`: Total number of neurons
- `n_devices`: Number of devices (GPUs/CPUs) being used
- `current_time`: Current simulation time
- `weights`: Current connection weights

## Examples

### Run Examples

```bash
# Basic demonstration
python examples/simple_demo.py

# Performance benchmarking
python examples/performance_demo.py

# Learning demonstration
python examples/learning_demo.py
```

### Example Output

```
HebLLM Demo
===========
Created network with 700 neurons
- Sensory: 100
- Associative: 500
- Inhibitory: 100
- Output: 100
Using 2 device(s)

Processing batch of 16 patterns for 50 time steps...
Processing completed in 0.045s
Performance: 17777.8 pattern-steps/sec
Total spikes generated: 1247
Network sparsity: 0.22%

Performance Statistics:
- Memory usage: 1.2 MB
- Connections: 49,000
- Connectivity density: 0.100
```

## Performance Benchmarks

Tested on dual RTX 3090 GPUs:

| Network Size | Batch Size | Performance | Memory Usage |
|-------------|------------|-------------|--------------|
| 3.5K neurons | 32 | 15,000 steps/sec | 162 MB |
| 7K neurons | 64 | 25,000 steps/sec | 503 MB |
| 14K neurons | 128 | 31,000 steps/sec | 994 MB |

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest hebbianllm/tests/ -v

# Run specific test
python -m pytest hebbianllm/tests/test_network.py::TestHebSNN::test_batch_processing -v

# Run GPU tests (requires GPU)
python -m pytest hebbianllm/tests/test_network.py::TestHebSNN::test_gpu_utilization -v
```

## Project Structure

```
hebbianllm/
├── hebbianllm/
│   ├── __init__.py          # Main package interface
│   ├── core/
│   │   ├── __init__.py
│   │   └── network.py       # High-performance HebSNN implementation
│   └── tests/
│       └── test_network.py  # Comprehensive test suite
├── examples/
│   ├── simple_demo.py       # Basic usage demonstration
│   ├── performance_demo.py  # Performance benchmarking
│   └── learning_demo.py     # Learning demonstration
├── setup.py                 # Package configuration
└── README.md               # This file
```

## Advanced Usage

### Custom Network Architecture

```python
# Create custom network with specific parameters
network = HebSNN(
    n_sensory=2000,
    n_associative=8000,
    n_inhibitory=2000,
    n_output=2000,
    connectivity_density=0.05,  # Sparser connections
    batch_size=128,
    seed=42
)
```

### Multi-GPU Scaling

```python
# Automatically uses all available GPUs
network = HebSNN(
    n_sensory=5000,
    n_associative=20000,
    n_inhibitory=5000,
    n_output=5000,
    batch_size=256  # Large batch size for multi-GPU
)

print(f"Using {network.n_devices} GPUs")
```

### Memory Monitoring

```python
# Monitor memory usage
stats = network.get_performance_stats()
print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")
print(f"Connections: {stats['n_connections']:,}")
print(f"Connectivity density: {stats['connectivity_density']:.4f}")
```

### Performance Optimization Tips

1. **Batch Size**: Use larger batch sizes (64-256) for better GPU utilization
2. **Connectivity**: Lower density (0.01-0.1) for better memory efficiency
3. **Network Size**: Scale all neuron types proportionally
4. **GPU Memory**: Monitor usage and adjust batch size accordingly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest hebbianllm/tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HebLLM in your research, please cite:

```bibtex
@software{hebbllm2024,
  title={HebLLM: High-Performance Hebbian Spiking Neural Network},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hebbianllm}
}
```

## Support

- GitHub Issues: Bug reports and feature requests
- Documentation: See examples/ directory
- Performance Issues: Check GPU setup and JAX installation

## Acknowledgments

This project implements advanced optimization techniques for biologically-inspired neural networks, drawing from research in computational neuroscience and high-performance computing.