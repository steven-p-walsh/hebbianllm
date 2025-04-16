# HebLLM: Hebbian Spiking Neural Network for Language Modeling

This project implements a biologically-inspired Spiking Neural Network (SNN) with Hebbian learning for language modeling. The implementation focuses on biological plausibility through Spike-Timing-Dependent Plasticity (STDP) and neuromodulation.

## Overview

HebLLM is a framework for building and experimenting with Hebbian learning in spiking neural networks. Key features include:

- Biologically plausible neuron dynamics
- STDP-based Hebbian learning
- Sparse connectivity optimized for performance
- Neuromodulation based on novelty and surprise
- Sleep-inspired consolidation phases
- Visualization tools for network activity

## Project Structure

```
hebbianllm/
├── core/
│   ├── neurons.py       # Neuron implementations
│   ├── synapses.py      # Synapse and connectivity
│   ├── network.py       # Main SNN implementation
│   └── neuromodulation.py # Neuromodulation system
├── utils/              # Utility functions
├── visualization/      # Visualization tools
│   └── activity_monitor.py # Activity visualization
└── tests/              # Test suite
    └── test_network.py # Network tests
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hebbianllm.git
cd hebbianllm

# Install dependencies
pip install -e .
```

## Requirements

- Python 3.8+
- JAX 0.4.2+
- NumPy 1.24.0+
- Matplotlib 3.7.0+

For GPU acceleration:
- CUDA 11.8+
- CuPy 12.0.0+

## Usage Example

```python
from hebbianllm.core.network import HebSNN
from hebbianllm.visualization.activity_monitor import ActivityMonitor

# Create network
network = HebSNN(
    n_sensory=100,
    n_associative=500,
    n_inhibitory=100,
    n_output=100
)

# Provide input
network.stimulate_by_token([5, 10, 15])  # Stimulate by token IDs

# Run simulation
network.run(500.0)  # Run for 500ms

# Get output
output = network.get_output_activity()
```

## Running Examples

Try the included demo:

```bash
python examples/simple_demo.py
```

## Testing

Run the test suite:

```bash
pytest
```

## Architecture

The HebLLM system consists of four main neuron types:

1. **Sensory Neurons**: Accept input tokens and convert them to spikes
2. **Associative Neurons**: Form the recurrent core of the network
3. **Inhibitory Neurons**: Regulate activity and enforce sparsity
4. **Output Neurons**: Map internal activity to output tokens

Learning occurs through STDP, which strengthens connections between neurons that fire in causal sequence and weakens those that fire out of sync.

## Development

This project is under active development. Planned improvements include:

- JAX-based optimization for faster simulation
- Custom CUDA kernels for spike propagation
- Support for larger-scale networks
- Integration with tokenizers for language processing
- Improved visualization and analysis tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is inspired by research in computational neuroscience, Hebbian learning, and biologically-plausible artificial intelligence. 