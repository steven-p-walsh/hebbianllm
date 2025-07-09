"""
HebLLM: High-Performance Hebbian Spiking Neural Network

A high-performance implementation of biologically-inspired Spiking Neural Networks
with GPU acceleration and multi-device support.
"""

from .core.network import HebSNN

__version__ = "0.1.0"
__all__ = ["HebSNN"]