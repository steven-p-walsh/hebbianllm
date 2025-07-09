"""
GPU Memory Calculator for Neural Networks

Calculates optimal neuron count for available GPU memory.
"""

import jax
import jax.numpy as jnp
import numpy as np


def estimate_memory_usage(n_neurons: int, dtype=jnp.float32) -> float:
    """
    Estimate GPU memory usage for a network with n_neurons.
    
    Returns memory usage in GB.
    """
    # Bytes per element based on dtype
    bytes_per_element = 4 if dtype == jnp.float32 else 8
    
    # Major memory components:
    # 1. Synaptic weight matrix: n_neurons x n_neurons
    synaptic_weights = n_neurons * n_neurons * bytes_per_element
    
    # 2. Neural states (voltage, spikes, etc): multiple arrays per neuron
    neural_states = n_neurons * 10 * bytes_per_element  # ~10 state variables
    
    # 3. Activity patterns and history
    activity_arrays = n_neurons * 50 * bytes_per_element  # Activity history
    
    # 4. Eligibility traces: n_neurons x n_neurons  
    eligibility_trace = n_neurons * n_neurons * bytes_per_element
    
    # 5. JAX compilation overhead and buffers (estimate 2x for safety)
    jax_overhead_multiplier = 2.0
    
    # Total memory
    total_bytes = (synaptic_weights + neural_states + activity_arrays + eligibility_trace) * jax_overhead_multiplier
    
    # Convert to GB
    total_gb = total_bytes / (1024**3)
    
    return total_gb


def find_max_neurons_for_memory(target_memory_gb: float, dtype=jnp.float32) -> int:
    """
    Find maximum number of neurons that fit in target memory.
    
    Uses binary search for efficiency.
    """
    print(f"Finding maximum neurons for {target_memory_gb}GB memory...")
    
    # Binary search bounds
    min_neurons = 100
    max_neurons = 100000  # Start with reasonable upper bound
    
    # Expand upper bound if needed
    while estimate_memory_usage(max_neurons, dtype) < target_memory_gb:
        max_neurons *= 2
        if max_neurons > 1000000:  # Sanity check
            break
    
    # Binary search for optimal size
    best_neurons = min_neurons
    
    while min_neurons <= max_neurons:
        mid_neurons = (min_neurons + max_neurons) // 2
        estimated_memory = estimate_memory_usage(mid_neurons, dtype)
        
        print(f"  Testing {mid_neurons:,} neurons: {estimated_memory:.2f}GB")
        
        if estimated_memory <= target_memory_gb:
            best_neurons = mid_neurons
            min_neurons = mid_neurons + 1
        else:
            max_neurons = mid_neurons - 1
    
    final_memory = estimate_memory_usage(best_neurons, dtype)
    print(f"  Optimal: {best_neurons:,} neurons using {final_memory:.2f}GB")
    
    return best_neurons


def get_gpu_memory_info():
    """Get available GPU memory information."""
    try:
        devices = jax.devices('gpu')
        if devices:
            # Try to get memory info (this is device-specific)
            print(f"Available GPUs: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"  GPU {i}: {device}")
            return True
        else:
            print("No GPUs available")
            return False
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return False


def test_memory_allocation(n_neurons: int):
    """Test actual memory allocation for given neuron count."""
    try:
        print(f"Testing allocation of {n_neurons:,} neurons...")
        
        # Try to allocate main arrays
        weights = jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32)
        activity = jnp.zeros(n_neurons, dtype=jnp.float32)
        
        # Force allocation on GPU
        weights = jax.device_put(weights)
        activity = jax.device_put(activity)
        
        print(f"✅ Successfully allocated {n_neurons:,} neurons")
        
        # Clean up
        del weights, activity
        
        return True
        
    except Exception as e:
        print(f"❌ Allocation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧠 GPU Memory Calculator for Neural Networks")
    print("=" * 50)
    
    # Show GPU info
    get_gpu_memory_info()
    print()
    
    # Calculate for 20GB
    target_memory = 20.0  # GB
    max_neurons = find_max_neurons_for_memory(target_memory)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Use {max_neurons:,} neurons for {target_memory}GB memory")
    print(f"   Estimated usage: {estimate_memory_usage(max_neurons):.2f}GB")
    
    # Test a smaller amount first
    test_neurons = max_neurons // 4
    print(f"\n🧪 Testing {test_neurons:,} neurons...")
    if test_memory_allocation(test_neurons):
        print("Memory allocation test passed!")
    else:
        print("Memory allocation test failed - reduce neuron count")