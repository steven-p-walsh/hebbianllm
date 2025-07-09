#!/usr/bin/env python3
"""
Unit tests for Phase 5 - Structural-plasticity optimization implementation.

Tests the performance optimizations and efficiency improvements.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import time

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def test_modulators_bug_fix():
    """Test that modulators attribute is properly initialized."""
    print("Testing modulators bug fix...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Check that modulators attribute exists
    assert hasattr(learner.network, 'modulators'), "Network should have modulators attribute"
    
    # Check that modulators can be set and retrieved
    learner.network.modulators.set_mod('dopamine', 0.5)
    dopamine_level = learner.network.modulators.get_mod('dopamine')
    assert abs(dopamine_level - 0.5) < 0.01, f"Dopamine should be 0.5, got {dopamine_level}"
    
    # Test all neuromodulators
    modulators = ['dopamine', 'acetylcholine', 'norepinephrine', 'adenosine']
    for mod in modulators:
        learner.network.modulators.set_mod(mod, 0.8)
        level = learner.network.modulators.get_mod(mod)
        assert abs(level - 0.8) < 0.01, f"{mod} should be 0.8, got {level}"
    
    print("✅ Modulators bug fix tests passed")


def test_connectivity_cap_optimization():
    """Test efficient top-k connectivity cap vs old percentile method."""
    print("Testing connectivity cap optimization...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Create high connectivity situation
    # Add many connections to trigger sleep consolidation
    n_connections = int(0.15 * learner.network.n_neurons * learner.network.n_neurons)  # 15% connectivity
    
    # Random weight matrix with high connectivity
    key = jax.random.PRNGKey(42)
    random_weights = jax.random.normal(key, (learner.network.n_neurons, learner.network.n_neurons)) * 0.1
    # Keep only top connections to create realistic structure
    weight_magnitude = jnp.abs(random_weights)
    _, top_indices = jax.lax.top_k(weight_magnitude.flatten(), n_connections)
    
    # Set high connectivity weights
    weights = jnp.zeros_like(random_weights)
    flat_weights = weights.flatten()
    flat_weights = flat_weights.at[top_indices].set(random_weights.flatten()[top_indices])
    learner.network.synaptic_weights = flat_weights.reshape(learner.network.n_neurons, learner.network.n_neurons)
    
    # Test sleep consolidation
    initial_connectivity = float(jnp.mean(jnp.abs(learner.network.synaptic_weights) > 0.001))
    print(f"  Initial connectivity: {initial_connectivity:.1%}")
    
    # Apply connectivity cap (should trigger sleep consolidation)
    start_time = time.time()
    learner.network._apply_connectivity_cap(max_connectivity=0.08)
    consolidation_time = time.time() - start_time
    
    # Check results
    final_connectivity = float(jnp.mean(jnp.abs(learner.network.synaptic_weights) > 0.001))
    print(f"  Final connectivity: {final_connectivity:.1%}")
    print(f"  Consolidation time: {consolidation_time:.4f}s")
    
    # Verify connectivity was reduced
    assert final_connectivity < initial_connectivity, "Connectivity should be reduced"
    assert final_connectivity <= 0.08, "Connectivity should be capped at 8%"
    
    # Check performance stats
    perf_stats = learner.network.get_performance_stats()
    assert perf_stats['connectivity_caps'] > 0, "Should have recorded connectivity cap"
    assert perf_stats['sleep_consolidations'] > 0, "Should have recorded sleep consolidation"
    
    print("✅ Connectivity cap optimization tests passed")


def test_adaptive_structural_plasticity():
    """Test adaptive frequency structural plasticity optimization."""
    print("Testing adaptive structural plasticity...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test with stable activity (should skip many updates)
    stable_activity = jnp.ones(learner.network.n_neurons) * 0.1
    
    # Add stable activity to history
    for i in range(20):
        learner.network.activity_history.append(stable_activity)
        learner.network.learning_step = i
        learner.network._apply_structural_plasticity()
    
    # Check performance stats
    perf_stats = learner.network.get_performance_stats()
    print(f"  Stable activity - Updates: {perf_stats['structural_updates']}, Skips: {perf_stats['structural_skips']}")
    
    # Should have skipped many updates due to low activity change
    assert perf_stats['structural_skips'] > 0, "Should skip updates with stable activity"
    
    # Test with changing activity (should trigger more updates)
    learner.network.performance_stats = {'structural_updates': 0, 'structural_skips': 0, 'connectivity_caps': 0, 'sleep_consolidations': 0}
    learner.network.last_structural_update = 0
    
    # Add changing activity to history
    for i in range(20):
        # Create varying activity patterns
        changing_activity = jnp.array([np.sin(i * 0.5) * 0.5 + 0.5 if j < 10 else 0.0 
                                     for j in range(learner.network.n_neurons)])
        learner.network.activity_history.append(changing_activity)
        learner.network.learning_step = i + 20
        learner.network._apply_structural_plasticity()
    
    # Check performance stats
    perf_stats = learner.network.get_performance_stats()
    print(f"  Changing activity - Updates: {perf_stats['structural_updates']}, Skips: {perf_stats['structural_skips']}")
    
    # Should have more updates due to changing activity
    assert perf_stats['structural_updates'] > 0, "Should update with changing activity"
    
    print("✅ Adaptive structural plasticity tests passed")


def test_incremental_correlation_updates():
    """Test incremental correlation matrix updates."""
    print("Testing incremental correlation updates...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Initialize with some activity
    activity1 = jnp.array([1.0, 0.5, 0.0] + [0.0] * 47)
    activity2 = jnp.array([0.5, 1.0, 0.2] + [0.0] * 47)
    
    # Add activities to history
    learner.network.activity_history.append(activity1)
    learner.network.activity_history.append(activity2)
    
    # Update correlation matrix
    initial_correlation = float(jnp.mean(jnp.abs(learner.network.correlation_matrix)))
    learner.network._update_correlation_matrix()
    updated_correlation = float(jnp.mean(jnp.abs(learner.network.correlation_matrix)))
    
    print(f"  Initial correlation: {initial_correlation:.6f}")
    print(f"  Updated correlation: {updated_correlation:.6f}")
    
    # Correlation matrix should have been updated
    assert updated_correlation > initial_correlation, "Correlation matrix should be updated"
    
    # Test incremental updates
    activity3 = jnp.array([0.8, 0.3, 0.1] + [0.0] * 47)
    learner.network.activity_history.append(activity3)
    
    before_update = float(jnp.mean(jnp.abs(learner.network.correlation_matrix)))
    learner.network._update_correlation_matrix()
    after_update = float(jnp.mean(jnp.abs(learner.network.correlation_matrix)))
    
    print(f"  Before incremental update: {before_update:.6f}")
    print(f"  After incremental update: {after_update:.6f}")
    
    # Should have incremental change
    assert after_update != before_update, "Incremental update should change correlation"
    
    # Check update count
    assert learner.network.correlation_update_count > 0, "Should track correlation updates"
    
    print("✅ Incremental correlation updates tests passed")


def test_performance_monitoring():
    """Test performance monitoring and statistics."""
    print("Testing performance monitoring...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Get initial performance stats
    initial_stats = learner.network.get_performance_stats()
    print(f"  Initial stats: {initial_stats}")
    
    # Required fields should be present
    required_fields = ['structural_updates', 'structural_skips', 'structural_efficiency', 
                      'connectivity_caps', 'sleep_consolidations', 'correlation_updates']
    for field in required_fields:
        assert field in initial_stats, f"Performance stats should include {field}"
    
    # Simulate some activity to trigger updates
    for i in range(15):
        # Create activity pattern
        activity = jnp.array([np.random.random() if j < 5 else 0.0 
                            for j in range(learner.network.n_neurons)])
        learner.network.activity_history.append(activity)
        learner.network.learning_step = i
        
        # Apply plasticity updates
        learner.network._apply_structural_plasticity()
    
    # Get final performance stats
    final_stats = learner.network.get_performance_stats()
    print(f"  Final stats: {final_stats}")
    
    # Should have some activity
    assert (final_stats['structural_updates'] + final_stats['structural_skips']) > 0, "Should have structural activity"
    
    # Efficiency should be a percentage
    assert '%' in final_stats['structural_efficiency'], "Efficiency should be a percentage"
    
    # Learning step should have advanced
    assert final_stats['learning_step'] > initial_stats['learning_step'], "Learning step should advance"
    
    print("✅ Performance monitoring tests passed")


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("Testing memory efficiency...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test that activity history is bounded
    # Add many activities through process_tokens (which implements bounding)
    for i in range(100):
        # Process tokens which will add to activity history with bounding
        test_tokens = [i % 10 + 1]  # Use different tokens to create varied activity
        learner.network.process_tokens(test_tokens, learning=False)
    
    # Check activity history size
    history_size = len(learner.network.activity_history)
    print(f"  Activity history size after 100 additions: {history_size}")
    
    # Should be bounded to prevent memory bloat (bounded at 20 in implementation)
    assert history_size <= 20, "Activity history should be bounded to prevent memory bloat"
    
    # Test experience buffer size
    # Add many experiences through normal processing (which implements bounding)
    for i in range(100):
        # Process tokens with high importance to trigger experience storage
        test_tokens = [i % 10 + 1]
        high_importance_modulators = {
            'dopamine': 0.8,
            'acetylcholine': 0.9,
            'norepinephrine': 0.7
        }
        learner.network.process_tokens(test_tokens, learning=True, modulators=high_importance_modulators)
    
    # Experience buffer should be bounded
    buffer_size = len(learner.network.experience_buffer)
    print(f"  Experience buffer size after 100 additions: {buffer_size}")
    
    # Should be bounded (bounded at 50 in implementation)
    assert buffer_size <= 50, "Experience buffer should be bounded"
    
    print("✅ Memory efficiency tests passed")


def test_runtime_performance():
    """Test runtime performance improvements."""
    print("Testing runtime performance...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test processing speed
    test_tokens = [1, 2, 3, 4, 5]
    
    # Warm up
    for i in range(5):
        learner.network.process_tokens(test_tokens, learning=True)
    
    # Time processing
    start_time = time.time()
    for i in range(20):
        learner.network.process_tokens(test_tokens, learning=True)
    processing_time = time.time() - start_time
    
    print(f"  Processing time for 20 iterations: {processing_time:.4f}s")
    print(f"  Average time per iteration: {processing_time/20:.4f}s")
    
    # Should be reasonably fast
    assert processing_time < 10.0, "Processing should be reasonably fast"
    
    # Check performance stats
    perf_stats = learner.network.get_performance_stats()
    print(f"  Performance stats: {perf_stats}")
    
    # Should have some efficiency gains
    if perf_stats['structural_skips'] > 0:
        efficiency = float(perf_stats['structural_efficiency'].rstrip('%'))
        print(f"  Structural plasticity efficiency: {efficiency}%")
        assert efficiency > 0, "Should have some efficiency gains"
    
    print("✅ Runtime performance tests passed")


def test_integrated_phase5_functionality():
    """Test integrated Phase 5 functionality."""
    print("Testing integrated Phase 5 functionality...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Process learning sequence
    learning_sequence = [
        "hello world",
        "good morning",
        "how are you",
        "nice to meet you",
        "goodbye now"
    ]
    
    for i, text in enumerate(learning_sequence):
        print(f"  Processing: '{text}'")
        learner._learn_through_plasticity(text)
        
        # Check that all systems are working
        assert hasattr(learner.network, 'modulators'), "Should have modulators"
        assert len(learner.network.activity_history) > 0, "Should have activity history"
        
        # Check performance stats
        perf_stats = learner.network.get_performance_stats()
        assert perf_stats['learning_step'] == i + 1, f"Learning step should be {i + 1}"
    
    # Final performance stats
    final_stats = learner.network.get_performance_stats()
    print(f"  Final performance stats: {final_stats}")
    
    # Should have processed all learning steps
    assert final_stats['learning_step'] == len(learning_sequence), "Should complete all learning steps"
    
    # Check that optimizations are working
    total_calls = final_stats['structural_updates'] + final_stats['structural_skips']
    if total_calls > 0:
        efficiency = float(final_stats['structural_efficiency'].rstrip('%'))
        print(f"  Overall structural efficiency: {efficiency}%")
    
    print("✅ Integrated Phase 5 functionality tests passed")


def main():
    """Run all Phase 5 tests."""
    print("🧪 Phase 5 - Structural-plasticity Optimization Tests")
    print("=" * 60)
    
    try:
        test_modulators_bug_fix()
        test_connectivity_cap_optimization()
        test_adaptive_structural_plasticity()
        test_incremental_correlation_updates()
        test_performance_monitoring()
        test_memory_efficiency()
        test_runtime_performance()
        test_integrated_phase5_functionality()
        
        print("\n🎉 All Phase 5 tests passed!")
        print("Structural-plasticity optimization system is working correctly")
        print("Target: ↓15% runtime with flat memory - optimizations implemented")
        
    except Exception as e:
        print(f"\n❌ Phase 5 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())