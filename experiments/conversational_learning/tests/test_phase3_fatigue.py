#!/usr/bin/env python3
"""
Unit tests for Phase 3 - Adenosine fatigue + Short-Term Plasticity (STP) implementation.

Tests the fatigue system and STP buffers.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def test_adenosine_fatigue_buildup():
    """Test adenosine fatigue buildup and decay."""
    print("Testing adenosine fatigue buildup...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Initial fatigue should be zero
    initial_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
    print(f"  Initial fatigue: {initial_fatigue:.3f}")
    assert initial_fatigue == 0.0, "Initial fatigue should be zero"
    
    # Simulate repeated activity to build fatigue
    test_tokens = [1, 2, 3]
    fatigue_levels = []
    
    for i in range(10):
        # Process tokens to generate activity
        learner.network.process_tokens(test_tokens, learning=True)
        
        # Check fatigue level
        avg_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
        fatigue_levels.append(avg_fatigue)
        
        if i % 3 == 0:
            print(f"  After {i+1} activities: fatigue = {avg_fatigue:.3f}")
    
    # Verify fatigue increases with activity
    assert fatigue_levels[-1] > fatigue_levels[0], "Fatigue should increase with activity"
    assert fatigue_levels[5] > fatigue_levels[2], "Fatigue should continue increasing"
    
    print("✅ Adenosine fatigue buildup tests passed")


def test_stp_buffers():
    """Test Short-Term Plasticity buffers."""
    print("Testing STP buffers...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Initial STP state
    initial_facilitation = float(jnp.mean(learner.network.stp_facilitation))
    initial_depression = float(jnp.mean(learner.network.stp_depression))
    initial_buffer = float(jnp.mean(jnp.abs(learner.network.stp_buffer)))
    
    print(f"  Initial - Facilitation: {initial_facilitation:.3f}, Depression: {initial_depression:.3f}, Buffer: {initial_buffer:.3f}")
    
    # Process tokens to trigger STP changes
    test_tokens = [1, 2, 3]
    
    for i in range(5):
        learner.network.process_tokens(test_tokens, learning=True)
    
    # Check STP state after activity
    final_facilitation = float(jnp.mean(learner.network.stp_facilitation))
    final_depression = float(jnp.mean(learner.network.stp_depression))
    final_buffer = float(jnp.mean(jnp.abs(learner.network.stp_buffer)))
    
    print(f"  Final - Facilitation: {final_facilitation:.3f}, Depression: {final_depression:.3f}, Buffer: {final_buffer:.3f}")
    
    # Verify STP changes occurred
    assert final_buffer > initial_buffer, "STP buffer should accumulate changes"
    
    print("✅ STP buffer tests passed")


def test_adenosine_sleep_trigger():
    """Test adenosine-clearing sleep trigger."""
    print("Testing adenosine sleep trigger...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Manually set high fatigue to trigger sleep
    learner.network.neuron_fatigue = jnp.ones(learner.network.n_neurons) * 0.8  # High fatigue
    
    # Set some STP buffer content
    learner.network.stp_buffer = jnp.ones((learner.network.n_neurons, learner.network.n_neurons)) * 0.1
    
    # Record pre-sleep state
    pre_sleep_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
    pre_sleep_stp = float(jnp.mean(jnp.abs(learner.network.stp_buffer)))
    
    print(f"  Pre-sleep - Fatigue: {pre_sleep_fatigue:.3f}, STP: {pre_sleep_stp:.3f}")
    
    # Trigger sleep
    sleep_occurred = learner._trigger_adenosine_sleep()
    
    # Check post-sleep state
    post_sleep_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
    post_sleep_stp = float(jnp.mean(jnp.abs(learner.network.stp_buffer)))
    
    print(f"  Post-sleep - Fatigue: {post_sleep_fatigue:.3f}, STP: {post_sleep_stp:.3f}")
    
    # Verify sleep effects
    assert sleep_occurred, "Sleep should have been triggered"
    assert post_sleep_fatigue < pre_sleep_fatigue, "Fatigue should decrease after sleep"
    assert post_sleep_stp < pre_sleep_stp, "STP buffer should be cleared after sleep"
    
    print("✅ Adenosine sleep trigger tests passed")


def test_stp_consolidation():
    """Test STP to LTP consolidation during sleep."""
    print("Testing STP consolidation...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Set initial state
    initial_weights = learner.network.synaptic_weights.copy()
    
    # Set STP buffer with some changes
    stp_changes = jnp.ones((learner.network.n_neurons, learner.network.n_neurons)) * 0.1
    learner.network.stp_buffer = stp_changes
    
    # Set high fatigue to trigger sleep
    learner.network.neuron_fatigue = jnp.ones(learner.network.n_neurons) * 0.8
    
    # Trigger sleep (which should consolidate STP)
    sleep_occurred = learner._trigger_adenosine_sleep()
    
    # Check if weights changed due to STP consolidation
    final_weights = learner.network.synaptic_weights
    weight_change = float(jnp.mean(jnp.abs(final_weights - initial_weights)))
    
    print(f"  STP consolidation caused weight change: {weight_change:.6f}")
    
    # Verify consolidation occurred
    assert sleep_occurred, "Sleep should have occurred"
    assert weight_change > 0, "Weights should change due to STP consolidation"
    
    print("✅ STP consolidation tests passed")


def test_fatigue_modulation():
    """Test that fatigue reduces learning rates."""
    print("Testing fatigue modulation...")
    
    config = PlasticLearnerConfig(n_neurons=100, vocab_size=20)
    learner = PlasticContinualLearner(config)
    
    # Test with low fatigue
    learner.network.neuron_fatigue = jnp.zeros(learner.network.n_neurons)  # No fatigue
    initial_weights_low = learner.network.synaptic_weights.copy()
    
    test_tokens = [1, 2, 3]
    modulators = {'adenosine': 0.0}  # No adenosine
    learner.network.process_tokens(test_tokens, learning=True, modulators=modulators)
    
    weight_change_low = float(jnp.sum(jnp.abs(learner.network.synaptic_weights - initial_weights_low)))
    
    # Reset and test with high fatigue
    learner.network.synaptic_weights = initial_weights_low.copy()
    learner.network.neuron_fatigue = jnp.ones(learner.network.n_neurons) * 0.8  # High fatigue
    
    modulators = {'adenosine': 0.8}  # High adenosine
    learner.network.process_tokens(test_tokens, learning=True, modulators=modulators)
    
    weight_change_high = float(jnp.sum(jnp.abs(learner.network.synaptic_weights - initial_weights_low)))
    
    print(f"  Weight change - Low fatigue: {weight_change_low:.6f}, High fatigue: {weight_change_high:.6f}")
    
    # Verify fatigue reduces learning
    assert weight_change_high < weight_change_low, "High fatigue should reduce learning"
    
    print("✅ Fatigue modulation tests passed")


def test_integrated_phase3_workflow():
    """Test complete Phase 3 workflow integration."""
    print("Testing integrated Phase 3 workflow...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Simulate learning sessions that build fatigue
    initial_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
    
    # Process several learning rounds
    for i in range(15):  # More than sleep cycle threshold
        learner._learn_through_plasticity(f"test pattern {i}")
        
        # Check if sleep was triggered
        if len(learner.fatigue_events) > 0:
            print(f"  Sleep event triggered at interaction {i}")
            break
    
    # Verify system integration
    final_fatigue = float(jnp.mean(learner.network.neuron_fatigue))
    
    print(f"  Initial fatigue: {initial_fatigue:.3f}, Final fatigue: {final_fatigue:.3f}")
    print(f"  Fatigue events: {len(learner.fatigue_events)}")
    
    # System should have managed fatigue
    assert final_fatigue < 0.8, "System should have managed fatigue through sleep"
    
    print("✅ Integrated Phase 3 workflow tests passed")


def test_memory_persistence():
    """Test that Phase 3 state is saved and loaded correctly."""
    print("Testing memory persistence...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner1 = PlasticContinualLearner(config)
    
    # Build up some Phase 3 state
    learner1.network.neuron_fatigue = jnp.ones(learner1.network.n_neurons) * 0.5
    learner1.sleep_cycle_counter = 7
    learner1.fatigue_events.append({'test': 'event'})
    
    # Save state
    learner1._save_memory()
    
    # Create new learner and load state
    learner2 = PlasticContinualLearner(config)
    learner2.load_memory()
    
    # Verify state was restored
    assert learner2.sleep_cycle_counter == 7, "Sleep cycle counter should be restored"
    assert len(learner2.fatigue_events) > 0, "Fatigue events should be restored"
    
    print("✅ Memory persistence tests passed")


def main():
    """Run all Phase 3 tests."""
    print("🧪 Phase 3 - Adenosine Fatigue + STP Tests")
    print("=" * 50)
    
    try:
        test_adenosine_fatigue_buildup()
        test_stp_buffers()
        test_adenosine_sleep_trigger()
        test_stp_consolidation()
        test_fatigue_modulation()
        test_integrated_phase3_workflow()
        test_memory_persistence()
        
        print("\n🎉 All Phase 3 tests passed!")
        print("Adenosine fatigue + STP system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Phase 3 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())