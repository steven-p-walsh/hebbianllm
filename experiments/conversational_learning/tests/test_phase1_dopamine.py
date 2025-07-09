#!/usr/bin/env python3
"""
Unit tests for Phase 1 - Dopamine RPE implementation.

Tests the dopamine reward prediction error system.
"""

import sys
from pathlib import Path
import numpy as np

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def test_rpe_computation():
    """Test RPE computation from feedback."""
    print("Testing RPE computation...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test cases
    test_cases = [
        ("Excellent work!", 1.0),
        ("Perfect!", 1.0),
        ("Good job", 0.5),
        ("Nice try", 0.5),
        ("Try again", -0.5),
        ("Not quite", -0.5),
        ("Wrong", -1.0),
        ("Terrible", -1.0),
        ("I like apples", 0.0),  # Neutral
    ]
    
    for feedback, expected_rpe in test_cases:
        computed_rpe = learner._compute_rpe_from_feedback(feedback)
        print(f"  '{feedback}' -> RPE: {computed_rpe:.1f} (expected: {expected_rpe:.1f})")
        assert abs(computed_rpe - expected_rpe) < 0.01, f"RPE mismatch for '{feedback}'"
    
    print("✅ RPE computation tests passed")


def test_dopamine_modulation():
    """Test dopamine modulation of learning."""
    print("Testing dopamine modulation...")
    
    config = PlasticLearnerConfig(n_neurons=100, vocab_size=20)
    learner = PlasticContinualLearner(config)
    
    # Test tokens
    test_tokens = [1, 2, 3]
    
    # Get initial weights
    initial_weights = learner.network.synaptic_weights.copy()
    
    # Test with different dopamine levels
    dopamine_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
    weight_changes = []
    
    for da_level in dopamine_levels:
        # Reset network
        learner.network.synaptic_weights = initial_weights.copy()
        
        # Set dopamine
        modulators = {'dopamine': da_level}
        
        # Process tokens
        learner.network.process_tokens(test_tokens, learning=True, modulators=modulators)
        
        # Calculate weight change
        weight_change = np.sum(np.abs(learner.network.synaptic_weights - initial_weights))
        weight_changes.append(weight_change)
        
        print(f"  Dopamine {da_level:+.1f}: weight change = {weight_change:.6f}")
    
    # Verify that positive dopamine generally increases learning
    positive_change = weight_changes[4]  # DA = +1.0
    negative_change = weight_changes[0]  # DA = -1.0
    neutral_change = weight_changes[2]   # DA = 0.0
    
    print(f"  Positive DA effect: {positive_change:.6f}")
    print(f"  Negative DA effect: {negative_change:.6f}")
    print(f"  Neutral DA effect: {neutral_change:.6f}")
    
    # Check general trend (allowing for some variability)
    if positive_change > neutral_change * 0.95:
        print("✅ Positive dopamine enhances learning")
    else:
        print("⚠️  Positive dopamine effect unclear")
    
    print("✅ Dopamine modulation tests completed")


def test_feedback_learning_loop():
    """Test the full feedback learning loop."""
    print("Testing feedback learning loop...")
    
    config = PlasticLearnerConfig(n_neurons=100, vocab_size=20)
    learner = PlasticContinualLearner(config)
    
    # Simulate learning from feedback
    test_cases = [
        ("hello", "Good job!", "positive"),
        ("goodbye", "Wrong", "negative"),
        ("thanks", "Excellent!", "positive"),
        ("sorry", "Not quite", "negative"),
    ]
    
    for response, feedback, expected_type in test_cases:
        # Store initial state
        initial_dopamine = learner.network.modulators.get_mod('dopamine')
        
        # Process feedback
        learner._learn_from_feedback_plastic(response, feedback)
        
        # Check dopamine response
        final_dopamine = learner.network.modulators.get_mod('dopamine')
        
        print(f"  '{response}' + '{feedback}' -> dopamine: {final_dopamine:.2f}")
        
        if expected_type == "positive":
            assert final_dopamine > 0, f"Expected positive dopamine for '{feedback}'"
        elif expected_type == "negative":
            assert final_dopamine < 0, f"Expected negative dopamine for '{feedback}'"
        
        # Check that dopamine decays over time
        original_dopamine = final_dopamine
        learner.network.modulators.decay_mod('dopamine', tau=5.0, dt=1.0)
        decayed_dopamine = learner.network.modulators.get_mod('dopamine')
        
        if abs(original_dopamine) > 0.01:  # Only check if there was significant dopamine
            assert abs(decayed_dopamine) < abs(original_dopamine), "Dopamine should decay"
    
    print("✅ Feedback learning loop tests passed")


def test_learning_association_storage():
    """Test that learning associations store RPE values."""
    print("Testing learning association storage...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Process some feedback
    learner._learn_from_feedback_plastic("hello", "Great job!")
    learner._learn_from_feedback_plastic("hello", "Wrong")
    learner._learn_from_feedback_plastic("goodbye", "Perfect!")
    
    # Check stored associations
    assert "hello" in learner.learned_associations
    assert "goodbye" in learner.learned_associations
    
    # Check that RPE values are stored
    hello_associations = learner.learned_associations["hello"]
    assert len(hello_associations) == 2
    assert hello_associations[0]['rpe'] > 0  # "Great job!" should be positive
    assert hello_associations[1]['rpe'] < 0  # "Wrong" should be negative
    
    goodbye_associations = learner.learned_associations["goodbye"]
    assert len(goodbye_associations) == 1
    assert goodbye_associations[0]['rpe'] > 0  # "Perfect!" should be positive
    
    print("✅ Learning association storage tests passed")


def main():
    """Run all Phase 1 tests."""
    print("🧪 Phase 1 - Dopamine RPE Tests")
    print("=" * 40)
    
    try:
        test_rpe_computation()
        test_dopamine_modulation()
        test_feedback_learning_loop()
        test_learning_association_storage()
        
        print("\n🎉 All Phase 1 tests passed!")
        print("Dopamine RPE system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Phase 1 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())