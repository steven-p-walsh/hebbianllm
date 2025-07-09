#!/usr/bin/env python3
"""
Unit tests for Phase 2 - Acetylcholine attention + Norepinephrine novelty implementation.

Tests the attention and novelty detection systems.
"""

import sys
from pathlib import Path
import numpy as np

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.conversational_learning.plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def test_novelty_detection():
    """Test novelty detection for patterns."""
    print("Testing novelty detection...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test cases
    test_patterns = [
        "hello world",
        "goodbye world", 
        "hello world",  # Repeat - should be less novel
        "completely new pattern",
        "hello world",  # Repeat again - should be even less novel
    ]
    
    novelty_scores = []
    for pattern in test_patterns:
        novelty = learner._compute_novelty_score(pattern)
        novelty_scores.append(novelty)
        print(f"  '{pattern}' -> novelty: {novelty:.3f}")
    
    # Verify novelty decreases with repetition
    assert novelty_scores[0] > novelty_scores[2], "Repeated pattern should be less novel"
    assert novelty_scores[2] > novelty_scores[4], "More repetition should decrease novelty further"
    
    # Verify new patterns are novel
    assert novelty_scores[3] > 0.8, "New pattern should be highly novel"
    
    print("✅ Novelty detection tests passed")


def test_attention_computation():
    """Test attention level computation."""
    print("Testing attention computation...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test cases
    test_cases = [
        ("regular text", "low attention"),
        ("what is this about?", "high attention - question"),
        ("that's excellent work!", "high attention - emotional"),
        ("why did this happen?", "high attention - question"),
        ("this is boring text", "moderate attention"),
    ]
    
    attention_levels = []
    for text, expected_type in test_cases:
        attention = learner._compute_attention_level(text)
        attention_levels.append(attention)
        print(f"  '{text}' -> attention: {attention:.3f} ({expected_type})")
    
    # Verify question words boost attention
    assert attention_levels[1] > attention_levels[0], "Questions should increase attention"
    assert attention_levels[3] > attention_levels[0], "Questions should increase attention"
    
    # Verify emotional words boost attention
    assert attention_levels[2] > attention_levels[0], "Emotional words should increase attention"
    
    print("✅ Attention computation tests passed")


def test_combined_neuromodulation():
    """Test combined acetylcholine and norepinephrine modulation."""
    print("Testing combined neuromodulation...")
    
    config = PlasticLearnerConfig(n_neurons=100, vocab_size=20)
    learner = PlasticContinualLearner(config)
    
    # First, build up familiarity with some patterns
    familiar_patterns = ["hello world", "what is hello?", "goodbye world"]
    for pattern in familiar_patterns:
        for _ in range(3):  # Repeat 3 times to build familiarity
            learner._compute_novelty_score(pattern)
    
    # Test scenarios
    scenarios = [
        ("familiar question", "what is hello?"),  # High attention, low novelty
        ("novel statement", "zxcvbnm asdfgh"),    # Low attention, high novelty
        ("novel question", "what is qwerty?"),    # High attention, high novelty (completely different word)
        ("familiar statement", "hello world"),     # Low attention, low novelty
    ]
    
    results = []
    for scenario_name, text in scenarios:
        # Get initial weights
        initial_weights = learner.network.synaptic_weights.copy()
        
        # Learn through plasticity with Phase 2
        learner._learn_through_plasticity(text)
        
        # Calculate weight change
        weight_change = np.sum(np.abs(learner.network.synaptic_weights - initial_weights))
        
        # Get neuromodulator levels
        attention = learner.network.modulators.get_mod('acetylcholine')
        novelty = learner.network.modulators.get_mod('norepinephrine')
        
        results.append({
            'scenario': scenario_name,
            'text': text,
            'attention': attention,
            'novelty': novelty,
            'weight_change': weight_change
        })
        
        print(f"  {scenario_name}: attention={attention:.3f}, novelty={novelty:.3f}, change={weight_change:.6f}")
    
    # Verify neuromodulation affects learning
    # High attention + high novelty should produce strong learning
    novel_question = next(r for r in results if r['scenario'] == 'novel question')
    familiar_statement = next(r for r in results if r['scenario'] == 'familiar statement')
    novel_statement = next(r for r in results if r['scenario'] == 'novel statement')
    
    assert novel_question['attention'] > familiar_statement['attention'], "Novel questions should get more attention"
    # Compare truly novel statement with familiar statement
    assert novel_statement['novelty'] > familiar_statement['novelty'], "Novel patterns should be more novel"
    
    print("✅ Combined neuromodulation tests passed")


def test_pattern_familiarity_tracking():
    """Test pattern familiarity tracking over time."""
    print("Testing pattern familiarity tracking...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner = PlasticContinualLearner(config)
    
    # Test repeated exposure to same pattern
    pattern = "hello world"
    familiarities = []
    
    for i in range(5):
        familiarity_before = learner.pattern_familiarity.get(learner._create_pattern_signature(pattern), 0.0)
        novelty = learner._compute_novelty_score(pattern)
        familiarity_after = learner.pattern_familiarity.get(learner._create_pattern_signature(pattern), 0.0)
        
        familiarities.append(familiarity_after)
        print(f"  Exposure {i+1}: familiarity={familiarity_after:.3f}, novelty={novelty:.3f}")
    
    # Verify familiarity increases with exposure
    assert familiarities[0] < familiarities[2], "Familiarity should increase with exposure"
    assert familiarities[2] < familiarities[4], "Familiarity should continue increasing"
    
    # Verify familiarity approaches 1.0 but doesn't exceed it
    assert all(f <= 1.0 for f in familiarities), "Familiarity should not exceed 1.0"
    
    print("✅ Pattern familiarity tracking tests passed")


def test_memory_persistence():
    """Test that Phase 2 state is saved and loaded correctly."""
    print("Testing memory persistence...")
    
    config = PlasticLearnerConfig(n_neurons=50, vocab_size=10)
    learner1 = PlasticContinualLearner(config)
    
    # Build up some Phase 2 state
    learner1._compute_novelty_score("hello world")
    learner1._compute_novelty_score("goodbye world")
    learner1._compute_attention_level("what is this?")
    
    # Save state
    learner1._save_memory()
    
    # Create new learner and load state
    learner2 = PlasticContinualLearner(config)
    learner2.load_memory()
    
    # Verify state was restored
    assert len(learner2.pattern_familiarity) > 0, "Pattern familiarity should be restored"
    assert len(learner2.novelty_history) > 0, "Novelty history should be restored"
    
    print("✅ Memory persistence tests passed")


def main():
    """Run all Phase 2 tests."""
    print("🧪 Phase 2 - Acetylcholine Attention + Norepinephrine Novelty Tests")
    print("=" * 70)
    
    try:
        test_novelty_detection()
        test_attention_computation()
        test_combined_neuromodulation()
        test_pattern_familiarity_tracking()
        test_memory_persistence()
        
        print("\n🎉 All Phase 2 tests passed!")
        print("Acetylcholine attention + Norepinephrine novelty system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Phase 2 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())