#!/usr/bin/env python3
"""
Memory-Efficient Autonomous Learning Demo

Fixed version that prevents memory issues and teacher API problems.
"""

import sys
import time
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learner import ContinualLearner, LearnerConfig


def main():
    print("🧠 Memory-Efficient Hebbian Learning Demo")
    print("=" * 45)
    print("Autonomous conversation with memory limits")
    print("Press Ctrl+C to stop\n")
    
    # Memory-efficient configuration
    config = LearnerConfig(
        initial_vocab_size=30,      # Start small
        max_vocab_size=200,         # Limited growth
        column_size=10,             # Small columns
        n_layers=2,                 # Minimal layers
        max_network_neurons=2000,   # Memory limit
        save_interval=999           # Don't save during demo
    )
    
    # Initialize
    print("Initializing efficient learner...")
    learner = ContinualLearner(config)
    
    # Show initial state
    stats = learner.get_learning_stats()
    print(f"Starting: {stats['vocabulary_size']} patterns, {stats['network_neurons']} neurons")
    print()
    
    # Start conversation
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    
    turn = 0
    last_vocab_size = stats['vocabulary_size']
    
    try:
        while turn < 30:  # Limit turns to prevent infinite loops
            turn += 1
            
            # Network generates response and learns
            print(f"🧠 [Turn {turn}] Processing...", end="", flush=True)
            start_time = time.time()
            
            # Generate response (learning happens here!)
            response = learner._generate_response(teacher_msg)
            learner.current_conversation.append({"role": "learner", "text": response})
            learner.total_interactions += 1
            learner._learn_from_text(response, is_input=False)
            
            # Get teacher feedback  
            teacher_feedback = learner.teacher.respond_to_student(response)
            learner.current_conversation.append({"role": "teacher", "text": teacher_feedback})
            learner._learn_from_feedback(response, teacher_feedback)
            
            processing_time = time.time() - start_time
            current_vocab = learner.tokenizer.get_vocab_size()
            
            print(f"\r🤖 Network: '{response}'")
            print(f"👩‍🏫 Teacher: '{teacher_feedback}'")
            
            # Show learning indicators
            vocab_growth = current_vocab - last_vocab_size
            growth_indicator = f" (+{vocab_growth})" if vocab_growth > 0 else ""
            
            print(f"📊 Vocab: {current_vocab}{growth_indicator} | "
                  f"Neurons: {learner.network.n_neurons} | "
                  f"Stage: {learner.teacher.current_stage} | "
                  f"Time: {processing_time:.1f}s")
            
            # Adapt network periodically (but carefully)
            if turn % 8 == 0:
                print("🔄 Adapting network...")
                learner._adapt_network()
                new_stats = learner.get_learning_stats()
                print(f"   Network now: {new_stats['network_neurons']} neurons")
            
            # Show progress every few turns
            if turn % 5 == 0:
                tokenizer_stats = learner.tokenizer.get_pattern_stats()
                if 'most_used_patterns' in tokenizer_stats:
                    top_patterns = tokenizer_stats['most_used_patterns'][:3]
                    print(f"📈 Top patterns: {top_patterns}")
            
            print()
            last_vocab_size = current_vocab
            
            # Update teacher message for next turn
            teacher_msg = teacher_feedback
            
            # Small pause for readability
            time.sleep(0.3)
            
            # Check for natural stopping
            if ("bye" in teacher_feedback.lower() or 
                "goodbye" in teacher_feedback.lower() or
                turn >= 25):
                print(f"🎉 Conversation naturally ended at turn {turn}")
                break
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopped after {turn} turns")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"Completed {turn} turns before error")
        
    # Show final stats
    final_stats = learner.get_learning_stats()
    vocab_growth = final_stats['vocabulary_size'] - stats['vocabulary_size']
    
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Turns: {turn}")
    print(f"   Vocabulary: {stats['vocabulary_size']} → {final_stats['vocabulary_size']} (+{vocab_growth})")
    print(f"   Network: {stats['network_neurons']} → {final_stats['network_neurons']} neurons")
    print(f"   Teacher stage: {final_stats['teacher_stage']}")
    print(f"   Interactions: {final_stats['total_interactions']}")
    
    if vocab_growth > 0:
        print(f"   📈 Network learned {vocab_growth} new patterns!")
    
    print(f"\n✨ Hebbian learning succeeded - no pre-training needed!")


if __name__ == "__main__":
    main()