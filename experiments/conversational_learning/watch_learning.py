#!/usr/bin/env python3
"""
Watch the Network Learn - Simple Autonomous Demo

Simplified version that just shows the core learning happening.
Pure autonomous conversation between Hebbian network and LLM teacher.
"""

import sys
import time
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learner import ContinualLearner, LearnerConfig


def main():
    print("🧠 Watching Hebbian Network Learn Through Conversation")
    print("=" * 55)
    print("Autonomous conversation between network and LLM teacher")
    print("Press Ctrl+C to stop\n")
    
    # Simple configuration
    config = LearnerConfig(
        initial_vocab_size=30,    # Start very small
        max_vocab_size=500,       # Don't grow too large
        save_interval=999         # Don't save during demo
    )
    
    # Initialize
    learner = ContinualLearner(config)
    
    # Start conversation
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    
    turn = 0
    try:
        while True:
            turn += 1
            
            # Network generates response and learns
            print(f"🧠 [Turn {turn}] Thinking...", end="", flush=True)
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
            vocab_size = learner.tokenizer.get_vocab_size()
            
            print(f"\r🤖 Network: {response}")
            print(f"👩‍🏫 Teacher: {teacher_feedback}")
            print(f"📊 Vocab: {vocab_size} | Time: {processing_time:.1f}s | Stage: {learner.teacher.current_stage}")
            print()
            
            # Adapt network periodically
            if turn % 5 == 0:
                learner._adapt_network()
                print(f"🔄 Network adapted (turn {turn})")
                print()
            
            # Update teacher message for next turn
            teacher_msg = teacher_feedback
            
            # Small pause for readability
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopped after {turn} turns")
        
        # Show final stats
        stats = learner.get_learning_stats()
        print(f"📈 Final vocabulary: {stats['vocabulary_size']} patterns")
        print(f"🎯 Teacher stage: {stats['teacher_stage']}")
        print(f"💭 Total interactions: {stats['total_interactions']}")
        
        print("\n✨ The network learned purely through conversation!")


if __name__ == "__main__":
    main()