#!/usr/bin/env python3
"""
Autonomous Learning Demo

This runs a fully autonomous conversation between the Hebbian network and LLM teacher.
No human in the loop - just watch the network learn through conversation!

The LLM teacher acts like a parent teaching a baby to talk, and the Hebbian network
learns in real-time through every exchange.
"""

import argparse
import time
import sys
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learner import ContinualLearner, LearnerConfig


def print_conversation_turn(turn_num, teacher_msg, learner_response, teacher_feedback, 
                          processing_time, vocab_size, total_interactions):
    """Print a formatted conversation turn."""
    print(f"\n{'='*60}")
    print(f"Turn {turn_num}")
    print(f"{'='*60}")
    print(f"🎓 Teacher: {teacher_msg}")
    print(f"🧠 Learner: {learner_response} (processed in {processing_time:.2f}s)")
    print(f"📝 Teacher feedback: {teacher_feedback}")
    print(f"📊 Vocab: {vocab_size} patterns | Interactions: {total_interactions}")


def run_autonomous_conversation(learner: ContinualLearner, 
                              max_turns: int = 50,
                              pause_between_turns: float = 1.0,
                              show_stats_every: int = 10):
    """Run autonomous conversation between teacher and learner."""
    
    print("🚀 Starting autonomous conversation...")
    print("   The Hebbian network will learn from the LLM teacher")
    print("   Watch the vocabulary grow and responses improve!")
    print()
    
    # Start the conversation
    teacher_opening = learner.start_conversation()
    print(f"🎓 Teacher opens: {teacher_opening}")
    print()
    
    turn_count = 0
    
    try:
        for turn in range(max_turns):
            turn_count += 1
            
            # Teacher's current message (either opening or previous feedback)
            if turn == 0:
                teacher_message = teacher_opening
            else:
                # Get the last teacher message from conversation history
                teacher_message = learner.current_conversation[-1]["text"]
            
            # Learner processes and responds (this is where learning happens!)
            print(f"🧠 [Turn {turn_count}] Learning from: '{teacher_message}'", end="", flush=True)
            start_time = time.time()
            
            # Generate learner response based on teacher's message
            learner_response = learner._generate_response(teacher_message)
            
            # Record learner's response
            learner.current_conversation.append({"role": "learner", "text": learner_response})
            learner.total_interactions += 1
            
            # Learn from generating this response
            learner._learn_from_text(learner_response, is_input=False)
            
            # Get teacher feedback
            teacher_feedback = learner.teacher.respond_to_student(learner_response)
            learner.current_conversation.append({"role": "teacher", "text": teacher_feedback})
            
            # Learn from teacher feedback
            learner._learn_from_feedback(learner_response, teacher_feedback)
            
            processing_time = time.time() - start_time
            
            # Display the conversation turn
            print_conversation_turn(
                turn_count, teacher_message, learner_response, teacher_feedback,
                processing_time, learner.tokenizer.get_vocab_size(), learner.total_interactions
            )
            
            # Periodic adaptation and stats
            if turn_count % 5 == 0:
                learner._adapt_network()
                
            if turn_count % show_stats_every == 0:
                stats = learner.get_learning_stats()
                print(f"\n📈 PROGRESS REPORT (Turn {turn_count}):")
                print(f"   📚 Vocabulary: {stats['vocabulary_size']} patterns")
                print(f"   🧠 Network: {stats['network_neurons']} neurons") 
                print(f"   🎯 Teacher stage: {stats['teacher_stage']}")
                print(f"   💭 Learned associations: {stats['learned_associations']}")
                
                # Show most used patterns
                tokenizer_stats = stats['tokenizer_stats']
                if 'most_used_patterns' in tokenizer_stats:
                    most_used = tokenizer_stats['most_used_patterns'][:5]
                    print(f"   🔤 Top patterns: {most_used}")
            
            # Pause between turns for readability
            if pause_between_turns > 0:
                time.sleep(pause_between_turns)
            
            # Check for conversation end signals
            if ("bye" in teacher_feedback.lower() or 
                "goodbye" in teacher_feedback.lower() or
                learner.teacher.current_stage == "conversation" and turn_count > 30):
                print(f"\n🎉 Natural conversation end reached at turn {turn_count}")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Autonomous learning stopped at turn {turn_count}")
    
    return turn_count


def main():
    parser = argparse.ArgumentParser(description='Run autonomous conversation between Hebbian network and LLM teacher')
    parser.add_argument('--turns', type=int, default=50, help='Maximum conversation turns')
    parser.add_argument('--pause', type=float, default=1.0, help='Pause between turns (seconds)')
    parser.add_argument('--stats-every', type=int, default=10, help='Show stats every N turns')
    parser.add_argument('--load-memory', action='store_true', help='Load previous learning memory')
    parser.add_argument('--save-progress', action='store_true', help='Save progress after conversation')
    parser.add_argument('--teacher-url', default='http://localhost:1234/v1/chat/completions',
                       help='Teacher LLM API URL')
    parser.add_argument('--teacher-model', default='gemma-3-27b-it-qat',
                       help='Teacher LLM model name')
    parser.add_argument('--vocab-size', type=int, default=50, 
                       help='Initial vocabulary size')
    parser.add_argument('--fast', action='store_true', 
                       help='Fast mode: no pauses, minimal output')
    
    args = parser.parse_args()
    
    print("🤖 Autonomous Hebbian-LLM Learning")
    print("="*40)
    print()
    print("This demo runs a fully autonomous conversation where:")
    print("• LLM teacher acts like a parent teaching a baby")  
    print("• Hebbian network learns in real-time through conversation")
    print("• No human intervention needed - just watch it learn!")
    print()
    
    # Test teacher connection
    print("Testing teacher connection...")
    try:
        import requests
        response = requests.get(args.teacher_url.replace('/v1/chat/completions', '/v1/models'), 
                               timeout=5)
        if response.status_code == 200:
            print("✅ Teacher LLM is running")
        else:
            print("❌ Teacher LLM connection failed")
            return 1
    except Exception as e:
        print(f"❌ Could not connect to teacher: {e}")
        return 1
    
    # Configure learner
    config = LearnerConfig(
        teacher_api_url=args.teacher_url,
        teacher_model=args.teacher_model,
        initial_vocab_size=args.vocab_size,
        max_vocab_size=2000,
        save_interval=20 if args.save_progress else 999999
    )
    
    if args.fast:
        args.pause = 0
        args.stats_every = 25
    
    # Initialize learner
    print("Initializing continual learner...")
    learner = ContinualLearner(config)
    
    # Load memory if requested
    if args.load_memory:
        print("Loading previous learning memory...")
        learner.load_memory()
        stats = learner.get_learning_stats()
        print(f"Resumed with {stats['conversations']} previous conversations")
    
    # Show initial state
    initial_stats = learner.get_learning_stats()
    print(f"\n🎯 Starting Configuration:")
    print(f"   Vocabulary: {initial_stats['vocabulary_size']} patterns")
    print(f"   Network: {initial_stats['network_neurons']} neurons")
    print(f"   Teacher stage: {initial_stats['teacher_stage']}")
    print(f"   Max turns: {args.turns}")
    
    # Run autonomous conversation
    total_turns = run_autonomous_conversation(
        learner, 
        max_turns=args.turns,
        pause_between_turns=args.pause,
        show_stats_every=args.stats_every
    )
    
    # Final statistics
    final_stats = learner.get_learning_stats()
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Turns completed: {total_turns}")
    print(f"   Total interactions: {final_stats['total_interactions']}")
    print(f"   Final vocabulary: {final_stats['vocabulary_size']} patterns")
    print(f"   Network size: {final_stats['network_neurons']} neurons")
    print(f"   Teacher stage: {final_stats['teacher_stage']}")
    print(f"   Learned associations: {final_stats['learned_associations']}")
    
    # Show vocabulary growth
    vocab_growth = final_stats['vocabulary_size'] - initial_stats['vocabulary_size']
    if vocab_growth > 0:
        print(f"   📈 Vocabulary grew by {vocab_growth} patterns!")
    
    # Save progress if requested
    if args.save_progress:
        learner._save_memory()
        print(f"   💾 Progress saved for future sessions")
    
    print(f"\n✨ Autonomous learning session complete!")
    print(f"The network learned purely through conversation - no pre-training!")
    
    return 0


if __name__ == "__main__":
    exit(main())