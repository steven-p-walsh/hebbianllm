#!/usr/bin/env python3
"""
Interactive Chat with Continual Learner

This demonstrates the core concept: a network that learns as you talk to it.
No pre-training, just real-time learning through conversation.

Usage:
    python chat_with_learner.py [--load-memory] [--debug]
"""

import argparse
import sys
import time
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learner import ContinualLearner, LearnerConfig


def print_stats(learner: ContinualLearner):
    """Print current learning statistics."""
    stats = learner.get_learning_stats()
    print(f"\n📊 Learning Stats:")
    print(f"   Conversations: {stats['conversations']}")
    print(f"   Total interactions: {stats['total_interactions']}")
    print(f"   Vocabulary size: {stats['vocabulary_size']}")
    print(f"   Network neurons: {stats['network_neurons']}")
    print(f"   Teacher stage: {stats['teacher_stage']}")
    print(f"   Learned patterns: {stats['learned_associations']}")
    
    # Show most used patterns
    tokenizer_stats = stats['tokenizer_stats']
    if 'most_used_patterns' in tokenizer_stats:
        print(f"   Most used patterns: {tokenizer_stats['most_used_patterns'][:5]}")


def main():
    parser = argparse.ArgumentParser(description='Chat with a continually learning network')
    parser.add_argument('--load-memory', action='store_true', 
                       help='Load previous learning memory')
    parser.add_argument('--debug', action='store_true', 
                       help='Show debug information')
    parser.add_argument('--teacher-url', default='http://localhost:1234/v1/chat/completions',
                       help='Teacher LLM API URL')
    parser.add_argument('--teacher-model', default='gemma-3-27b-it-qat',
                       help='Teacher LLM model name')
    
    args = parser.parse_args()
    
    print("🧠 Continual Learning Chat Demo")
    print("================================")
    print()
    print("This demo shows a network that learns during every conversation.")
    print("The network starts nearly blank and grows through experience.")
    print("Unlike traditional AI, there's no pre-training - it learns as you talk!")
    print()
    
    # Test teacher connection first
    print("Testing teacher connection...")
    try:
        import requests
        response = requests.get(args.teacher_url.replace('/v1/chat/completions', '/v1/models'), 
                               timeout=5)
        if response.status_code == 200:
            print("✓ Teacher LLM is running")
        else:
            print("⚠️  Teacher LLM connection failed")
            print("   Make sure your local LLM is running on the specified URL")
            return 1
    except Exception as e:
        print(f"⚠️  Could not connect to teacher: {e}")
        print("   The demo will still work but teacher feedback will be limited")
    
    # Create learner configuration
    config = LearnerConfig(
        teacher_api_url=args.teacher_url,
        teacher_model=args.teacher_model,
        initial_vocab_size=50,  # Start very small
        max_vocab_size=1000,
        save_interval=5  # Save frequently in demo
    )
    
    # Initialize learner
    print("Initializing continual learner...")
    learner = ContinualLearner(config)
    
    # Load previous memory if requested
    if args.load_memory:
        print("Loading previous learning memory...")
        learner.load_memory()
    
    # Show initial stats
    if args.debug:
        print_stats(learner)
    
    print("\n🤖 Starting conversation...")
    print("Commands:")
    print("  'quit' or 'exit' - End conversation")
    print("  'stats' - Show learning statistics") 
    print("  'help' - Show this help")
    print("  'reset' - Start fresh (lose current memory)")
    print("  'save' - Save current progress")
    print()
    
    try:
        # Start conversation with teacher
        teacher_opening = learner.start_conversation()
        print(f"Teacher: {teacher_opening}")
        print()
        
        conversation_active = True
        while conversation_active:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'stats':
                    print_stats(learner)
                    continue
                elif user_input.lower() == 'help':
                    print("Available commands: quit, exit, stats, help, reset, save")
                    continue
                elif user_input.lower() == 'reset':
                    confirm = input("Are you sure you want to reset all learning? (y/N): ")
                    if confirm.lower() == 'y':
                        learner = ContinualLearner(config)  # Create fresh learner
                        teacher_opening = learner.start_conversation()
                        print(f"🔄 Reset complete. Teacher: {teacher_opening}")
                    continue
                elif user_input.lower() == 'save':
                    learner._save_memory()
                    print("💾 Progress saved")
                    continue
                
                # Process input and get response (this is where learning happens!)
                print("🧠 [Learning...]", end="", flush=True)
                start_time = time.time()
                
                response = learner.process_input_and_respond(user_input)
                
                processing_time = time.time() - start_time
                print(f"\r🤖 Learner: {response}")
                
                if args.debug:
                    print(f"   (processed in {processing_time:.2f}s)")
                
                # Show learning progress occasionally
                if learner.total_interactions % 5 == 0:
                    stats = learner.get_learning_stats()
                    print(f"   📈 Learned {stats['vocabulary_size']} patterns from "
                          f"{stats['total_interactions']} interactions")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nConversation interrupted")
                break
            except Exception as e:
                print(f"\nError during conversation: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"Failed to start conversation: {e}")
        return 1
    
    # Final save and stats
    learner._save_memory()
    print("\n📊 Final Learning Statistics:")
    print_stats(learner)
    
    print("\n✨ Session complete!")
    print("The network has learned from this conversation and will remember for next time.")
    print("Run again with --load-memory to continue from where you left off.")
    
    return 0


if __name__ == "__main__":
    exit(main())