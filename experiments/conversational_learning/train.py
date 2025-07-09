#!/usr/bin/env python3
"""
Main Training Script for Conversational Learning

This script runs the full conversational learning system where a local LLM 
acts as a teacher to train the Hebbian SNN through conversation.

Usage:
    python train.py [--config CONFIG_FILE] [--episodes N] [--resume CHECKPOINT]
"""

import argparse
import sys
import json
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.trainer import ConversationalTrainer, TrainingConfig


def create_default_config() -> TrainingConfig:
    """Create default training configuration."""
    return TrainingConfig(
        # Network parameters
        vocab_size=500,  # Start smaller for faster training
        max_seq_length=64,
        column_size=30,
        n_layers=3,
        batch_size=1,  # Process one conversation at a time
        
        # Training parameters  
        max_episodes=200,
        steps_per_episode=15,
        conversation_turns_per_episode=4,
        
        # Learning parameters
        learning_rate=0.01,
        temperature=1.2,
        context_length=20,
        
        # Teacher parameters (your local LLM)
        teacher_api_url="http://localhost:1234/v1/chat/completions",
        teacher_model="gemma-3-27b-it-qat",
        teacher_temperature=0.7,
        
        # Logging and saving
        save_interval=25,
        log_interval=5,
        save_conversations=True
    )


def load_config_from_file(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object with loaded parameters
    return TrainingConfig(**config_dict)


def save_config_to_file(config: TrainingConfig, config_path: str):
    """Save configuration to JSON file."""
    from dataclasses import asdict
    
    config_dict = asdict(config)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Hebbian SNN with conversational learning')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-config', type=str, help='Save default config to file and exit')
    parser.add_argument('--test-teacher', action='store_true', help='Test teacher connection and exit')
    parser.add_argument('--interactive', action='store_true', help='Run interactive session after training')
    
    args = parser.parse_args()
    
    # Save default config if requested
    if args.save_config:
        config = create_default_config()
        save_config_to_file(config, args.save_config)
        return
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_from_file(args.config)
    else:
        print("Using default configuration")
        config = create_default_config()
    
    # Override episodes if specified
    if args.episodes:
        config.max_episodes = args.episodes
    
    print(f"Training Configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max episodes: {config.max_episodes}")
    print(f"  Teacher API: {config.teacher_api_url}")
    print(f"  Teacher model: {config.teacher_model}")
    print()
    
    # Test teacher connection if requested
    if args.test_teacher:
        print("Testing teacher connection...")
        from utils.llm_teacher import LLMTeacher, TeacherConfig
        
        teacher_config = TeacherConfig(
            api_url=config.teacher_api_url,
            model=config.teacher_model,
            temperature=config.teacher_temperature
        )
        teacher = LLMTeacher(teacher_config)
        
        # Try to start a conversation
        response = teacher.start_conversation()
        if response:
            print(f"✓ Teacher connection successful!")
            print(f"  Teacher says: '{response}'")
        else:
            print("✗ Teacher connection failed!")
            print("  Make sure your local LLM is running on the specified URL")
            return 1
        return 0
    
    # Initialize trainer
    print("Initializing conversational trainer...")
    trainer = ConversationalTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    try:
        # Start training
        print("Starting conversational training...")
        print("Press Ctrl+C to stop training and save progress")
        print("=" * 50)
        
        trainer.train()
        
        print("=" * 50)
        print("Training completed successfully!")
        
        # Run interactive session if requested
        if args.interactive:
            print("Starting interactive session...")
            trainer.interactive_session()
        
        # Show some evaluation
        test_prompts = ["hi", "hello", "how are you", "what is your name"]
        eval_results = trainer.evaluate_student(test_prompts)
        
        print("\nFinal Evaluation:")
        for result in eval_results['responses']:
            print(f"  Prompt: '{result['prompt']}' -> Response: '{result['response']}'")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Progress has been saved automatically")
        return 0
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())