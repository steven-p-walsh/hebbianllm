#!/usr/bin/env python3
"""
Fancy Neural Plasticity Demo with Multi-Pane CLI

Features a beautiful terminal interface with:
- Real-time conversation display
- Network monitoring and insights  
- Live statistics dashboard
- Learning progress visualization
"""

import sys
import time
import os
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from plastic_learner import PlasticContinualLearner, PlasticLearnerConfig
from fancy_cli import CLIManager
import signal


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Graceful shutdown initiated...")
    sys.exit(0)


def check_dependencies():
    """Check if required packages are available."""
    try:
        import rich
        return True
    except ImportError:
        print("❌ Missing required package 'rich'")
        print("📦 Install with: pip install rich")
        return False


def check_teacher_connection():
    """Check if teacher LLM is available."""
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check teacher connection
    if not check_teacher_connection():
        print("❌ Could not connect to teacher LLM at localhost:1234")
        print("🔧 Please start your LLM server first")
        return 1
    
    # Configuration optimized for learning
    config = PlasticLearnerConfig(
        n_neurons=14173,              # Safe allocation for GPU 1
        vocab_size=1000,              # Rich vocabulary
        initial_connectivity=0.08,    # Good starting connectivity
        plasticity_rate=0.01,         # Enhanced learning rate
        structural_plasticity=True,   # Allow rewiring
        save_interval=5,              # Frequent saves
        network_save_interval=3       # Frequent network saves
    )
    
    # Start the fancy CLI
    with CLIManager() as cli:
        # Welcome messages
        cli.log("🎯 Neural Plasticity Learning Demo", "system")
        cli.log("🚀 Initializing biologically-inspired network...", "system")
        
        # Initialize learner
        try:
            learner = PlasticContinualLearner(config)
            cli.log(f"✅ Network ready: {learner.network.n_neurons:,} neurons", "learning")
            
            # Load previous state
            cli.log("🔄 Loading previous memory and network state...", "system")
            learner.load_memory()
            
            if learner.total_interactions > 0:
                cli.log(f"📚 Resumed from {learner.total_interactions:,} interactions", "learning")
            else:
                cli.log("🌱 Starting fresh learning session", "learning")
                
        except Exception as e:
            cli.log_error(f"Failed to initialize: {e}")
            return 1
        
        # Initial stats
        cli.update_stats(learner.get_learning_stats())
        
        # Start conversation
        try:
            if learner.total_interactions == 0:
                teacher_msg = learner.start_conversation()
                cli.add_conversation("teacher", teacher_msg)
                cli.log("🎓 New conversation started", "learning")
            else:
                teacher_msg = "Let's continue our conversation! What would you like to talk about?"
                cli.add_conversation("teacher", teacher_msg)
                cli.log("🔄 Conversation resumed", "learning")
            
            turn = learner.total_interactions
            
            # Main learning loop
            while True:
                turn += 1
                
                # Learning step with timing
                start_time = time.time()
                
                # Generate response
                response = learner._generate_plastic_response(teacher_msg)
                learner.current_conversation.append({"role": "learner", "text": response})
                learner.total_interactions += 1
                
                # Show in conversation pane (clean)
                cli.add_conversation("learner", response)
                
                # Log the learning event
                cli.log(f"Generated response (turn {turn})", "learning")
                
                # Apply plasticity
                learner._learn_through_plasticity(response)
                
                # Get teacher feedback
                teacher_feedback = learner.teacher.respond_to_student(response)
                learner.current_conversation.append({"role": "teacher", "text": teacher_feedback})
                
                # Show in conversation pane
                cli.add_conversation("teacher", teacher_feedback)
                
                # Learn from feedback with reward modulation
                learner._learn_from_feedback_plastic(response, teacher_feedback)
                
                processing_time = time.time() - start_time
                cli.log(f"Processing time: {processing_time:.2f}s", "system")
                
                # Update stats every turn
                stats = learner.get_learning_stats()
                cli.update_stats(stats)
                
                # Log insights periodically
                if turn % 10 == 0:
                    connectivity = stats['connectivity']
                    active_neurons = stats['active_neurons']
                    
                    if connectivity > 0.1:
                        cli.log_learning_insight("High connectivity - rich associations forming")
                    elif connectivity < 0.03:
                        cli.log_learning_insight("Low connectivity - network exploring")
                    
                    if stats.get('response_feedback_overlap', 0) > 0.2:
                        cli.log_learning_insight("Strong context-response associations detected")
                    
                    # Log plasticity events
                    maturity = stats.get('maturity_factor', 1.0)
                    if maturity < 0.5:
                        cli.log_plasticity_event("Maturation", "Network transitioning to stable learning")
                
                # Periodic monitoring
                if turn % 5 == 0:
                    learner._monitor_plasticity()
                    cli.log("🔬 Plasticity monitoring completed", "system")
                
                # Auto-adjustment check
                if turn % 100 == 0:
                    old_ltp = stats.get('current_ltp_rate', 0)
                    learner._auto_adjust_learning_rates()
                    new_stats = learner.get_learning_stats()
                    new_ltp = new_stats.get('current_ltp_rate', 0)
                    
                    if abs(new_ltp - old_ltp) > 0.001:
                        cli.log_learning_insight(f"Auto-adjusted LTP rate: {old_ltp:.4f} → {new_ltp:.4f}")
                
                # Save states
                if turn % 100 == 0:
                    cli.log("💾 Saving network state...", "system")
                    learner._save_network_state()
                    learner._save_memory()
                    cli.log("✅ State saved successfully", "system")
                
                # Progress milestones
                if turn in [100, 500, 1000, 2000, 5000]:
                    cli.log_learning_insight(f"🎉 Milestone: {turn} interactions completed!")
                    
                    # Log learning quality assessment
                    if stats.get('learned_associations', 0) > 10:
                        cli.log_learning_insight("✨ Strong learning patterns emerging")
                    elif stats.get('learned_associations', 0) > 5:
                        cli.log_learning_insight("🌱 Basic associations forming")
                    else:
                        cli.log_learning_insight("🔍 Still exploring - associations developing")
                
                # Update teacher message for next turn
                teacher_msg = teacher_feedback
                
                # Brief pause to prevent overwhelming
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            cli.log("🛑 Learning session stopped by user", "system")
            
        except Exception as e:
            cli.log_error(f"Learning error: {e}")
        
        finally:
            # Final save
            try:
                cli.log("💾 Final save in progress...", "system")
                learner._save_network_state()
                learner._save_memory()
                cli.log("✅ Final save completed", "system")
            except Exception as e:
                cli.log_error(f"Final save failed: {e}")
            
            # Final statistics
            final_stats = learner.get_learning_stats()
            cli.update_stats(final_stats)
            
            cli.log("🎓 Learning session completed", "system")
            cli.log(f"📊 Final: {turn} turns, {final_stats['connectivity']:.1%} connectivity", "learning")
            
            # Keep interface open for a moment to view final state
            cli.log("Press Ctrl+C to exit...", "system")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code or 0)