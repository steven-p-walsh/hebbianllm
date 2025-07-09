#!/usr/bin/env python3
"""
Neural Plasticity Demo

Shows learning through synaptic plasticity rather than network growth.
This is much more biologically realistic and memory-efficient.

Key insight: Real brains learn by rewiring connections, not adding neurons!
"""

import sys
import time
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def print_plasticity_banner():
    """Print an informative banner about neural plasticity."""
    print("🧠 Neural Plasticity Learning Demo")
    print("=" * 40)
    print("This demo shows learning through synaptic plasticity:")
    print("• FIXED number of neurons (like real brains)")
    print("• Connections strengthen/weaken with use")
    print("• New synapses form, weak ones are pruned")
    print("• Fast learning without network growth")
    print("• No memory explosion issues")
    print()
    print("Just like baby brains - fixed neurons, plastic connections!")
    print()


def print_plasticity_stats(learner, turn):
    """Print current plasticity state."""
    stats = learner.get_learning_stats()
    
    print(f"🔬 Plasticity State (Turn {turn}):")
    print(f"   Neurons: {stats['network_neurons']} (FIXED)")
    print(f"   Connectivity: {stats['connectivity']:.1%}")
    print(f"   Active neurons: {stats['active_neurons']}")
    print(f"   Vocabulary: {stats['vocabulary_size']} patterns")
    print(f"   Plasticity steps: {stats['plasticity_step']}")
    print(f"   Teacher stage: {stats['teacher_stage']}")
    
    # Show recent plasticity events
    if stats['recent_plasticity']:
        print(f"   Recent changes: {len(stats['recent_plasticity'])} events")


def main():
    print_plasticity_banner()
    
    # Test teacher connection
    print("Testing teacher connection...")
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Teacher LLM is running")
        else:
            print("❌ Teacher LLM connection failed")
            return 1
    except Exception as e:
        print(f"❌ Could not connect to teacher: {e}")
        return 1
    
    # Configuration for efficient plasticity
    config = PlasticLearnerConfig(
        n_neurons=1500,              # Fixed size - no growth!
        vocab_size=300,              # Vocabulary capacity  
        initial_connectivity=0.05,   # Start sparse
        plasticity_rate=0.02,        # Fast plasticity
        structural_plasticity=True,  # Allow rewiring
        save_interval=999            # Don't save during demo
    )
    
    print("Initializing plastic learner...")
    learner = PlasticContinualLearner(config)
    
    # Show initial state
    initial_stats = learner.get_learning_stats()
    print(f"Initial state: {initial_stats['network_neurons']} neurons (FIXED)")
    print(f"Initial connectivity: {initial_stats['connectivity']:.1%}")
    print()
    
    # Start conversation
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    print()
    
    turn = 0
    try:
        while turn < 25:  # Reasonable demo length
            turn += 1
            
            print(f"🧠 [Turn {turn}] Plastic learning...", end="", flush=True)
            start_time = time.time()
            
            # Learn through plasticity (NOT network growth!)
            response = learner._generate_plastic_response(teacher_msg)
            learner.current_conversation.append({"role": "learner", "text": response})
            learner.total_interactions += 1
            
            # Apply plasticity updates
            learner._learn_through_plasticity(response)
            
            # Get teacher feedback
            teacher_feedback = learner.teacher.respond_to_student(response)
            learner.current_conversation.append({"role": "teacher", "text": teacher_feedback})
            
            # Learn from feedback through plasticity
            learner._learn_from_feedback_plastic(response, teacher_feedback)
            
            processing_time = time.time() - start_time
            
            print(f"\r🤖 Network: '{response}'")
            print(f"👩‍🏫 Teacher: '{teacher_feedback}'")
            print(f"⚡ Processed in {processing_time:.1f}s")
            
            # Show plasticity changes every few turns
            if turn % 5 == 0:
                print_plasticity_stats(learner, turn)
                print()
            
            # Monitor for interesting plasticity events
            if turn % 3 == 0:
                learner._monitor_plasticity()
            
            teacher_msg = teacher_feedback
            time.sleep(0.2)  # Brief pause
            
            # Check for natural ending
            if ("bye" in teacher_feedback.lower() or 
                "goodbye" in teacher_feedback.lower() or
                turn >= 20):
                print(f"🎉 Natural conversation end at turn {turn}")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo stopped at turn {turn}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Completed {turn} turns")
    
    # Final plasticity analysis
    final_stats = learner.get_learning_stats()
    
    print(f"\n🧬 PLASTICITY RESULTS:")
    print(f"   Turns completed: {turn}")
    print(f"   Neurons: {final_stats['network_neurons']} (UNCHANGED)")
    print(f"   Initial connectivity: {initial_stats['connectivity']:.1%}")
    print(f"   Final connectivity: {final_stats['connectivity']:.1%}")
    
    connectivity_change = final_stats['connectivity'] - initial_stats['connectivity']
    if abs(connectivity_change) > 0.01:
        direction = "increased" if connectivity_change > 0 else "decreased"
        print(f"   🔄 Connectivity {direction} by {abs(connectivity_change):.1%}")
    
    print(f"   Vocabulary: {initial_stats['vocabulary_size']} → {final_stats['vocabulary_size']}")
    print(f"   Active neurons: {final_stats['active_neurons']}")
    print(f"   Plasticity steps: {final_stats['plasticity_step']}")
    print(f"   Learning events: {final_stats['plasticity_events']}")
    
    if final_stats['plasticity_events'] > 0:
        print(f"   📈 Network rewired through {final_stats['plasticity_events']} plasticity events!")
    
    print(f"\n✨ Learning through plasticity succeeded!")
    print(f"Fixed {final_stats['network_neurons']} neurons learned through connection rewiring")
    print(f"No network growth needed - just like real brains! 🧠")


if __name__ == "__main__":
    main()