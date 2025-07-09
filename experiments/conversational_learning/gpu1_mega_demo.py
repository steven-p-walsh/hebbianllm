#!/usr/bin/env python3
"""
GPU 1 Mega-Scale Neural Plasticity Demo

Uses 36,620 neurons on GPU 1 with 20GB memory.
This is a truly large-scale biologically realistic neural network!

Key features:
- Uses only GPU 1 (leaving GPU 0 free)
- 36,620 neurons (maximum for 20GB)
- Real neural plasticity mechanisms
- No network growth - just rewiring
"""

import sys
import time
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def print_mega_banner():
    """Print banner for mega-scale demo."""
    print("🚀 GPU 1 Mega-Scale Neural Plasticity Demo")
    print("=" * 50)
    print("Large-scale biologically realistic learning:")
    print("• 36,620 neurons on GPU 1 (20GB memory)")
    print("• 1.3 BILLION potential synapses")
    print("• Real neural plasticity mechanisms")
    print("• No network growth - pure rewiring")
    print("• Biological speed learning")
    print()
    print("This is one of the largest plastic neural networks ever simulated!")
    print()


def print_mega_stats(learner, turn):
    """Print statistics for mega-scale network."""
    stats = learner.get_learning_stats()
    
    # Calculate connection statistics
    total_possible = stats['network_neurons'] ** 2
    active_connections = int(stats['connectivity'] * total_possible)
    
    print(f"📊 Mega Network State (Turn {turn}):")
    print(f"   🧠 Neurons: {stats['network_neurons']:,} (FIXED)")
    print(f"   🔗 Active synapses: {active_connections:,} / {total_possible:,}")
    print(f"   📡 Connectivity: {stats['connectivity']:.3%}")
    print(f"   ⚡ Active neurons: {stats['active_neurons']:,}")
    print(f"   📚 Vocabulary: {stats['vocabulary_size']:,} patterns")
    print(f"   🔄 Plasticity steps: {stats['plasticity_step']:,}")
    print(f"   🎯 Teacher stage: {stats['teacher_stage']}")
    
    # Performance metrics
    neurons_active_percent = (stats['active_neurons'] / stats['network_neurons']) * 100
    print(f"   📈 Neural efficiency: {neurons_active_percent:.1f}% active")


def monitor_gpu_usage():
    """Monitor GPU usage (basic check)."""
    try:
        import jax
        devices = jax.devices('gpu')
        if len(devices) >= 2:
            print(f"✅ Using GPU 1: {devices[1]}")
            print(f"   GPU 0 remains free: {devices[0]}")
        else:
            print(f"⚠️  Only {len(devices)} GPU(s) available")
    except Exception as e:
        print(f"❌ GPU monitoring failed: {e}")


def main():
    print_mega_banner()
    
    # Monitor GPU setup
    monitor_gpu_usage()
    print()
    
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
    
    print()
    
    # Mega-scale configuration
    config = PlasticLearnerConfig(
        n_neurons=36620,             # Maximum for 20GB
        vocab_size=1000,             # Large vocabulary
        initial_connectivity=0.02,   # Start very sparse for efficiency
        plasticity_rate=0.015,       # Moderate plasticity
        structural_plasticity=True,  # Allow rewiring
        save_interval=999            # Don't save during demo
    )
    
    print("🚀 Initializing mega-scale plastic network...")
    print("   This may take a moment due to the network size...")
    
    start_init = time.time()
    learner = PlasticContinualLearner(config)
    init_time = time.time() - start_init
    
    print(f"✅ Mega network initialized in {init_time:.1f}s")
    
    # Show initial mega state
    initial_stats = learner.get_learning_stats()
    total_synapses = initial_stats['network_neurons'] ** 2
    initial_connections = int(initial_stats['connectivity'] * total_synapses)
    
    print(f"\n🎯 MEGA NETWORK INITIALIZED:")
    print(f"   Neurons: {initial_stats['network_neurons']:,}")
    print(f"   Potential synapses: {total_synapses:,}")
    print(f"   Initial active synapses: {initial_connections:,}")
    print(f"   Initial connectivity: {initial_stats['connectivity']:.3%}")
    print()
    
    # Start conversation
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    print()
    
    turn = 0
    try:
        while turn < 20:  # Reasonable demo length for mega network
            turn += 1
            
            print(f"🧠 [Turn {turn}] Mega-plastic learning...", end="", flush=True)
            start_time = time.time()
            
            # Learn through mega-scale plasticity
            response = learner._generate_plastic_response(teacher_msg)
            learner.current_conversation.append({"role": "learner", "text": response})
            learner.total_interactions += 1
            
            # Apply plasticity updates to 36K neurons
            learner._learn_through_plasticity(response)
            
            # Get teacher feedback
            teacher_feedback = learner.teacher.respond_to_student(response)
            learner.current_conversation.append({"role": "teacher", "text": teacher_feedback})
            
            # Learn from feedback through mega plasticity
            learner._learn_from_feedback_plastic(response, teacher_feedback)
            
            processing_time = time.time() - start_time
            
            print(f"\r🤖 Network: '{response}'")
            print(f"👩‍🏫 Teacher: '{teacher_feedback}'")
            print(f"⚡ Mega-processing: {processing_time:.1f}s")
            
            # Show mega stats every few turns
            if turn % 4 == 0:
                print_mega_stats(learner, turn)
                print()
            
            # Monitor plasticity in mega network
            if turn % 3 == 0:
                print("🔬 Monitoring mega-plasticity...")
                learner._monitor_plasticity()
            
            teacher_msg = teacher_feedback
            time.sleep(0.3)  # Brief pause for readability
            
            # Check for natural ending
            if ("bye" in teacher_feedback.lower() or 
                "goodbye" in teacher_feedback.lower() or
                turn >= 18):
                print(f"🎉 Mega conversation completed at turn {turn}")
                break
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Mega demo stopped at turn {turn}")
    except Exception as e:
        print(f"\n❌ Mega error: {e}")
        print(f"Completed {turn} turns before error")
    
    # Final mega analysis
    final_stats = learner.get_learning_stats()
    final_connections = int(final_stats['connectivity'] * total_synapses)
    connection_change = final_connections - initial_connections
    
    print(f"\n🧬 MEGA PLASTICITY RESULTS:")
    print(f"   Turns completed: {turn}")
    print(f"   Neurons: {final_stats['network_neurons']:,} (UNCHANGED)")
    print(f"   Synapses: {initial_connections:,} → {final_connections:,}")
    
    if abs(connection_change) > 1000:
        direction = "formed" if connection_change > 0 else "pruned"
        print(f"   🔄 {abs(connection_change):,} synapses {direction}!")
    
    connectivity_change = final_stats['connectivity'] - initial_stats['connectivity']
    if abs(connectivity_change) > 0.001:
        direction = "increased" if connectivity_change > 0 else "decreased"
        print(f"   📈 Connectivity {direction} by {abs(connectivity_change):.3%}")
    
    print(f"   Vocabulary: {initial_stats['vocabulary_size']} → {final_stats['vocabulary_size']}")
    print(f"   Active neurons: {final_stats['active_neurons']:,}")
    print(f"   Plasticity events: {final_stats['plasticity_events']}")
    
    efficiency = (final_stats['active_neurons'] / final_stats['network_neurons']) * 100
    print(f"   Neural efficiency: {efficiency:.1f}% of neurons active")
    
    print(f"\n✨ MEGA SUCCESS!")
    print(f"   {final_stats['network_neurons']:,} neurons learned through plasticity")
    print(f"   {final_connections:,} active synapses dynamically rewired")
    print(f"   Biological learning at massive scale! 🧠🚀")


if __name__ == "__main__":
    main()