#!/usr/bin/env python3
"""
Resume Demo - Continue Learning from Saved State

Loads previously saved network state and continues learning.
This demonstrates true persistent neural plasticity.
"""

import sys
import time
import os
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def check_saved_states():
    """Check what saved states are available."""
    memory_dir = Path("experiments/conversational_learning/memory")
    
    if not memory_dir.exists():
        print("❌ No memory directory found")
        return False
    
    print("📁 Checking for saved states...")
    
    # Check for memory file
    memory_file = memory_dir / "plastic_memory.json"
    if memory_file.exists():
        print(f"✅ Found memory file: {memory_file}")
    else:
        print("❌ No memory file found")
    
    # Check for network states
    network_files = list(memory_dir.glob("network_state_*.npz"))
    latest_file = memory_dir / "network_state_latest.npz"
    
    if latest_file.exists():
        print(f"✅ Found latest network state: {latest_file}")
    else:
        print("❌ No latest network state found")
    
    if network_files:
        print(f"📊 Found {len(network_files)} historical network states:")
        for nf in sorted(network_files)[-5:]:  # Show last 5
            print(f"   {nf.name}")
    
    return memory_file.exists() or latest_file.exists()


def main():
    print("🔄 Resume Neural Plasticity Demo")
    print("=" * 40)
    print("Continue learning from saved network state")
    print()
    
    # Check for saved states
    if not check_saved_states():
        print("❌ No saved states found. Run the main demo first.")
        return 1
    
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
    
    # Configuration for resuming
    config = PlasticLearnerConfig(
        n_neurons=14173,             # Match original
        vocab_size=800,              # Match original
        initial_connectivity=0.06,   # Match original
        plasticity_rate=0.02,        # Match original
        structural_plasticity=True,  # Match original
        save_interval=25,            # Save more frequently when resuming
        network_save_interval=5      # Save network more frequently
    )
    
    print("🧠 Initializing learner and loading saved state...")
    
    start_init = time.time()
    learner = PlasticContinualLearner(config)
    
    # Load previous state
    print("📂 Loading previous memory and network state...")
    learner.load_memory()
    
    init_time = time.time() - start_init
    print(f"✅ Learner restored in {init_time:.1f}s")
    
    # Show restored state
    initial_stats = learner.get_learning_stats()
    total_synapses = initial_stats['network_neurons'] ** 2
    initial_connections = int(initial_stats['connectivity'] * total_synapses)
    
    print(f"\n🎯 RESTORED STATE:")
    print(f"   Previous conversations: {initial_stats['conversations']}")
    print(f"   Previous interactions: {initial_stats['total_interactions']}")
    print(f"   Neurons: {initial_stats['network_neurons']:,}")
    print(f"   Current connectivity: {initial_stats['connectivity']:.2%}")
    print(f"   Active synapses: {initial_connections:,}")
    print(f"   Vocabulary: {initial_stats['vocabulary_size']} patterns")
    print(f"   Plasticity step: {initial_stats['plasticity_step']}")
    print(f"   Teacher stage: {initial_stats['teacher_stage']}")
    print()
    
    # Continue conversation
    print("🔄 Resuming continuous learning...")
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    print()
    
    turn = initial_stats['total_interactions']  # Continue from where we left off
    session_start_time = time.time()
    
    try:
        while True:  # Run until manually stopped
            turn += 1
            
            print(f"🧠 [Turn {turn}] Resumed plastic learning...", end="", flush=True)
            start_time = time.time()
            
            # Continue learning through plasticity
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
            print(f"⚡ Processing: {processing_time:.1f}s | Turn {turn} (resumed)")
            
            # Show stats every few turns
            if turn % 5 == 0:
                current_stats = learner.get_learning_stats()
                connectivity_change = current_stats['connectivity'] - initial_stats['connectivity']
                vocab_change = current_stats['vocabulary_size'] - initial_stats['vocabulary_size']
                
                print(f"📊 Progress since resume:")
                print(f"   🧠 Connectivity: {initial_stats['connectivity']:.1%} → {current_stats['connectivity']:.1%} ({connectivity_change:+.1%})")
                print(f"   📚 Vocabulary: {initial_stats['vocabulary_size']} → {current_stats['vocabulary_size']} ({vocab_change:+d})")
                print(f"   ⚡ Active neurons: {current_stats['active_neurons']:,}")
                print()
            
            # Show progress summary every 20 turns
            if turn % 20 == 0:
                current_time = time.strftime("%H:%M:%S")
                current_stats = learner.get_learning_stats()
                turns_since_resume = turn - initial_stats['total_interactions']
                
                print(f"📅 Resume Progress [{current_time}] - Turn {turn}")
                print(f"   💡 {turns_since_resume} new interactions since resume")
                print(f"   🧠 Connectivity: {current_stats['connectivity']:.1%}")
                print(f"   📚 Vocabulary: {current_stats['vocabulary_size']} patterns")
                print(f"   🔄 Plasticity events: {current_stats['plasticity_events']}")
                print()
            
            # Monitor plasticity
            if turn % 4 == 0:
                print("🔬 Monitoring resumed plasticity...")
                learner._monitor_plasticity()
            
            teacher_msg = teacher_feedback
            time.sleep(0.2)  # Brief pause
    
    except KeyboardInterrupt:
        session_duration = time.time() - session_start_time
        turns_this_session = turn - initial_stats['total_interactions']
        print(f"\n\n⏹️  Resumed session stopped by user at turn {turn}")
        print(f"🕒 This session: {session_duration/60:.1f} minutes ({turns_this_session} new interactions)")
    except Exception as e:
        session_duration = time.time() - session_start_time
        turns_this_session = turn - initial_stats['total_interactions']
        print(f"\n❌ Error occurred: {e}")
        print(f"Completed {turns_this_session} new turns in {session_duration/60:.1f} minutes before error")
        
        # Emergency save
        print("💾 Emergency saving...")
        try:
            learner._save_network_state()
            learner._save_memory()
            print("✅ Emergency save completed")
        except Exception as save_error:
            print(f"❌ Emergency save failed: {save_error}")
    
    # Final save
    print("💾 Final save...")
    try:
        learner._save_network_state()
        learner._save_memory()
        print("✅ Final save completed")
    except Exception as save_error:
        print(f"⚠️  Final save had issues: {save_error}")
    
    # Final analysis
    final_stats = learner.get_learning_stats()
    final_connections = int(final_stats['connectivity'] * total_synapses)
    connection_change = final_connections - initial_connections
    turns_this_session = turn - initial_stats['total_interactions']
    
    print(f"\n🧬 RESUME SESSION RESULTS:")
    print(f"   New turns completed: {turns_this_session}")
    print(f"   Total turns ever: {turn}")
    print(f"   Neurons: {final_stats['network_neurons']:,} (UNCHANGED)")
    print(f"   Synapses: {initial_connections:,} → {final_connections:,}")
    
    if abs(connection_change) > 1000:
        direction = "formed" if connection_change > 0 else "pruned"
        print(f"   🔄 {abs(connection_change):,} synapses {direction} this session!")
    
    connectivity_change = final_stats['connectivity'] - initial_stats['connectivity']
    if abs(connectivity_change) > 0.001:
        direction = "increased" if connectivity_change > 0 else "decreased"
        print(f"   📈 Connectivity {direction} by {abs(connectivity_change):.2%} this session")
    
    vocab_change = final_stats['vocabulary_size'] - initial_stats['vocabulary_size']
    print(f"   📚 Vocabulary: {initial_stats['vocabulary_size']} → {final_stats['vocabulary_size']} ({vocab_change:+d})")
    print(f"   🔄 Total plasticity events: {final_stats['plasticity_events']}")
    
    print(f"\n✨ RESUMED LEARNING SUCCESS!")
    print(f"   Network continued learning from saved state")
    print(f"   {turns_this_session} new interactions with persistent plasticity")
    print(f"   True continuous biological learning! 🧠✅")


if __name__ == "__main__":
    main()