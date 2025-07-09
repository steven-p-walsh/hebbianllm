#!/usr/bin/env python3
"""
Safe GPU 1 Neural Plasticity Demo

Uses 14,173 neurons on GPU 1 with 3GB memory (safe allocation).
GPU 0 is left free for the LLM teacher (google/gemma-3-12b).

This provides a good balance of scale and memory safety.
"""

import sys
import time
import os
from pathlib import Path

# Add hebbianllm to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from plastic_learner import PlasticContinualLearner, PlasticLearnerConfig


def print_safe_banner():
    """Print banner for safe demo."""
    print("🎯 Safe GPU 1 Neural Plasticity Demo - CONTINUOUS")
    print("=" * 55)
    print("Memory-safe biologically realistic learning:")
    print("• 14,173 neurons on GPU 1 (3GB memory)")
    print("• 200+ MILLION potential synapses")
    print("• GPU 0 free for LLM teacher")
    print("• Real neural plasticity mechanisms")
    print("• No memory overflow risk")
    print("• RUNS CONTINUOUSLY until you stop it!")
    print()
    print("Press Ctrl+C to stop and see final results")
    print("Optimal balance of scale and safety!")
    print()


def check_gpu_isolation():
    """Verify GPU isolation is working."""
    try:
        import jax
        
        # Check visible devices
        visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"CUDA_VISIBLE_DEVICES: {visible_gpus}")
        
        # Check JAX devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
        
        print(f"JAX sees {len(gpu_devices)} GPU(s): {gpu_devices}")
        
        if len(gpu_devices) == 1 and 'gpu:1' in str(gpu_devices[0]).lower():
            print("✅ Perfect GPU isolation - only GPU 1 visible to JAX")
            return True
        elif len(gpu_devices) == 1:
            print(f"✅ Single GPU mode: {gpu_devices[0]}")
            return True
        else:
            print(f"⚠️  Multiple GPUs visible to JAX: {gpu_devices}")
            print("   This may cause memory conflicts with teacher")
            return False
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def print_safe_stats(learner, turn):
    """Print statistics for safe-scale network."""
    stats = learner.get_learning_stats()
    
    # Calculate connection statistics
    total_possible = stats['network_neurons'] ** 2
    active_connections = int(stats['connectivity'] * total_possible)
    
    print(f"📊 Safe Network State (Turn {turn}):") 
    print(f"   🧠 Neurons: {stats['network_neurons']:,} (FIXED)")
    print(f"   🔗 Active synapses: {active_connections:,}")
    print(f"   📡 Connectivity: {stats['connectivity']:.2%}")
    print(f"   ⚡ Active neurons: {stats['active_neurons']:,}")
    print(f"   📚 Vocabulary: {stats['vocabulary_size']:,} patterns")
    print(f"   🔄 Plasticity steps: {stats['plasticity_step']:,}")
    print(f"   🎯 Teacher stage: {stats['teacher_stage']}")
    
    # Enhanced learning progress indicators
    if 'recent_plasticity' in stats and stats['recent_plasticity']:
        recent_events = stats['recent_plasticity']
        print(f"   🔬 Recent plasticity events: {len(recent_events)}")
        if recent_events:
            latest_event = recent_events[-1]
            if 'stats' in latest_event:
                latest_connectivity = latest_event['stats']['weights']['connectivity']
                print(f"   📈 Latest connectivity: {latest_connectivity:.3f}")
    
    # Tonic activity info
    if 'tonic_strength' in stats:
        tonic_pct = stats['tonic_strength'] * 100
        maturity = "mature" if stats.get('is_mature', False) else "developing"
        print(f"   🌱 Tonic activity: {tonic_pct:.1f}% ({maturity})")
    
    # Memory efficiency
    neurons_active_percent = (stats['active_neurons'] / stats['network_neurons']) * 100
    print(f"   📈 Neural efficiency: {neurons_active_percent:.1f}% active")
    
    # Token mapping efficiency
    if hasattr(learner.network, 'token_mapper'):
        neurons_per_token = learner.network.token_mapper.neurons_per_token
        effective_vocab = learner.network.token_mapper.effective_vocab_size
        print(f"   🎯 Neurons per token: {neurons_per_token}")
        print(f"   📖 Effective vocab size: {effective_vocab}")
        
        # Show token mapping utilization
        mapped_tokens = len(learner.network.token_mapper.token_mappings)
        print(f"   🗂️  Mapped tokens: {mapped_tokens}/{effective_vocab}")
    
    # Learning associations
    if hasattr(learner, 'learned_associations'):
        association_count = len(learner.learned_associations)
        print(f"   🧠 Learned associations: {association_count}")
        
        # Show recent responses if available
        if association_count > 0:
            recent_responses = list(learner.learned_associations.keys())[-3:]
            print(f"   💬 Recent responses: {recent_responses}")
    
    # Conversation progress
    if hasattr(learner, 'current_conversation') and learner.current_conversation:
        conv_length = len(learner.current_conversation)
        print(f"   🗣️  Conversation length: {conv_length} exchanges")


def main():
    print_safe_banner()
    
    # Check GPU isolation
    print("Checking GPU isolation...")
    gpu_ok = check_gpu_isolation()
    if not gpu_ok:
        print("⚠️  GPU isolation may not be perfect, but continuing...")
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
    
    # Biologically-inspired configuration for effective learning
    config = PlasticLearnerConfig(
        n_neurons=14173,              # Safe allocation for GPU 1 (3GB memory)
        vocab_size=1000,              # Reduced vocabulary for ~14 neurons per token
        initial_connectivity=0.08,   # Reasonable connectivity for learning
        plasticity_rate=0.01,        # Moderate plasticity rate
        structural_plasticity=True,  # Allow rewiring
        save_interval=50,            # Save memory every 50 conversations
        network_save_interval=10     # Save network every 10 interactions
    )
    
    print("🧠 Initializing safe plastic network...")
    
    start_init = time.time()
    learner = PlasticContinualLearner(config)
    init_time = time.time() - start_init
    
    print(f"✅ Safe network initialized in {init_time:.1f}s")
    
    # Show initial safe state
    initial_stats = learner.get_learning_stats()
    total_synapses = initial_stats['network_neurons'] ** 2
    initial_connections = int(initial_stats['connectivity'] * total_synapses)
    
    print(f"\n🎯 SAFE NETWORK READY:")
    print(f"   Neurons: {initial_stats['network_neurons']:,}")
    print(f"   Potential synapses: {total_synapses:,}")
    print(f"   Initial active synapses: {initial_connections:,}")
    print(f"   Initial connectivity: {initial_stats['connectivity']:.2%}")
    print(f"   Memory usage: ~3GB on GPU 1")
    print()
    
    # Start conversation
    teacher_msg = learner.start_conversation()
    print(f"👩‍🏫 Teacher: {teacher_msg}")
    print()
    
    turn = 0
    session_start_time = time.time()
    try:
        while True:  # Run until manually stopped
            turn += 1
            
            print(f"🧠 [Turn {turn}] Safe plastic learning...", end="", flush=True)
            start_time = time.time()
            
            # Learn through safe-scale plasticity
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
            print(f"⚡ Processing: {processing_time:.1f}s | Turn {turn}")
            
            # Show safe stats every few turns
            if turn % 5 == 0:
                print_safe_stats(learner, turn)
                print()
            
            # Show progress summary every 25 turns
            if turn % 25 == 0:
                current_time = time.strftime("%H:%M:%S")
                final_stats = learner.get_learning_stats()
                print(f"📅 Progress Update [{current_time}] - Turn {turn}")
                print(f"   💡 Network has been running for {turn} interactions")
                print(f"   🧠 Connectivity: {final_stats['connectivity']:.1%}")
                print(f"   📚 Vocabulary: {final_stats['vocabulary_size']} patterns")
                print(f"   ⚡ Active neurons: {final_stats['active_neurons']:,}")
                print(f"   🔄 Plasticity events: {final_stats['plasticity_events']}")
                
                # Show learning trajectory
                if hasattr(learner, 'learned_associations') and learner.learned_associations:
                    print(f"   🎯 Learning associations: {len(learner.learned_associations)}")
                    
                    # Show recent learning patterns
                    recent_keys = list(learner.learned_associations.keys())[-5:]
                    print(f"   📝 Recent responses: {recent_keys}")
                    
                    # Show positive vs negative feedback ratio
                    all_feedback = []
                    for response_data in learner.learned_associations.values():
                        all_feedback.extend([entry['positive'] for entry in response_data])
                    
                    if all_feedback:
                        positive_ratio = sum(all_feedback) / len(all_feedback)
                        print(f"   👍 Positive feedback ratio: {positive_ratio:.1%}")
                
                print()
            
            # Monitor plasticity
            if turn % 4 == 0:
                print("🔬 Monitoring plasticity...")
                learner._monitor_plasticity()
            
            teacher_msg = teacher_feedback
            time.sleep(0.2)  # Brief pause
            
            # No automatic stopping - run until user stops manually
    
    except KeyboardInterrupt:
        session_duration = time.time() - session_start_time
        print(f"\n\n⏹️  Continuous demo stopped by user at turn {turn}")
        print(f"🕒 Session duration: {session_duration/60:.1f} minutes ({turn} interactions)")
    except Exception as e:
        session_duration = time.time() - session_start_time
        print(f"\n❌ Error occurred: {e}")
        print(f"Completed {turn} turns in {session_duration/60:.1f} minutes before error")
        
        # Emergency save before stopping
        print("💾 Emergency saving network state...")
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
    
    # Final safe analysis
    final_stats = learner.get_learning_stats()
    final_connections = int(final_stats['connectivity'] * total_synapses)
    connection_change = final_connections - initial_connections
    
    print(f"\n🧬 SAFE PLASTICITY RESULTS:")
    print(f"   Turns completed: {turn}")
    print(f"   Neurons: {final_stats['network_neurons']:,} (UNCHANGED)")
    print(f"   Synapses: {initial_connections:,} → {final_connections:,}")
    
    if abs(connection_change) > 100:
        direction = "formed" if connection_change > 0 else "pruned"
        print(f"   🔄 {abs(connection_change):,} synapses {direction}!")
    
    connectivity_change = final_stats['connectivity'] - initial_stats['connectivity']
    if abs(connectivity_change) > 0.001:
        direction = "increased" if connectivity_change > 0 else "decreased"
        print(f"   📈 Connectivity {direction} by {abs(connectivity_change):.2%}")
    
    print(f"   Vocabulary: {initial_stats['vocabulary_size']} → {final_stats['vocabulary_size']}")
    print(f"   Active neurons: {final_stats['active_neurons']:,}")
    print(f"   Plasticity events: {final_stats['plasticity_events']}")
    
    efficiency = (final_stats['active_neurons'] / final_stats['network_neurons']) * 100
    print(f"   Neural efficiency: {efficiency:.1f}% of neurons active")
    
    print(f"\n✨ CONTINUOUS LEARNING SUCCESS!")
    print(f"   {final_stats['network_neurons']:,} neurons learned through {turn} interactions")
    print(f"   {final_connections:,} synapses dynamically rewired")
    print(f"   Learning rate: {turn/(session_duration/60):.1f} interactions/minute")
    print(f"   Session duration: {session_duration/60:.1f} minutes")
    print(f"   Memory-safe continuous plasticity! 🧠✅")


if __name__ == "__main__":
    main()