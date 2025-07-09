"""
Plastic Continual Learner

Uses a fixed-size network with neural plasticity instead of growth.
This is much more biologically realistic and memory-efficient.

Key features:
- Fixed number of neurons (like human brains)
- Synaptic weights change through use
- Fast learning through connection rewiring
- Homeostatic balance to prevent instability
- No memory growth issues
"""

import time
import json
import logging
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from models.plastic_snn import PlasticHebSNN
from utils.biologically_inspired_tokenizer import BiologicalTokenizer
from utils.llm_teacher import LLMTeacher, TeacherConfig


@dataclass
class PlasticLearnerConfig:
    """Configuration for the plastic learner."""
    
    # Fixed network parameters (no growth!) - Safe size for GPU 1 only
    n_neurons: int = 14173        # Conservative neurons for 3GB memory
    vocab_size: int = 800         # Vocabulary capacity
    initial_connectivity: float = 0.08  # Reasonable density
    
    # Learning parameters
    plasticity_rate: float = 0.01  # How fast synapses change
    homeostatic_strength: float = 0.001  # Balance mechanism
    structural_plasticity: bool = True  # Allow synapse formation/pruning
    
    # Interaction parameters
    teacher_api_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model: str = "google/gemma-3-12b"
    teacher_temperature: float = 0.7
    
    # Memory and saving
    save_interval: int = 5        # Save memory every 5 conversations (more frequent)
    network_save_interval: int = 3  # Save network state every 3 interactions (more frequent)
    memory_file: str = "memory/plastic_memory.json"     # Relative to current directory
    network_file: str = "memory/network_state.npz"     # Relative to current directory
    log_file: str = "logs/plastic_learning.log"        # Relative to current directory


class PlasticContinualLearner:
    """
    Continual learner using fixed-size network with plasticity.
    
    This is more biologically realistic:
    - No network growth, just rewiring
    - Fast learning through synaptic plasticity
    - Stable memory capacity
    - No GPU memory issues
    """
    
    def __init__(self, config: PlasticLearnerConfig = None):
        self.config = config or PlasticLearnerConfig()
        
        # Setup directories
        Path(self.config.memory_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tokenizer
        self.tokenizer = BiologicalTokenizer(max_vocab_size=self.config.vocab_size)
        
        # Initialize teacher
        teacher_config = TeacherConfig(
            api_url=self.config.teacher_api_url,
            model=self.config.teacher_model,
            temperature=self.config.teacher_temperature
        )
        self.teacher = LLMTeacher(teacher_config)
        
        # Initialize fixed-size plastic network
        self.network = PlasticHebSNN(
            n_neurons=self.config.n_neurons,
            vocab_size=self.config.vocab_size,
            initial_connectivity=self.config.initial_connectivity
        )
        
        # Learning state
        self.conversation_count = 0
        self.total_interactions = 0
        self.learned_associations = {}
        self.plasticity_events = []  # Track major plasticity changes
        
        # Phase 2: Attention and novelty state
        self.pattern_familiarity = {}  # Track pattern familiarity for novelty detection
        self.attention_context = []    # Recent attended patterns
        self.novelty_history = []      # Track novelty levels over time
        
        # Phase 3: Fatigue and STP state
        self.sleep_cycle_counter = 0   # Track when to trigger sleep
        self.fatigue_events = []       # Track fatigue buildup events
        
        # Phase 4: Sleep replay state
        self.replay_events = []        # Track replay events during sleep
        self.synaptic_capture_events = []  # Track synaptic capture events
        
        # Conversation state
        self.current_conversation = []
        
        self.logger.info("Plastic Continual Learner initialized")
        self.logger.info(f"Fixed network: {self.network.n_neurons} neurons")
        self.logger.info(f"Vocab capacity: {self.config.vocab_size}")
        self.logger.info(f"Initial connectivity: {self.config.initial_connectivity:.1%}")
    
    def _setup_logging(self):
        """Setup logging for the learner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Suppress verbose JAX logging
        logging.getLogger('jax._src.dispatch').setLevel(logging.WARNING)
        logging.getLogger('jax._src.compiler').setLevel(logging.WARNING)
        logging.getLogger('jax._src.cache_key').setLevel(logging.WARNING)
        logging.getLogger('jax._src.compilation_cache').setLevel(logging.WARNING)
        logging.getLogger('jax._src.interpreters.pxla').setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def start_conversation(self) -> str:
        """Start a new conversation session."""
        self.current_conversation = []
        self.conversation_count += 1
        
        # Get opening from teacher
        teacher_opening = self.teacher.start_conversation()
        self.current_conversation.append({"role": "teacher", "text": teacher_opening})
        
        self.logger.info(f"Conversation {self.conversation_count} started")
        self.logger.info(f"Teacher: {teacher_opening}")
        
        return teacher_opening
    
    def process_input_and_respond(self, input_text: str) -> str:
        """
        Process input and generate response with real-time plasticity.
        
        This is where the key difference is: instead of growing the network,
        we change the synaptic connections through plasticity.
        """
        self.total_interactions += 1
        
        # Record input
        self.current_conversation.append({"role": "input", "text": input_text})
        
        # Learn from input through plasticity (not network growth!)
        self._learn_through_plasticity(input_text)
        
        # Generate response using current network state
        response = self._generate_plastic_response(input_text)
        
        # Record response
        self.current_conversation.append({"role": "learner", "text": response})
        
        # Get teacher feedback
        teacher_feedback = self.teacher.respond_to_student(response)
        self.current_conversation.append({"role": "teacher", "text": teacher_feedback})
        
        # Learn from feedback through plasticity
        self._learn_from_feedback_plastic(response, teacher_feedback)
        
        # Monitor plasticity changes
        if self.total_interactions % 5 == 0:
            self._monitor_plasticity()
        
        # Phase 3: Check for sleep need every 10 interactions
        self.sleep_cycle_counter += 1
        if self.sleep_cycle_counter >= 10:
            self._trigger_adenosine_sleep()
        
        # Save network state more frequently
        if self.total_interactions % self.config.network_save_interval == 0:
            self._save_network_state()
        
        # Save progress periodically based on interactions
        if self.total_interactions % self.config.save_interval == 0:
            self._save_memory()
        
        self.logger.info(f"Input: {input_text}")
        self.logger.info(f"Learner: {response}")
        self.logger.info(f"Teacher: {teacher_feedback}")
        
        return response
    
    def _learn_through_plasticity(self, text: str):
        """
        Learn from text through synaptic plasticity with Phase 2 neuromodulation.
        
        This is the key insight: real brains learn by changing connections,
        not by adding neurons.
        """
        # Encode text
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) > 0:
            # Phase 2: Compute attention and novelty
            attention_level = self._compute_attention_level(text)
            novelty_score = self._compute_novelty_score(text)
            
            # Set neuromodulators based on attention and novelty
            self.network.modulators.set_mod('acetylcholine', attention_level)
            self.network.modulators.set_mod('norepinephrine', novelty_score)
            
            # Process through network with Phase 2 neuromodulation
            modulators = self.network.modulators.to_dict()
            result = self.network.process_tokens(token_ids, learning=True, modulators=modulators)
            
            # Debug logging for Phase 2
            self.logger.debug(f"Phase 2 - Attention: {attention_level:.2f}, Novelty: {novelty_score:.2f}")
            
            # Track plasticity event
            plasticity_stats = self.network.get_plasticity_stats()
            
            # Record significant plasticity changes
            if len(self.plasticity_events) == 0 or self._is_significant_change(plasticity_stats):
                self.plasticity_events.append({
                    'step': self.total_interactions,
                    'text': text,
                    'stats': plasticity_stats,
                    'attention': attention_level,
                    'novelty': novelty_score,
                    'timestamp': time.time()
                })
                
                # Keep recent events only
                if len(self.plasticity_events) > 50:
                    self.plasticity_events.pop(0)
    
    def _is_significant_change(self, current_stats: Dict) -> bool:
        """Check if plasticity changes are significant enough to record."""
        if not self.plasticity_events:
            return True
        
        last_stats = self.plasticity_events[-1]['stats']
        
        # Check for significant changes in connectivity
        connectivity_change = abs(current_stats['weights']['connectivity'] - 
                                last_stats['weights']['connectivity'])
        
        # Check for significant changes in activity
        activity_change = abs(current_stats['activity']['current_activity_mean'] - 
                            last_stats['activity']['current_activity_mean'])
        
        return connectivity_change > 0.01 or activity_change > 0.05
    
    def _generate_plastic_response(self, context: str) -> str:
        """
        Generate response using current synaptic state.
        
        The network's response depends on its current connectivity pattern,
        which has been shaped by all previous learning.
        """
        try:
            # Get conversation context
            recent_context = self._get_conversation_context()
            context_tokens = self.tokenizer.encode(recent_context, add_special_tokens=False)
            
            # Generate using current plastic network state
            generated_tokens = self.network.generate_tokens(
                context_tokens[-8:],  # Recent context
                max_length=8         # Short responses
            )
            
            # Remove context to get just the response
            if len(generated_tokens) > len(context_tokens):
                response_tokens = generated_tokens[len(context_tokens):]
            else:
                response_tokens = generated_tokens[-3:]  # Fallback
            
            # Debug: Show what tokens were generated
            token_names = []
            for token_id in response_tokens:
                if token_id in self.tokenizer.id_to_pattern:
                    token_names.append(self.tokenizer.id_to_pattern[token_id])
                else:
                    token_names.append(f"<UNK:{token_id}>")
            
            print(f"🔍 Debug - Generated token IDs: {response_tokens}")
            print(f"🔍 Debug - Token names: {token_names}")
            
            # Decode response (keep PAUSE tokens to preserve spacing)
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
            print(f"🔍 Debug - Raw decoded: '{response}'")
            
            # Clean up the response but preserve internal spaces
            response = response.replace('<BOS>', '').replace('<EOS>', '').replace('<UNK>', '')
            
            # Don't strip if response contains meaningful internal spaces
            # Only strip if it's all whitespace or starts/ends with spaces but has content
            if response.strip():  # If there's actual content
                # Remove leading/trailing single spaces, but preserve internal spaces
                response = response.strip()
                # If response was just spaces, keep it as is (neural babbling)
            
            # Collapse multiple spaces to single spaces
            import re
            response = re.sub(r'\s+', ' ', response)
            
            print(f"🔍 Debug - Final response: '{response}'")
            
            # Debug info for sparse networks
            self.logger.debug(f"Generated tokens: {response_tokens}")
            self.logger.debug(f"Decoded response: '{response}'")
            
            # For very early learning, even minimal responses are valid
            if len(response) < 1:
                # Generate simple babbling response for tonic phase
                babble_sounds = ["hi", "ma", "da", "ba", "o", "a", "e"]
                import random
                response = random.choice(babble_sounds)
                self.logger.debug(f"Using fallback babbling: '{response}'")
                
                if len(response) < 1:
                    raise Exception(f"Network generated empty response from context: '{context}' - neural generation failed")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            raise Exception(f"Neural response generation completely failed: {e}")
    
    def _learn_from_feedback_plastic(self, response: str, feedback: str):
        """
        Learn from teacher feedback through dopamine-gated plasticity.
        
        Implements reward prediction error (RPE) signaling:
        - Positive feedback: RPE = +1 (unexpected reward)
        - Negative feedback: RPE = -1 (negative prediction error)  
        - Neutral feedback: RPE = 0 (no prediction error)
        """
        # Compute reward prediction error (RPE) from feedback
        rpe = self._compute_rpe_from_feedback(feedback)
        
        # Set dopamine level based on RPE
        self.network.modulators.set_mod('dopamine', rpe)
        
        # Encode feedback for plasticity
        feedback_tokens = self.tokenizer.encode(feedback)
        
        if feedback_tokens:
            # Process feedback through network with dopamine modulation
            modulators = self.network.modulators.to_dict()
            self.network.process_tokens(feedback_tokens, learning=True, modulators=modulators)
            
            self.logger.debug(f"RPE signal: {rpe:.2f}, Dopamine: {modulators['dopamine']:.2f}")
        
        # Decay dopamine over time (τ = 5 time steps)
        self.network.modulators.decay_mod('dopamine', tau=5.0, dt=1.0)
        
        # Record association with RPE
        if response not in self.learned_associations:
            self.learned_associations[response] = []
        
        self.learned_associations[response].append({
            'feedback': feedback,
            'rpe': rpe,
            'positive': rpe > 0,
            'negative': rpe < 0,
            'plasticity_step': self.network.learning_step,
            'timestamp': time.time()
        })
    
    def _compute_novelty_score(self, text: str) -> float:
        """
        Compute novelty score for text pattern.
        
        Returns:
            Novelty score ∈ [0, 1]:
            - 1.0: Completely novel pattern
            - 0.5: Moderately familiar pattern
            - 0.0: Very familiar pattern
        """
        # Create pattern signature from text
        pattern_signature = self._create_pattern_signature(text)
        
        # Get current familiarity (defaults to 0 for new patterns)
        current_familiarity = self.pattern_familiarity.get(pattern_signature, 0.0)
        
        # Update familiarity (exponential moving average)
        α = 0.1  # Learning rate for familiarity
        new_familiarity = current_familiarity + α * (1.0 - current_familiarity)
        self.pattern_familiarity[pattern_signature] = new_familiarity
        
        # Novelty is inverse of familiarity
        novelty = 1.0 - current_familiarity
        
        # Keep novelty history for attention computation
        self.novelty_history.append(novelty)
        if len(self.novelty_history) > 20:  # Keep recent history
            self.novelty_history.pop(0)
        
        return novelty
    
    def _create_pattern_signature(self, text: str) -> str:
        """Create a signature for pattern recognition."""
        # Simple pattern signature: first few tokens + length
        tokens = self.tokenizer.encode(text)[:5]  # First 5 tokens
        return f"{len(tokens)}:{':'.join(map(str, tokens))}"
    
    def _compute_attention_level(self, context: str) -> float:
        """
        Compute attention level for current context.
        
        Returns:
            Attention level ∈ [0, 1]:
            - 1.0: High attention (important/relevant context)
            - 0.5: Moderate attention 
            - 0.0: Low attention (irrelevant context)
        """
        # Attention based on recent interaction quality and novelty
        base_attention = 0.5  # Baseline attention
        
        # Boost attention for novel patterns
        if self.novelty_history:
            recent_novelty = np.mean(self.novelty_history[-3:])
            novelty_boost = 0.3 * recent_novelty
        else:
            novelty_boost = 0.0
        
        # Boost attention for context containing question words (engagement)
        question_words = ['what', 'why', 'how', 'where', 'when', 'who', '?']
        engagement_boost = 0.2 if any(word in context.lower() for word in question_words) else 0.0
        
        # Boost attention for emotional context (important for learning)
        emotional_words = ['good', 'bad', 'excellent', 'wrong', 'great', 'terrible']
        emotional_boost = 0.2 if any(word in context.lower() for word in emotional_words) else 0.0
        
        # Combine attention factors
        attention = base_attention + novelty_boost + engagement_boost + emotional_boost
        
        # Normalize to [0, 1] range
        attention = np.clip(attention, 0.0, 1.0)
        
        return attention
    
    def _trigger_adenosine_sleep(self):
        """
        Trigger adenosine-clearing sleep when fatigue is high.
        
        This simulates the biological function of sleep in clearing adenosine
        and consolidating memories through STP → LTP conversion.
        """
        # Check if sleep is needed
        avg_fatigue = float(jnp.mean(self.network.neuron_fatigue))
        
        if avg_fatigue > 0.6:  # High fatigue threshold
            self.logger.info(f"😴 Triggering adenosine-clearing sleep (fatigue: {avg_fatigue:.3f})")
            
            # Clear adenosine (fatigue recovery)
            self.network.neuron_fatigue = self.network.neuron_fatigue * 0.1  # 90% clearance
            
            # Phase 4: Sleep replay of important experiences
            replay_consolidation = self._perform_sleep_replay()
            
            # Phase 4: Synaptic capture - tagged synapses get preferential consolidation
            self.network.synaptic_weights = self.network.plasticity.apply_synaptic_capture(
                self.network.synaptic_weights, 
                self.network.synaptic_tags, 
                self.network.stp_buffer
            )
            
            # Standard STP consolidation for untagged synapses
            remaining_stp = self.network.stp_buffer * (1.0 - self.network.synaptic_tags)
            stp_consolidation = remaining_stp * 0.3  # 30% of remaining STP becomes permanent
            self.network.synaptic_weights = self.network.synaptic_weights + stp_consolidation
            
            # Reset STP buffer after consolidation
            self.network.stp_buffer = self.network.stp_buffer * 0.2  # 80% clearance
            
            # Homeostatic scaling during sleep
            self.network.synaptic_weights = self.network.synaptic_weights * 0.95  # Mild scaling
            
            # Reset sleep counter
            self.sleep_cycle_counter = 0
            
            # Log sleep event
            self.fatigue_events.append({
                'step': self.total_interactions,
                'event': 'adenosine_sleep',
                'pre_fatigue': avg_fatigue,
                'post_fatigue': float(jnp.mean(self.network.neuron_fatigue)),
                'stp_consolidated': float(jnp.mean(jnp.abs(stp_consolidation))),
                'replay_consolidation': replay_consolidation,
                'n_tagged_synapses': int(jnp.sum(self.network.synaptic_tags > 0.1)),
                'experience_buffer_size': len(self.network.experience_buffer),
                'timestamp': time.time()
            })
            
            self.logger.info(f"😴 Sleep completed: fatigue cleared, STP consolidated")
            
            return True
        
        return False
    
    def _perform_sleep_replay(self) -> float:
        """
        Perform sleep replay of important experiences.
        
        During sleep, the brain replays important experiences to consolidate memories.
        This strengthens the synaptic connections involved in those experiences.
        """
        if not self.network.experience_buffer:
            return 0.0
        
        total_consolidation = 0.0
        replayed_experiences = 0
        
        # Sort experiences by importance (most important first)
        experiences = sorted(self.network.experience_buffer, 
                           key=lambda x: x['importance'], reverse=True)
        
        # Replay top experiences (limit to prevent excessive processing)
        max_replays = min(10, len(experiences))
        
        for i in range(max_replays):
            experience = experiences[i]
            
            # Generate replay activity pattern
            replay_activity = self.network.plasticity.generate_replay_activity(
                experience['activity'], 
                self.network.replay_traces
            )
            
            # Apply replay learning with reduced learning rate
            replay_modulators = experience['modulators'].copy()
            
            # Replay learning is offline (no adenosine fatigue)
            replay_modulators['adenosine'] = 0.0
            
            # Apply replay learning
            old_weights = self.network.synaptic_weights.copy()
            
            # Simulate pre-post activity for replay
            pre_activity = replay_activity
            post_activity = replay_activity * 0.8  # Slightly compressed
            
            # Apply plasticity during replay
            # Create eligibility trace specifically for replay
            replay_eligibility = jnp.outer(post_activity, pre_activity) * 0.5
            
            self.network.synaptic_weights = self.network.plasticity.update_weights(
                self.network.synaptic_weights,
                pre_activity,
                post_activity,
                replay_eligibility,
                replay_modulators
            )
            
            # Calculate consolidation strength
            consolidation = float(jnp.sum(jnp.abs(self.network.synaptic_weights - old_weights)))
            total_consolidation += consolidation
            replayed_experiences += 1
            
            # Update replay traces
            self.network.replay_traces = self.network.replay_traces * 0.95 + jnp.outer(replay_activity, replay_activity) * 0.05
        
        # Log replay activity
        if replayed_experiences > 0:
            avg_consolidation = total_consolidation / replayed_experiences
            self.logger.info(f"🔄 Sleep replay: {replayed_experiences} experiences, "
                           f"avg consolidation: {avg_consolidation:.6f}")
        
        return total_consolidation
    
    def _compute_rpe_from_feedback(self, feedback: str) -> float:
        """
        Compute reward prediction error (RPE) from teacher feedback.
        
        Returns:
            RPE ∈ {-1, 0, +1}:
            - +1.0: Strong positive feedback (unexpected reward)
            - +0.5: Mild positive feedback
            - 0.0: Neutral feedback (no prediction error)
            - -0.5: Mild negative feedback
            - -1.0: Strong negative feedback (negative prediction error)
        """
        feedback_lower = feedback.lower()
        
        # Strong positive indicators
        strong_positive = ["excellent", "perfect", "wonderful", "amazing", "fantastic", "brilliant"]
        # Mild positive indicators  
        mild_positive = ["good", "great", "nice", "well done", "correct", "right", "yes", "👍", "👏", "🎉"]
        # Strong negative indicators (use word boundaries to avoid partial matches)
        strong_negative = ["wrong", "incorrect", "bad", "terrible", "awful", " no ", "stop"]
        # Mild negative indicators (check specific phrases first)
        mild_negative = ["try again", "not quite", "almost", "close", "hmm", "⚠️"]
        
        # Check for specific mild negative phrases first (to avoid substring conflicts)
        if any(phrase in feedback_lower for phrase in mild_negative):
            return -0.5
        # Then check for strong signals
        elif any(pos in feedback_lower for pos in strong_positive):
            return 1.0
        elif any(neg in feedback_lower for neg in strong_negative):
            return -1.0
        # Then check for mild positive
        elif any(pos in feedback_lower for pos in mild_positive):
            return 0.5
        else:
            # Neutral feedback - no prediction error
            return 0.0
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context."""
        recent_entries = self.current_conversation[-4:]
        context_parts = []
        
        for entry in recent_entries:
            if entry['role'] in ['input', 'teacher']:
                context_parts.append(entry['text'])
        
        return " ".join(context_parts)
    
    def _monitor_plasticity(self):
        """Monitor and log plasticity changes."""
        stats = self.network.get_plasticity_stats()
        
        self.logger.info(f"Plasticity monitoring (step {stats['learning_step']}):")
        self.logger.info(f"  Connectivity: {stats['weights']['connectivity']:.3f}")
        self.logger.info(f"  Active neurons: {stats['activity']['n_active_neurons']}")
        self.logger.info(f"  Token mappings: {stats['token_mappings']}")
        
        # Trigger homeostatic reset if needed
        if stats['weights']['connectivity'] > 0.5:  # Too dense
            self.logger.info("Network too dense, applying homeostatic reset")
            self.network.reset_plasticity()
        elif stats['activity']['n_active_neurons'] < 10:  # Too quiet
            self.logger.info("Network too quiet, boosting activity")
            self.network.plasticity.homeostatic_rate *= 1.1
    
    def _save_memory(self):
        """Save learner's plasticity state."""
        plasticity_stats = self.network.get_plasticity_stats()
        
        memory_data = {
            'conversation_count': self.conversation_count,
            'total_interactions': self.total_interactions,
            'learned_associations': self.learned_associations,
            'plasticity_events': self.plasticity_events[-20:],  # Recent events
            'plasticity_stats': plasticity_stats,
            'tokenizer_stats': self.tokenizer.get_pattern_stats(),
            'teacher_stage': self.teacher.current_stage,
            'network_config': {
                'n_neurons': self.network.n_neurons,
                'learning_step': self.network.learning_step
            },
            # Phase 2 state
            'pattern_familiarity': dict(list(self.pattern_familiarity.items())[-50:]),  # Recent patterns
            'attention_context': self.attention_context[-10:],  # Recent context
            'novelty_history': self.novelty_history[-20:],  # Recent novelty
            # Phase 3 state
            'sleep_cycle_counter': self.sleep_cycle_counter,
            'fatigue_events': self.fatigue_events[-10:],  # Recent fatigue events
            # Phase 4 state
            'replay_events': self.replay_events[-10:],  # Recent replay events
            'synaptic_capture_events': self.synaptic_capture_events[-10:],  # Recent capture events
            'timestamp': time.time()
        }
        
        # Save main memory
        with open(self.config.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2, default=str)
        
        # Save tokenizer
        vocab_file = self.config.memory_file.replace('.json', '_vocab.json')
        self.tokenizer.save_vocabulary(vocab_file)
        
        self.logger.info(f"Plastic memory saved: {self.conversation_count} conversations")
    
    def _save_network_state(self):
        """Save the network's synaptic state."""
        try:
            # Create timestamped filename
            timestamp = int(time.time())
            network_file = self.config.network_file.replace('.npz', f'_step_{self.network.learning_step}_{timestamp}.npz')
            
            # Save network state
            self.network.save_network_state(network_file)
            
            # Also save as latest
            latest_file = self.config.network_file.replace('.npz', '_latest.npz')
            self.network.save_network_state(latest_file)
            
            self.logger.info(f"Network state saved at step {self.network.learning_step}")
            
        except Exception as e:
            self.logger.error(f"Failed to save network state: {e}")
    
    def load_memory(self, memory_file: str = None):
        """Load previous plasticity state."""
        memory_file = memory_file or self.config.memory_file
        
        try:
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
            
            self.conversation_count = memory_data.get('conversation_count', 0)
            self.total_interactions = memory_data.get('total_interactions', 0)
            self.learned_associations = memory_data.get('learned_associations', {})
            self.plasticity_events = memory_data.get('plasticity_events', [])
            
            # Restore Phase 2 state
            self.pattern_familiarity = memory_data.get('pattern_familiarity', {})
            self.attention_context = memory_data.get('attention_context', [])
            self.novelty_history = memory_data.get('novelty_history', [])
            
            # Restore Phase 3 state
            self.sleep_cycle_counter = memory_data.get('sleep_cycle_counter', 0)
            self.fatigue_events = memory_data.get('fatigue_events', [])
            
            # Restore Phase 4 state
            self.replay_events = memory_data.get('replay_events', [])
            self.synaptic_capture_events = memory_data.get('synaptic_capture_events', [])
            
            # Load tokenizer
            vocab_file = memory_file.replace('.json', '_vocab.json')
            if Path(vocab_file).exists():
                self.tokenizer.load_vocabulary(vocab_file)
            
            # Try to load network state
            latest_network_file = self.config.network_file.replace('.npz', '_latest.npz')
            if Path(latest_network_file).exists():
                self.network.load_network_state(latest_network_file)
                self.logger.info("Network synaptic state restored")
            else:
                self.logger.info("No network state found, starting with fresh synapses")
            
            self.logger.info(f"Plastic memory loaded: {self.conversation_count} conversations")
            
        except FileNotFoundError:
            self.logger.info("No previous memory found, starting with fresh plasticity")
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        plasticity_stats = self.network.get_plasticity_stats()
        tokenizer_stats = self.tokenizer.get_pattern_stats()
        
        return {
            'conversations': self.conversation_count,
            'total_interactions': self.total_interactions,
            'vocabulary_size': self.tokenizer.get_vocab_size(),
            'network_neurons': self.network.n_neurons,  # Fixed size!
            'connectivity': plasticity_stats['weights']['connectivity'],
            'active_neurons': plasticity_stats['activity']['n_active_neurons'],
            'plasticity_step': plasticity_stats['learning_step'],
            'learned_associations': len(self.learned_associations),
            'teacher_stage': self.teacher.current_stage,
            'tokenizer_stats': tokenizer_stats,
            'plasticity_events': len(self.plasticity_events),
            'recent_plasticity': self.plasticity_events[-3:] if self.plasticity_events else []
        }