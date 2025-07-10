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
        Process input and generate response with real-time plasticity and adaptive teacher feedback.
        
        This is where the key difference is: instead of growing the network,
        we change the synaptic connections through plasticity. Now enhanced with
        adaptive teacher integration for continuous symbiotic learning.
        """
        self.total_interactions += 1
        
        # Record input
        self.current_conversation.append({"role": "input", "text": input_text})
        
        # Learn from input through plasticity (not network growth!)
        self._learn_through_plasticity(input_text)
        
        # Update teacher with current network state for adaptive prompting
        if self.total_interactions % 10 == 0:  # Every 10 interactions to avoid overhead
            network_stats = self.get_learning_stats()
            simplified_stats = {
                'vocabulary_size': network_stats['vocabulary_size'],
                'connectivity': network_stats['connectivity'],
                'active_neurons': network_stats['active_neurons'],
                'plasticity_step': network_stats['plasticity_step'],
                'maturity_factor': network_stats.get('maturity_factor', 1.0)
            }
            self.teacher.set_learner_stats(simplified_stats)
        
        # Generate response using current network state
        response = self._generate_plastic_response(input_text)
        
        # Record response
        self.current_conversation.append({"role": "learner", "text": response})
        
        # Get adaptive teacher feedback (now enhanced with progress tracking)
        teacher_feedback = self.teacher.respond_to_student(response)
        self.current_conversation.append({"role": "teacher", "text": teacher_feedback})
        
        # Learn from feedback through plasticity
        self._learn_from_feedback_plastic(response, teacher_feedback)
        
        # Monitor plasticity changes and auto-adjust if needed
        if self.total_interactions % 5 == 0:
            self._monitor_plasticity()
        
        # Auto-adjust learning parameters based on progress
        if self.total_interactions % 100 == 0:
            self._auto_adjust_learning_rates()
        
        # Save network state more frequently
        if self.total_interactions % self.config.network_save_interval == 0:
            self._save_network_state()
        
        # Save progress periodically based on interactions
        if self.total_interactions % self.config.save_interval == 0:
            self._save_memory()
        
        # Log with enhanced metrics
        teacher_metrics = self.teacher.get_learning_metrics()
        self.logger.info(f"Input: {input_text}")
        self.logger.info(f"Learner: {response}")
        self.logger.info(f"Teacher: {teacher_feedback}")
        self.logger.info(f"Teacher metrics - Stage: {teacher_metrics['current_stage']}, "
                        f"Vocab: {teacher_metrics['vocabulary_size']}, "
                        f"Improvement streak: {teacher_metrics['improvement_streak']}")
        
        return response
    
    def _learn_through_plasticity(self, text: str):
        """
        Learn from text through synaptic plasticity, not network growth.
        
        This is the key insight: real brains learn by changing connections,
        not by adding neurons.
        """
        # Encode text
        token_ids = self.tokenizer.encode(text)
        
        if len(token_ids) > 0:
            # Process through network with plasticity enabled
            result = self.network.process_tokens(token_ids, learning=True)
            
            # Track plasticity event
            plasticity_stats = self.network.get_plasticity_stats()
            
            # Record significant plasticity changes
            if len(self.plasticity_events) == 0 or self._is_significant_change(plasticity_stats):
                self.plasticity_events.append({
                    'step': self.total_interactions,
                    'text': text,
                    'stats': plasticity_stats,
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
            
            # Generate using current plastic network state WITH exploration plasticity
            # Store original plasticity rates
            original_ltp = self.network.plasticity.ltp_rate
            original_ltd = self.network.plasticity.ltd_rate
            
            # Enable reduced plasticity during generation for exploration
            self.network.plasticity.ltp_rate *= 0.5  # Reduced learning during exploration
            self.network.plasticity.ltd_rate *= 0.5  # Reduced forgetting during exploration
            
            generated_tokens = self.network.generate_tokens(
                context_tokens[-8:],  # Recent context
                max_length=8,        # Short responses
                learning=True        # Enable exploration plasticity
            )
            
            # Restore original plasticity rates
            self.network.plasticity.ltp_rate = original_ltp
            self.network.plasticity.ltd_rate = original_ltd
            
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
        Learn from teacher feedback through plasticity adjustments.
        
        Key improvement: Implement reward-modulated replay that directly
        associates context with response based on feedback valence.
        Enhanced with special reinforcement for proper pause usage.
        """
        # Analyze feedback sentiment
        positive_indicators = ["good", "great", "yes", "right", "nice", "excellent"]
        negative_indicators = ["no", "try again", "not quite", "wrong"]
        
        feedback_lower = feedback.lower()
        is_positive = any(pos in feedback_lower for pos in positive_indicators)
        is_negative = any(neg in feedback_lower for neg in negative_indicators)
        
        # Special reinforcement for pause usage
        response_has_spaces = ' ' in response.strip()
        feedback_mentions_clarity = any(word in feedback_lower for word in 
                                      ["clear", "space", "pause", "better", "easier", "understand"])
        
        # Boost positive feedback if response has proper spacing
        if is_positive and response_has_spaces:
            is_positive = "enhanced_positive"  # Mark for stronger reinforcement
            self.logger.debug("Enhanced positive feedback for spaced response")
        
        # Provide corrective feedback if response lacks spaces but should have them
        if not response_has_spaces and feedback_mentions_clarity:
            # Artificial negative signal for concatenated responses
            is_negative = True
            self.logger.debug("Negative signal for concatenated response")
        
        # CRITICAL IMPROVEMENT: Reward-modulated replay for context-response associations
        context = self._get_conversation_context()
        context_tokens = self.tokenizer.encode(context)
        response_tokens = self.tokenizer.encode(response)
        
        # Create combined context + response sequence for joint learning
        combined_tokens = context_tokens + response_tokens
        
        # Update neuromodulation based on feedback
        if is_positive == "enhanced_positive":
            reward_signal = 1.0  # Strong reward for spaced responses
            novelty_signal = 0.5  # Moderate novelty
        elif is_positive:
            reward_signal = 0.7  # Good reward
            novelty_signal = 0.3  # Some novelty
        elif is_negative:
            reward_signal = -0.5  # Negative reward
            novelty_signal = 0.8  # High novelty (need to learn)
        else:
            reward_signal = 0.0  # Neutral
            novelty_signal = 0.2  # Low novelty
            
        # Update neuromodulators
        self.network.plasticity.neuromodulator.update_dopamine(reward_signal)
        self.network.plasticity.neuromodulator.update_acetylcholine(novelty_signal)
        
        # Set error signal for error-driven plasticity
        if is_negative:
            self.network.set_error_signal(-reward_signal)  # Negative error for wrong responses
        else:
            self.network.set_error_signal(0.0)  # No error for correct responses
        
        if combined_tokens:
            # Store original plasticity rates
            original_ltp = self.network.plasticity.ltp_rate
            original_ltd = self.network.plasticity.ltd_rate
            
            # Modulate plasticity based on feedback valence
            if is_positive == "enhanced_positive":
                # Extra strong reinforcement for spaced responses
                self.network.plasticity.ltp_rate *= 2.0  # Strong boost for good spaced responses
                self.network.plasticity.ltd_rate *= 0.6  # Minimal forgetting
                self.logger.debug("Enhanced positive feedback: strongly reinforcing spaced response")
                
                # Specifically strengthen pathways leading to pause tokens
                pause_tokens = self.tokenizer.encode(' ')  # Get pause token representations
                if pause_tokens:
                    # Process pause tokens with extra strengthening
                    self.network.process_tokens(pause_tokens, learning=True)
                    
            elif is_positive:
                # Strengthen context-response pathway for good responses
                self.network.plasticity.ltp_rate *= 1.5  # Boost learning significantly
                self.network.plasticity.ltd_rate *= 0.8  # Reduce forgetting
                self.logger.debug("Positive feedback: strengthening context-response pathway")
            elif is_negative:
                # Weaken context-response pathway for bad responses
                self.network.plasticity.ltp_rate *= 0.7  # Reduce learning
                self.network.plasticity.ltd_rate *= 1.5  # Boost forgetting
                self.logger.debug("Negative feedback: weakening context-response pathway")
                
                # If response lacked spaces, specifically weaken non-pause pathways
                if not response_has_spaces:
                    # Gentle weakening of concatenated patterns
                    self.network.plasticity.ltd_rate *= 1.2  # Slight additional forgetting
                    self.logger.debug("Weakening concatenated response patterns")
            
            # Process the combined sequence to create direct associations
            self.network.process_tokens(combined_tokens, learning=True)
            
            # Reset plasticity rates
            self.network.plasticity.ltp_rate = original_ltp
            self.network.plasticity.ltd_rate = original_ltd
        
        # Now process feedback separately (teacher correction patterns)
        feedback_tokens = self.tokenizer.encode(feedback)
        if feedback_tokens:
            # Always learn from teacher feedback with slight LTP boost (corrections)
            self.network.plasticity.ltp_rate *= 1.2
            self.network.process_tokens(feedback_tokens, learning=True)
            self.network.plasticity.ltp_rate /= 1.2
            
        # Add successful sequences to replay buffer for consolidation
        if combined_tokens:
            if is_positive == "enhanced_positive":
                self.network.add_to_replay_buffer(combined_tokens, reward=1.0)
            elif is_positive:
                self.network.add_to_replay_buffer(combined_tokens, reward=0.7)
            elif is_negative:
                # Still add negative examples for learning, but with negative reward
                self.network.add_to_replay_buffer(combined_tokens, reward=-0.3)
        
        # Record association
        if response not in self.learned_associations:
            self.learned_associations[response] = []
        
        self.learned_associations[response].append({
            'feedback': feedback,
            'positive': is_positive,
            'negative': is_negative,
            'plasticity_step': self.network.learning_step,
            'timestamp': time.time()
        })
    
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
        """Save learner's plasticity state and teacher adaptive metrics."""
        plasticity_stats = self.network.get_plasticity_stats()
        teacher_metrics = self.teacher.get_learning_metrics()
        
        memory_data = {
            'conversation_count': self.conversation_count,
            'total_interactions': self.total_interactions,
            'learned_associations': self.learned_associations,
            'plasticity_events': self.plasticity_events[-20:],  # Recent events
            'plasticity_stats': plasticity_stats,
            'tokenizer_stats': self.tokenizer.get_pattern_stats(),
            'teacher_stage': self.teacher.current_stage,
            'teacher_metrics': teacher_metrics,  # Enhanced teacher tracking
            'network_config': {
                'n_neurons': self.network.n_neurons,
                'learning_step': self.network.learning_step
            },
            'symbiotic_learning': {
                'teacher_learner_vocab_ratio': teacher_metrics['vocabulary_size'] / max(1, self.tokenizer.get_vocab_size()),
                'plasticity_teacher_alignment': plasticity_stats['weights']['connectivity'] * teacher_metrics['improvement_streak'],
                'adaptive_feedback_quality': teacher_metrics.get('vocabulary_growth_rate', 0.0)
            },
            'timestamp': time.time()
        }
        
        # Save main memory
        with open(self.config.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2, default=str)
        
        # Save tokenizer
        vocab_file = self.config.memory_file.replace('.json', '_vocab.json')
        self.tokenizer.save_vocabulary(vocab_file)
        
        # Save teacher conversation state
        teacher_conversation_file = self.config.memory_file.replace('.json', '_teacher_conversation.json')
        self.teacher.save_conversation(teacher_conversation_file)
        
        self.logger.info(f"Plastic memory saved: {self.conversation_count} conversations, "
                        f"Teacher stage: {teacher_metrics['current_stage']}, "
                        f"Symbiotic alignment: {memory_data['symbiotic_learning']['plasticity_teacher_alignment']:.3f}")
    
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
            
            # Load teacher conversation state
            teacher_conversation_file = memory_file.replace('.json', '_teacher_conversation.json')
            if Path(teacher_conversation_file).exists():
                self.teacher.load_conversation(teacher_conversation_file)
                self.logger.info("Teacher conversation state restored")
            
            self.logger.info(f"Plastic memory loaded: {self.conversation_count} conversations")
            
        except FileNotFoundError:
            self.logger.info("No previous memory found, starting with fresh plasticity")
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics with enhanced monitoring."""
        plasticity_stats = self.network.get_plasticity_stats()
        tokenizer_stats = self.tokenizer.get_pattern_stats()
        
        # Calculate additional learning metrics
        weight_change_norm = 0.0
        if len(self.plasticity_events) >= 2:
            recent_connectivity = [event['stats']['weights']['connectivity'] for event in self.plasticity_events[-5:]]
            weight_change_norm = np.std(recent_connectivity) if len(recent_connectivity) > 1 else 0.0
        
        # Response-feedback similarity for credit assignment monitoring
        response_feedback_overlap = 0.0
        if self.learned_associations:
            recent_associations = list(self.learned_associations.items())[-10:]
            overlaps = []
            for response, feedback_list in recent_associations:
                for feedback_data in feedback_list[-3:]:  # Recent feedback for this response
                    feedback = feedback_data['feedback']
                    # Simple token overlap measure
                    response_tokens = set(self.tokenizer.encode(response))
                    feedback_tokens = set(self.tokenizer.encode(feedback))
                    if response_tokens and feedback_tokens:
                        overlap = len(response_tokens & feedback_tokens) / len(response_tokens | feedback_tokens)
                        overlaps.append(overlap)
            response_feedback_overlap = np.mean(overlaps) if overlaps else 0.0
        
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
            'recent_plasticity': self.plasticity_events[-3:] if self.plasticity_events else [],
            # Enhanced monitoring metrics
            'weight_change_norm': weight_change_norm,
            'response_feedback_overlap': response_feedback_overlap,
            'maturity_factor': plasticity_stats.get('maturity_factor', 1.0),
            'tonic_strength': plasticity_stats.get('tonic_strength', 0.0),
            'current_ltp_rate': self.network.plasticity.ltp_rate,
            'current_ltd_rate': self.network.plasticity.ltd_rate
        }
    
    def _auto_adjust_learning_rates(self):
        """Auto-adjust learning rates based on progress monitoring."""
        stats = self.get_learning_stats()
        
        # If responses don't show good overlap with feedback after many turns, boost rates
        if (self.total_interactions > 1000 and 
            stats['response_feedback_overlap'] < 0.2 and 
            stats['learned_associations'] < 10):
            
            self.network.plasticity.base_ltp_rate *= 1.1
            self.network.plasticity.base_ltd_rate *= 1.05
            self.logger.info(f"Auto-boosted learning rates: LTP={self.network.plasticity.base_ltp_rate:.4f}")
        
        # If connectivity stays too low for too long, reduce decay
        if (stats['connectivity'] < 0.03 and 
            self.total_interactions > 500):
            
            self.network.plasticity.decay_rate *= 0.9
            self.logger.info(f"Reduced decay rate to {self.network.plasticity.decay_rate:.4f}")