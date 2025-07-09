"""
Continual Learning Conversational Agent

This implements the core goal: a network that learns during every conversation,
not through pre-training. The network starts nearly blank and grows through experience.

Key principles:
- No pre-training phase
- Learn during every interaction
- Real-time weight updates via Hebbian learning
- Vocabulary grows through exposure
- Teacher provides real-time feedback during conversations
"""

import time
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from models.sequence_snn import SequenceHebSNN
from utils.biologically_inspired_tokenizer import BiologicalTokenizer
from utils.llm_teacher import LLMTeacher, TeacherConfig


@dataclass
class LearnerConfig:
    """Configuration for the continual learner."""
    
    # Network parameters (start small, grow as needed)
    initial_vocab_size: int = 50   # Start very small
    max_vocab_size: int = 500     # Limited growth to prevent memory issues
    max_seq_length: int = 32      # Start with short sequences
    column_size: int = 15         # Smaller columns to save memory
    n_layers: int = 2             # Fewer layers initially
    max_network_neurons: int = 5000  # Limit total network size
    
    # Learning parameters
    learning_rate: float = 0.05   # Higher learning rate for faster adaptation
    hebbian_strength: float = 1.0
    adaptation_rate: float = 0.1  # How quickly to adapt to new patterns
    
    # Interaction parameters
    teacher_api_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model: str = "gemma-3-27b-it-qat"
    teacher_temperature: float = 0.7
    
    # Growth parameters
    vocab_growth_threshold: int = 5    # Add new patterns after N exposures
    network_growth_episodes: int = 50  # Expand network every N conversations
    
    # Memory and saving
    save_interval: int = 10  # Save progress every N conversations
    memory_file: str = "experiments/conversational_learning/memory/learner_memory.json"
    log_file: str = "experiments/conversational_learning/logs/continual_learning.log"


class ContinualLearner:
    """
    A conversational agent that learns continuously through every interaction.
    
    Unlike traditional training, this agent:
    1. Starts with minimal knowledge
    2. Learns during each conversation
    3. Grows its vocabulary and capabilities through exposure
    4. Updates weights via Hebbian learning in real-time
    """
    
    def __init__(self, config: LearnerConfig = None):
        self.config = config or LearnerConfig()
        
        # Setup directories
        Path(self.config.memory_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tokenizer (starts small, grows through experience)
        self.tokenizer = BiologicalTokenizer(max_vocab_size=self.config.max_vocab_size)
        
        # Initialize teacher
        teacher_config = TeacherConfig(
            api_url=self.config.teacher_api_url,
            model=self.config.teacher_model,
            temperature=self.config.teacher_temperature
        )
        self.teacher = LLMTeacher(teacher_config)
        
        # Initialize network (starts small)
        self.network = SequenceHebSNN(
            vocab_size=self.config.initial_vocab_size,
            max_seq_length=self.config.max_seq_length,
            column_size=self.config.column_size,
            n_layers=self.config.n_layers
        )
        
        # Learning state
        self.conversation_count = 0
        self.total_interactions = 0
        self.learned_associations = {}  # What the network has learned
        self.experience_memory = []     # Recent experiences for adaptation
        
        # Conversation state
        self.current_conversation = []
        self.is_learning = True
        
        self.logger.info("Continual Learner initialized")
        self.logger.info(f"Starting vocab: {self.tokenizer.get_vocab_size()} patterns")
        self.logger.info(f"Network: {self.network.n_neurons} neurons")
    
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
    
    def process_input_and_respond(self, human_input: str) -> str:
        """
        Process human input and generate response while learning.
        This is where the real-time learning happens.
        """
        self.total_interactions += 1
        
        # Record human input
        self.current_conversation.append({"role": "human", "text": human_input})
        
        # Learn from the input (Hebbian learning)
        self._learn_from_text(human_input, is_input=True)
        
        # Generate response based on current network state
        response = self._generate_response(human_input)
        
        # Record our response
        self.current_conversation.append({"role": "learner", "text": response})
        
        # Get teacher feedback on our response
        teacher_feedback = self.teacher.respond_to_student(response)
        self.current_conversation.append({"role": "teacher", "text": teacher_feedback})
        
        # Learn from teacher feedback
        self._learn_from_feedback(response, teacher_feedback)
        
        # Adapt network if needed
        if self.total_interactions % 10 == 0:
            self._adapt_network()
        
        # Save progress periodically
        if self.conversation_count % self.config.save_interval == 0:
            self._save_memory()
        
        self.logger.info(f"Human: {human_input}")
        self.logger.info(f"Learner: {response}")
        self.logger.info(f"Teacher: {teacher_feedback}")
        
        return response
    
    def _learn_from_text(self, text: str, is_input: bool = True):
        """
        Learn from text using Hebbian mechanisms.
        This is real-time learning, not pre-training.
        """
        # Encode text with current tokenizer (may learn new patterns)
        token_ids = self.tokenizer.encode(text)
        
        # Process through network to create neural activity
        if len(token_ids) > 0:
            processing_result = self.network.process_sequence(
                token_ids, 
                n_steps=5  # Quick processing for real-time
            )
            
            # Store experience for later adaptation
            experience = {
                'text': text,
                'tokens': token_ids,
                'activity': processing_result['activity'],
                'timestamp': time.time(),
                'is_input': is_input
            }
            self.experience_memory.append(experience)
            
            # Keep memory manageable
            if len(self.experience_memory) > 100:
                self.experience_memory.pop(0)
    
    def _generate_response(self, context: str) -> str:
        """
        Generate a response based on current network state.
        This uses the network's learned patterns, not pre-trained knowledge.
        """
        try:
            # Get context from recent conversation
            recent_context = self._get_conversation_context()
            context_tokens = self.tokenizer.encode(recent_context, add_special_tokens=False)
            
            # Generate using current network state
            generated_tokens = self.network.generate_sequence(
                prompt_tokens=context_tokens[-10:],  # Recent context
                max_length=15,  # Short responses initially
                temperature=1.5  # High temperature for exploration
            )
            
            # Remove context tokens to get just the response
            if len(generated_tokens) > len(context_tokens):
                response_tokens = generated_tokens[len(context_tokens):]
            else:
                response_tokens = generated_tokens[-5:]  # Fallback
            
            # Decode response
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # If response is empty or too short, use simple fallback
            if len(response) < 2:
                simple_responses = ["hi", "ok", "yes", "mm"]
                # Choose based on network activity
                vocab_activity = self.network.get_column_activities(0, self.network.get_activity())
                if len(vocab_activity) > 0:
                    choice_idx = int(np.argmax(vocab_activity)) % len(simple_responses)
                    response = simple_responses[choice_idx]
                else:
                    response = "hi"
            
            return response
            
        except Exception as e:
            self.logger.warning(f"Response generation failed: {e}")
            # Very simple fallback
            return "mm"
    
    def _learn_from_feedback(self, our_response: str, teacher_feedback: str):
        """
        Learn from teacher feedback to improve future responses.
        """
        # Analyze if feedback is positive or negative
        positive_indicators = ["good", "great", "yes", "right", "nice", "well done"]
        negative_indicators = ["no", "try again", "not quite", "wrong"]
        
        feedback_lower = teacher_feedback.lower()
        is_positive = any(pos in feedback_lower for pos in positive_indicators)
        is_negative = any(neg in feedback_lower for neg in negative_indicators)
        
        # Encode our response and teacher feedback
        response_tokens = self.tokenizer.encode(our_response)
        feedback_tokens = self.tokenizer.encode(teacher_feedback)
        
        # Create learning signal based on feedback
        if is_positive:
            # Strengthen associations for positive feedback
            learning_signal = 1.5
        elif is_negative:
            # Weaken associations for negative feedback
            learning_signal = 0.5
        else:
            # Neutral - slight strengthening for exposure
            learning_signal = 1.1
        
        # Apply learning signal (this would modify network weights in a full implementation)
        # For now, just record the association
        if our_response not in self.learned_associations:
            self.learned_associations[our_response] = {
                'feedback_history': [],
                'strength': 1.0
            }
        
        self.learned_associations[our_response]['feedback_history'].append({
            'feedback': teacher_feedback,
            'positive': is_positive,
            'negative': is_negative,
            'timestamp': time.time()
        })
        
        # Update association strength
        self.learned_associations[our_response]['strength'] *= learning_signal
    
    def _get_conversation_context(self) -> str:
        """Get recent conversation context for response generation."""
        # Get last few exchanges
        recent_entries = self.current_conversation[-6:]  # Last 3 exchanges
        context_parts = []
        
        for entry in recent_entries:
            if entry['role'] == 'human':
                context_parts.append(entry['text'])
            elif entry['role'] == 'teacher':
                context_parts.append(entry['text'])
        
        return " ".join(context_parts)
    
    def _adapt_network(self):
        """
        Adapt the network based on recent experience.
        This allows the network to grow and specialize.
        """
        # Adapt tokenizer vocabulary based on usage
        self.tokenizer.adapt_vocabulary(min_usage=2)
        
        # Check if we need to expand network capacity
        current_vocab = self.tokenizer.get_vocab_size()
        if current_vocab > self.network.vocab_size * 0.8:  # Approaching capacity
            self._expand_network_capacity()
        
        # Log adaptation
        stats = self.tokenizer.get_pattern_stats()
        self.logger.info(f"Network adaptation: vocab={current_vocab}, "
                        f"total_usage={stats['total_usage']}")
    
    def _expand_network_capacity(self):
        """
        Expand network capacity to handle larger vocabulary.
        This simulates biological neural growth but respects memory limits.
        """
        old_vocab_size = self.network.vocab_size
        new_vocab_size = min(old_vocab_size * 2, self.config.max_vocab_size)
        
        # Check if expansion would exceed memory limits
        estimated_neurons = (new_vocab_size + new_vocab_size//2 + new_vocab_size//4) * self.config.column_size
        
        if (new_vocab_size > old_vocab_size and 
            estimated_neurons <= self.config.max_network_neurons):
            
            self.logger.info(f"Expanding network capacity: {old_vocab_size} -> {new_vocab_size}")
            self.logger.info(f"Estimated neurons: {estimated_neurons}")
            
            # Create new larger network (in a full implementation, we'd transfer weights)
            self.network = SequenceHebSNN(
                vocab_size=new_vocab_size,
                max_seq_length=self.config.max_seq_length,
                column_size=self.config.column_size,
                n_layers=self.config.n_layers
            )
        else:
            self.logger.info(f"Network expansion limited by memory constraints")
            self.logger.info(f"Current size: {old_vocab_size}, estimated neurons: {estimated_neurons}")
    
    def _save_memory(self):
        """Save the learner's accumulated memory and experience."""
        memory_data = {
            'conversation_count': self.conversation_count,
            'total_interactions': self.total_interactions,
            'learned_associations': self.learned_associations,
            'tokenizer_stats': self.tokenizer.get_pattern_stats(),
            'network_config': {
                'vocab_size': self.network.vocab_size,
                'n_neurons': self.network.n_neurons,
                'n_layers': self.network.n_layers
            },
            'teacher_stage': self.teacher.current_stage,
            'recent_experiences': self.experience_memory[-20:],  # Save recent experiences
            'timestamp': time.time()
        }
        
        # Save main memory
        with open(self.config.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2, default=str)
        
        # Save tokenizer vocabulary
        vocab_file = self.config.memory_file.replace('.json', '_vocab.json')
        self.tokenizer.save_vocabulary(vocab_file)
        
        self.logger.info(f"Memory saved: {self.conversation_count} conversations, "
                        f"{self.total_interactions} interactions")
    
    def load_memory(self, memory_file: str = None):
        """Load previously saved memory and experience."""
        memory_file = memory_file or self.config.memory_file
        
        try:
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
            
            self.conversation_count = memory_data.get('conversation_count', 0)
            self.total_interactions = memory_data.get('total_interactions', 0)
            self.learned_associations = memory_data.get('learned_associations', {})
            
            # Load tokenizer vocabulary
            vocab_file = memory_file.replace('.json', '_vocab.json')
            if Path(vocab_file).exists():
                self.tokenizer.load_vocabulary(vocab_file)
            
            # Rebuild network with correct size
            network_config = memory_data.get('network_config', {})
            if 'vocab_size' in network_config:
                self.network = SequenceHebSNN(
                    vocab_size=network_config['vocab_size'],
                    max_seq_length=self.config.max_seq_length,
                    column_size=self.config.column_size,
                    n_layers=network_config.get('n_layers', self.config.n_layers)
                )
            
            self.logger.info(f"Memory loaded: {self.conversation_count} conversations")
            
        except FileNotFoundError:
            self.logger.info("No previous memory found, starting fresh")
        except Exception as e:
            self.logger.warning(f"Failed to load memory: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about what the learner has learned."""
        return {
            'conversations': self.conversation_count,
            'total_interactions': self.total_interactions,
            'vocabulary_size': self.tokenizer.get_vocab_size(),
            'network_neurons': self.network.n_neurons,
            'learned_associations': len(self.learned_associations),
            'teacher_stage': self.teacher.current_stage,
            'tokenizer_stats': self.tokenizer.get_pattern_stats(),
            'recent_experiences': len(self.experience_memory)
        }