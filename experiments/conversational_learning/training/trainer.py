"""
Conversational Trainer

Main training loop that orchestrates the conversation between the LLM teacher 
and the Hebbian student network. Implements a "mother teaching baby" paradigm.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from ..models.sequence_snn import SequenceHebSNN
from ..utils.llm_teacher import LLMTeacher, TeacherConfig
from ..utils.tokenizer import BiologicalTokenizer, SequenceProcessor


@dataclass
class TrainingConfig:
    """Configuration for conversational training."""
    
    # Network parameters
    vocab_size: int = 1000
    max_seq_length: int = 64
    column_size: int = 50
    n_layers: int = 3
    batch_size: int = 8
    
    # Training parameters
    max_episodes: int = 1000
    steps_per_episode: int = 20
    conversation_turns_per_episode: int = 5
    
    # Learning parameters
    learning_rate: float = 0.01
    temperature: float = 1.0
    context_length: int = 10
    
    # Teacher parameters
    teacher_api_url: str = "http://localhost:1234/v1/chat/completions"
    teacher_model: str = "gemma-3-27b-it-qat"
    teacher_temperature: float = 0.7
    
    # Logging and saving
    save_interval: int = 50
    log_interval: int = 10
    save_conversations: bool = True
    
    # Paths
    checkpoint_dir: str = "experiments/conversational_learning/checkpoints"
    log_dir: str = "experiments/conversational_learning/logs"
    data_dir: str = "experiments/conversational_learning/data"


class ConversationalTrainer:
    """
    Trainer that implements conversational learning between LLM teacher and Hebbian student.
    
    Training loop:
    1. Teacher starts conversation or responds to student
    2. Student processes teacher's message and generates response
    3. Teacher responds to student's attempt
    4. Student learns from the interaction
    5. Repeat
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.tokenizer = BiologicalTokenizer(max_vocab_size=config.vocab_size)
        self.sequence_processor = SequenceProcessor(self.tokenizer)
        
        # Initialize teacher
        teacher_config = TeacherConfig(
            api_url=config.teacher_api_url,
            model=config.teacher_model,
            temperature=config.teacher_temperature
        )
        self.teacher = LLMTeacher(teacher_config)
        
        # Initialize student network
        self.student = SequenceHebSNN(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            column_size=config.column_size,
            n_layers=config.n_layers,
            batch_size=config.batch_size
        )
        
        # Training state
        self.episode = 0
        self.total_conversations = 0
        self.student_attempts = []
        self.teacher_responses = []
        self.training_metrics = []
        
        self.logger.info("ConversationalTrainer initialized")
        self.logger.info(f"Network: {self.student.n_neurons} neurons, {self.student.vocab_size} vocab")
        self.logger.info(f"Tokenizer: {self.tokenizer.get_vocab_size()} tokens")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.log_dir) / f"training_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_student_response(self, context_tokens: List[int]) -> Tuple[str, List[int]]:
        """Generate student response based on context."""
        try:
            # Generate response tokens
            response_tokens = self.student.generate_sequence(
                prompt_tokens=context_tokens,
                max_length=20,  # Keep responses short initially
                temperature=self.config.temperature
            )
            
            # Remove prompt tokens to get just the generated part
            generated_tokens = response_tokens[len(context_tokens):]
            
            # Decode to text
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up response
            response_text = response_text.strip()
            if not response_text:
                response_text = "uh"  # Minimal response if network generates nothing
            
            return response_text, generated_tokens
            
        except Exception as e:
            self.logger.warning(f"Student generation failed: {e}")
            # Fallback response
            fallback_tokens = [self.tokenizer.char_to_id.get('h', 1), 
                             self.tokenizer.char_to_id.get('i', 1)]
            return "hi", fallback_tokens
    
    def _process_conversation_turn(self, teacher_message: str, 
                                 conversation_context: List[str]) -> Dict:
        """Process a single conversation turn."""
        
        # Encode teacher message
        teacher_tokens = self.tokenizer.encode(teacher_message, add_special_tokens=True)
        
        # Create context from conversation history
        context_tokens = self.sequence_processor.create_conversation_context(
            conversation_context, max_history=3
        )
        
        # Add current teacher message to context
        full_context = context_tokens + teacher_tokens
        if len(full_context) > self.config.context_length:
            full_context = full_context[-self.config.context_length:]
        
        # Process teacher message through student network
        self.logger.debug(f"Processing teacher message: '{teacher_message}'")
        processing_result = self.student.process_sequence(
            teacher_tokens, n_steps=self.config.steps_per_episode
        )
        
        # Generate student response
        student_response, student_tokens = self._generate_student_response(full_context)
        
        # Get network activity for analysis
        top_active_tokens = self.student.get_top_active_tokens(
            processing_result['activity'], top_k=5
        )
        
        return {
            'teacher_message': teacher_message,
            'teacher_tokens': teacher_tokens,
            'student_response': student_response,
            'student_tokens': student_tokens,
            'network_activity': processing_result['activity'],
            'top_active_tokens': top_active_tokens,
            'processing_result': processing_result
        }
    
    def _train_episode(self) -> Dict:
        """Train a single episode (conversation session)."""
        
        self.logger.info(f"Starting episode {self.episode + 1}")
        
        # Reset student state for new conversation
        self.student.reset_sequence_state()
        
        # Start conversation with teacher
        initial_message = self.teacher.start_conversation()
        conversation_context = []
        episode_metrics = {
            'episode': self.episode,
            'turns': [],
            'total_student_words': 0,
            'avg_response_length': 0,
            'network_activity_stats': [],
            'learning_progress': {}
        }
        
        # Conversation loop
        for turn in range(self.config.conversation_turns_per_episode):
            self.logger.info(f"  Turn {turn + 1}: Teacher says: '{initial_message}'")
            
            # Process conversation turn
            turn_result = self._process_conversation_turn(
                initial_message, conversation_context
            )
            
            student_response = turn_result['student_response']
            self.logger.info(f"  Turn {turn + 1}: Student says: '{student_response}'")
            
            # Store student attempt
            self.student_attempts.append(student_response)
            conversation_context.append(initial_message)
            conversation_context.append(student_response)
            
            # Get teacher's response to student
            teacher_response = self.teacher.respond_to_student(student_response)
            self.teacher_responses.append(teacher_response)
            
            self.logger.info(f"  Turn {turn + 1}: Teacher responds: '{teacher_response}'")
            
            # Update metrics
            episode_metrics['turns'].append({
                'turn': turn,
                'teacher_message': initial_message,
                'student_response': student_response,
                'teacher_response': teacher_response,
                'student_tokens': len(turn_result['student_tokens']),
                'top_active_tokens': turn_result['top_active_tokens']
            })
            
            episode_metrics['total_student_words'] += len(student_response.split())
            
            # Network activity statistics
            activity_stats = {
                'mean_activity': float(np.mean(turn_result['network_activity'])),
                'max_activity': float(np.max(turn_result['network_activity'])),
                'active_neurons': int(np.sum(turn_result['network_activity'] > 0.1)),
                'sparsity': float(np.mean(turn_result['network_activity'] > 0.1))
            }
            episode_metrics['network_activity_stats'].append(activity_stats)
            
            # Prepare for next turn
            initial_message = teacher_response
            self.total_conversations += 1
        
        # Compute episode statistics
        if episode_metrics['turns']:
            episode_metrics['avg_response_length'] = (
                episode_metrics['total_student_words'] / len(episode_metrics['turns'])
            )
        
        # Evaluate student progress
        recent_attempts = self.student_attempts[-10:] if len(self.student_attempts) >= 10 else self.student_attempts
        progress_eval = self.teacher.evaluate_student_progress(recent_attempts)
        episode_metrics['learning_progress'] = progress_eval
        
        self.training_metrics.append(episode_metrics)
        
        self.logger.info(f"Episode {self.episode + 1} completed")
        self.logger.info(f"  Student progress: {progress_eval['progress']}")
        self.logger.info(f"  Avg response length: {episode_metrics['avg_response_length']:.1f} words")
        
        return episode_metrics
    
    def train(self):
        """Run the full training loop."""
        
        self.logger.info("Starting conversational training")
        self.logger.info(f"Training for {self.config.max_episodes} episodes")
        
        start_time = time.time()
        
        try:
            for episode in range(self.config.max_episodes):
                self.episode = episode
                
                # Train episode
                episode_metrics = self._train_episode()
                
                # Logging
                if (episode + 1) % self.config.log_interval == 0:
                    self._log_progress()
                
                # Saving
                if (episode + 1) % self.config.save_interval == 0:
                    self._save_checkpoint()
                
                # Check for early stopping or adaptation
                if episode > 50:  # After some warmup
                    self._adapt_training_parameters(episode_metrics)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Final save
            self._save_checkpoint()
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.1f} seconds")
            self.logger.info(f"Total conversations: {self.total_conversations}")
    
    def _log_progress(self):
        """Log training progress."""
        if not self.training_metrics:
            return
        
        recent_metrics = self.training_metrics[-10:]  # Last 10 episodes
        
        # Compute averages
        avg_response_length = np.mean([m['avg_response_length'] for m in recent_metrics])
        avg_sparsity = np.mean([
            np.mean([stats['sparsity'] for stats in m['network_activity_stats']])
            for m in recent_metrics
        ])
        
        # Progress distribution
        progress_counts = {}
        for m in recent_metrics:
            progress = m['learning_progress'].get('progress', 'unknown')
            progress_counts[progress] = progress_counts.get(progress, 0) + 1
        
        self.logger.info(f"Progress Report (Episodes {self.episode - len(recent_metrics) + 1}-{self.episode + 1}):")
        self.logger.info(f"  Avg response length: {avg_response_length:.2f} words")
        self.logger.info(f"  Avg network sparsity: {avg_sparsity:.3f}")
        self.logger.info(f"  Progress distribution: {progress_counts}")
        self.logger.info(f"  Teacher stage: {self.teacher.current_stage}")
    
    def _adapt_training_parameters(self, episode_metrics: Dict):
        """Adapt training parameters based on progress."""
        
        # Adjust temperature based on student progress
        progress = episode_metrics['learning_progress'].get('progress', 'just_starting')
        
        if progress == 'just_starting':
            # High temperature for more exploration
            self.config.temperature = min(2.0, self.config.temperature * 1.01)
        elif progress == 'conversational':
            # Lower temperature for more focused responses
            self.config.temperature = max(0.5, self.config.temperature * 0.99)
        
        # Adjust sequence length based on student capability
        avg_length = episode_metrics['avg_response_length']
        if avg_length > 3 and self.config.max_seq_length < 100:
            # Student is generating longer responses, allow longer sequences
            self.config.max_seq_length = min(100, self.config.max_seq_length + 1)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_episode_{self.episode + 1}.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'episode': self.episode,
            'config': asdict(self.config),
            'total_conversations': self.total_conversations,
            'training_metrics': self.training_metrics[-10:],  # Save recent metrics
            'teacher_stage': self.teacher.current_stage,
            'teacher_turns': self.teacher.turns_in_stage,
            'student_vocab_size': self.student.vocab_size,
            'tokenizer_vocab_size': self.tokenizer.get_vocab_size()
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save conversation history if requested
        if self.config.save_conversations:
            conversation_path = Path(self.config.data_dir) / f"conversation_episode_{self.episode + 1}.json"
            self.teacher.save_conversation(str(conversation_path))
        
        # Save tokenizer vocabulary
        vocab_path = Path(self.config.data_dir) / "tokenizer_vocab.json"
        self.tokenizer.save_vocabulary(str(vocab_path))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.episode = checkpoint_data['episode']
        self.total_conversations = checkpoint_data['total_conversations']
        self.training_metrics = checkpoint_data.get('training_metrics', [])
        
        # Load teacher state
        self.teacher.current_stage = checkpoint_data.get('teacher_stage', 'baby_talk')
        self.teacher.turns_in_stage = checkpoint_data.get('teacher_turns', 0)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from episode {self.episode + 1}")
    
    def evaluate_student(self, test_prompts: List[str]) -> Dict:
        """Evaluate student performance on test prompts."""
        
        self.logger.info("Evaluating student performance")
        
        evaluation_results = {
            'test_prompts': test_prompts,
            'responses': [],
            'metrics': {}
        }
        
        for prompt in test_prompts:
            # Encode prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            
            # Generate response
            response_tokens = self.student.generate_sequence(
                prompt_tokens, max_length=30, temperature=0.8
            )
            
            # Decode response
            generated_tokens = response_tokens[len(prompt_tokens):]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            evaluation_results['responses'].append({
                'prompt': prompt,
                'response': response_text,
                'response_length': len(response_text.split())
            })
        
        # Compute metrics
        avg_response_length = np.mean([
            r['response_length'] for r in evaluation_results['responses']
        ])
        
        evaluation_results['metrics'] = {
            'avg_response_length': avg_response_length,
            'total_responses': len(evaluation_results['responses']),
            'current_stage': self.teacher.current_stage
        }
        
        return evaluation_results
    
    def interactive_session(self):
        """Run an interactive session where user can chat with the student."""
        
        print("Interactive session started. Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                # Process user input
                input_tokens = self.tokenizer.encode(user_input, add_special_tokens=True)
                
                # Generate student response
                response_tokens = self.student.generate_sequence(
                    input_tokens, max_length=50, temperature=1.0
                )
                
                # Decode and display
                generated_tokens = response_tokens[len(input_tokens):]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                print(f"Student: {response_text}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Interactive session ended.")