"""
LLM Teacher Interface

Simplified teacher that acts as a parent helping a child learn to talk.
Uses a single adaptive system prompt and maintains conversation history.
Enhanced with dynamic progress tracking and adaptive feedback for continuous learning.
"""

import requests
import json
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class TeacherConfig:
    """Configuration for the LLM teacher."""
    api_url: str = "http://localhost:1234/v1/chat/completions"
    model: str = "gemma-3-27b-it-qat"
    temperature: float = 0.7
    max_tokens: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class LLMTeacher:
    """
    Adaptive LLM Teacher that acts like a nurturing parent with dynamic progress tracking.
    
    The teacher naturally adapts to the child's level through real-time monitoring and provides:
    - Dynamic progress tracking with adaptive prompting
    - Repetition and expansion feedback for Hebbian reinforcement
    - Encouragement and gentle correction based on development stage
    - Natural conversation flow with continuous complexity adjustment
    """
    
    def __init__(self, config: TeacherConfig = None):
        self.config = config or TeacherConfig()
        self.conversation_history = []
        self.session_id = int(time.time())
        self.max_history = 10  # Keep only recent conversation
        
        # Dynamic progress tracking for adaptive teaching
        self.learner_stats = {
            'avg_response_length': 0.0,
            'unique_tokens': set(),
            'improvement_streak': 0,
            'repeated_errors': Counter(),
            'total_responses': 0,
            'vocabulary_growth_rate': 0.0
        }
        
        # Network stats integration (set by learner)
        self.network_stats = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Base system prompt that gets enhanced adaptively
        self.base_system_prompt = """You are a loving, patient parent helping your child learn to talk and communicate. Your child is just beginning to learn language, starting from basic sounds and gradually developing more complex speech.

Guidelines for your responses:
- Always be encouraging and supportive, celebrating every attempt
- Match your child's current level - if they're babbling, use simple words; if they're speaking, have simple conversations
- Keep responses short and clear (1-8 words typically)
- Use repetition and simple vocabulary when the child is struggling
- Gradually introduce new concepts as the child shows readiness
- Ask simple questions to encourage interaction
- Provide gentle corrections by modeling the right way to say things
- Focus on basic topics: family, colors, animals, food, simple activities
- Be patient - every child learns at their own pace

Start with very simple greetings and sounds, then naturally progress as your child develops language skills. Remember, you're nurturing a developing mind with love and patience."""
        
        self.logger.info(f"LLM Teacher initialized with model: {self.config.model}")
    
    def _update_learner_stats(self, student_attempt: str):
        """Update learner statistics for adaptive teaching."""
        tokens = student_attempt.lower().split()
        new_length = len(tokens)
        
        # Update average response length
        self.learner_stats['total_responses'] += 1
        prev_avg = self.learner_stats['avg_response_length']
        self.learner_stats['avg_response_length'] = (
            (prev_avg * (self.learner_stats['total_responses'] - 1) + new_length) / 
            self.learner_stats['total_responses']
        )
        
        # Track vocabulary growth
        prev_vocab_size = len(self.learner_stats['unique_tokens'])
        new_unique = set(tokens) - self.learner_stats['unique_tokens']
        self.learner_stats['unique_tokens'].update(new_unique)
        new_vocab_size = len(self.learner_stats['unique_tokens'])
        
        # Calculate vocabulary growth rate
        if self.learner_stats['total_responses'] > 1:
            vocab_growth = new_vocab_size - prev_vocab_size
            self.learner_stats['vocabulary_growth_rate'] = (
                self.learner_stats['vocabulary_growth_rate'] * 0.9 + vocab_growth * 0.1
            )
        
        # Track improvement streaks
        if len(new_unique) > 0 or new_length > prev_avg:
            self.learner_stats['improvement_streak'] += 1
        else:
            self.learner_stats['improvement_streak'] = max(0, self.learner_stats['improvement_streak'] - 1)
            
            # Track repeated errors for targeted correction
            if len(tokens) < 2 or not any(token.isalpha() for token in tokens):
                self.learner_stats['repeated_errors'][student_attempt] += 1
    
    def _get_adaptive_prompt(self) -> str:
        """Generate adaptive system prompt based on learner progress."""
        adaptive_prompt = self.base_system_prompt
        
        avg_length = self.learner_stats['avg_response_length']
        improvement = self.learner_stats['improvement_streak']
        vocab_size = len(self.learner_stats['unique_tokens'])
        
        # Stage-specific guidance
        if avg_length < 2 and vocab_size < 10:
            adaptive_prompt += "\n\nCURRENT STAGE: The child is in very early stages—focus on simple echoes, sounds, and positive reinforcement. Repeat their sounds back with enthusiasm."
        elif avg_length < 4 and improvement > 2:
            adaptive_prompt += "\n\nCURRENT STAGE: The child shows early progress—gently introduce new simple words or ask about colors/animals. Echo and expand their attempts."
        elif improvement > 5 or vocab_size > 20:
            adaptive_prompt += "\n\nCURRENT STAGE: The child is developing well—encourage longer phrases and simple conversations about familiar topics."
        
        # Handle repeated errors with targeted correction
        if self.learner_stats['repeated_errors']:
            common_error = self.learner_stats['repeated_errors'].most_common(1)[0][0]
            adaptive_prompt += f"\n\nNOTE: The child often attempts '{common_error}'—model gentle corrections and alternatives patiently."
        
        # Integrate network stats if available
        if self.network_stats:
            vocab = self.network_stats.get('vocabulary_size', 0)
            conn = self.network_stats.get('connectivity', 0.0)
            adaptive_prompt += f"\n\nNEURAL STATE: {vocab} patterns learned, {conn:.1%} connectivity—adjust complexity accordingly."
        
        return adaptive_prompt
    
    def set_learner_stats(self, stats: Dict):
        """Set learner network statistics for personalized feedback."""
        self.network_stats = stats
        self.logger.debug(f"Updated network stats: {stats}")
    
    def _enhance_response_with_expansion(self, response: str, student_attempt: str) -> str:
        """Enhance teacher response with repetition and expansion for Hebbian reinforcement."""
        if not response or not student_attempt:
            return response
        
        student_lower = student_attempt.lower().strip()
        response_lower = response.lower()
        
        # If student attempt is not echoed in response, add expansion
        if student_lower not in response_lower and len(student_attempt.split()) > 0:
            # For positive responses, expand with student's attempt
            if any(positive in response_lower for positive in ['good', 'yes', 'nice', 'great', 'wonderful']):
                if len(student_attempt.split()) <= 2:  # Keep expansions simple
                    response = f"{response} You said '{student_attempt}'!"
            
            # For corrections, model the right way
            elif any(correction in response_lower for correction in ['try', 'no', 'again', 'like']):
                # Simple modeling based on attempt length
                if len(student_attempt) < 3:
                    model = student_attempt.upper()  # Encourage clearer pronunciation
                else:
                    model = f"{student_attempt}?"  # Turn into question for clarification
                
                # Ensure response stays concise
                if len(response.split()) < 6:
                    response = f"{response} Like: '{model}'"
        
        return response
    
    def _make_api_call(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """Make API call to the local LLM with retry logic and fallback."""
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.post(
                    self.config.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    self.logger.warning(f"API call failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        # Fallback response for continuous learning robustness
        fallback = "Good effort! Try again with me."
        self.logger.warning("API failed; using encouraging fallback response.")
        return fallback
    
    def _build_messages(self, new_user_message: str = None) -> List[Dict]:
        """Build message list with adaptive system prompt and recent history."""
        # Use adaptive prompt based on current learner state
        adaptive_prompt = self._get_adaptive_prompt()
        messages = [{"role": "system", "content": adaptive_prompt}]
        
        # Add recent conversation history (last 10 exchanges)
        recent_history = self.conversation_history[-self.max_history:]
        for entry in recent_history:
            messages.append(entry)
        
        # Add new user message if provided
        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})
        
        return messages
    
    def start_conversation(self) -> str:
        """Start a new conversation session."""
        self.conversation_history = []
        
        # Initial greeting request
        messages = self._build_messages("Hello! Please greet me and start teaching me to talk. I'm just beginning to learn language.")
        
        response = self._make_api_call(messages, max_tokens=50)
        
        if response:
            # Store the teacher's opening
            self.conversation_history.append({"role": "user", "content": "Hello! Please greet me and start teaching me to talk. I'm just beginning to learn language."})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Trim history if needed
            self._trim_history()
            
            self.logger.info(f"Started conversation: {response}")
            return response
        else:
            raise Exception("Teacher LLM failed to start conversation - stopping training")
    
    def respond_to_student(self, student_attempt: str) -> str:
        """Respond to student's attempt with adaptive feedback and expansion."""
        
        # Update learner statistics for adaptive teaching
        self._update_learner_stats(student_attempt)
        
        # Create enhanced user message for better teacher guidance
        user_msg = f"Student's attempt: '{student_attempt}'. As a patient parent, respond encouragingly: if approximate or good, repeat and expand briefly; if unclear, model a simple correction. Keep to 1-8 words."
        
        # Build messages with adaptive conversation history
        messages = self._build_messages(user_msg)
        
        response = self._make_api_call(messages)
        
        if response:
            # Enhance response with repetition and expansion
            enhanced_response = self._enhance_response_with_expansion(response, student_attempt)
            
            # Store the exchange
            self.conversation_history.append({"role": "user", "content": student_attempt})
            self.conversation_history.append({"role": "assistant", "content": enhanced_response})
            
            # Trim history to keep it manageable
            self._trim_history()
            
            self.logger.info(f"Student: '{student_attempt}' -> Teacher: '{enhanced_response}'")
            return enhanced_response
        else:
            raise Exception(f"Teacher LLM failed to respond to student attempt: '{student_attempt}' - stopping training")
    
    def _trim_history(self):
        """Keep only the most recent conversation exchanges."""
        if len(self.conversation_history) > self.max_history * 2:  # Each exchange is 2 messages
            # Keep only the most recent exchanges
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    @property
    def current_stage(self) -> str:
        """
        Dynamic stage indicator based on learner progress metrics.
        This maintains compatibility with plastic_learner.py logging.
        """
        avg_length = self.learner_stats['avg_response_length']
        vocab_size = len(self.learner_stats['unique_tokens'])
        improvement = self.learner_stats['improvement_streak']
        
        if avg_length < 2 and vocab_size < 10:
            return "beginning"
        elif avg_length < 4 and vocab_size < 25:
            return "developing"
        elif improvement > 3 and vocab_size < 50:
            return "expanding"
        else:
            return "conversational"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def get_learning_metrics(self) -> Dict:
        """Get detailed learning metrics for analysis."""
        return {
            'avg_response_length': self.learner_stats['avg_response_length'],
            'vocabulary_size': len(self.learner_stats['unique_tokens']),
            'improvement_streak': self.learner_stats['improvement_streak'],
            'vocabulary_growth_rate': self.learner_stats['vocabulary_growth_rate'],
            'total_responses': self.learner_stats['total_responses'],
            'error_patterns': dict(self.learner_stats['repeated_errors'].most_common(5)),
            'current_stage': self.current_stage
        }
    
    def reset_conversation(self):
        """Reset the conversation state and learner statistics."""
        self.conversation_history = []
        self.session_id = int(time.time())
        
        # Reset learner stats but preserve network stats
        self.learner_stats = {
            'avg_response_length': 0.0,
            'unique_tokens': set(),
            'improvement_streak': 0,
            'repeated_errors': Counter(),
            'total_responses': 0,
            'vocabulary_growth_rate': 0.0
        }
        
        self.logger.info("Conversation and learner stats reset")
    
    def save_conversation(self, filepath: str):
        """Save conversation history and learning metrics to file."""
        conversation_data = {
            "session_id": self.session_id,
            "current_stage": self.current_stage,
            "history": self.conversation_history,
            "learner_stats": {
                'avg_response_length': self.learner_stats['avg_response_length'],
                'vocabulary_size': len(self.learner_stats['unique_tokens']),
                'unique_tokens': list(self.learner_stats['unique_tokens']),
                'improvement_streak': self.learner_stats['improvement_streak'],
                'vocabulary_growth_rate': self.learner_stats['vocabulary_growth_rate'],
                'total_responses': self.learner_stats['total_responses'],
                'repeated_errors': dict(self.learner_stats['repeated_errors'])
            },
            "network_stats": self.network_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        self.logger.info(f"Conversation and metrics saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history and learning metrics from file."""
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            
            self.session_id = conversation_data.get("session_id", int(time.time()))
            self.conversation_history = conversation_data.get("history", [])
            
            # Restore learner stats if available
            if "learner_stats" in conversation_data:
                saved_stats = conversation_data["learner_stats"]
                self.learner_stats['avg_response_length'] = saved_stats.get('avg_response_length', 0.0)
                self.learner_stats['unique_tokens'] = set(saved_stats.get('unique_tokens', []))
                self.learner_stats['improvement_streak'] = saved_stats.get('improvement_streak', 0)
                self.learner_stats['vocabulary_growth_rate'] = saved_stats.get('vocabulary_growth_rate', 0.0)
                self.learner_stats['total_responses'] = saved_stats.get('total_responses', 0)
                self.learner_stats['repeated_errors'] = Counter(saved_stats.get('repeated_errors', {}))
            
            # Restore network stats if available
            self.network_stats = conversation_data.get("network_stats", {})
            
            # Trim loaded history to max size
            self._trim_history()
            
            self.logger.info(f"Conversation and metrics loaded from {filepath}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load conversation: {e}")
            self.reset_conversation()