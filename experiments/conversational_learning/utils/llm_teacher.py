"""
LLM Teacher Interface

Simplified teacher that acts as a parent helping a child learn to talk.
Uses a single adaptive system prompt and maintains conversation history.
"""

import requests
import json
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass


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
    Simplified LLM Teacher that acts like a nurturing parent.
    
    The teacher naturally adapts to the child's level and provides:
    - Encouragement and gentle correction
    - Age-appropriate responses
    - Natural conversation flow
    - Gradual complexity increase
    """
    
    def __init__(self, config: TeacherConfig = None):
        self.config = config or TeacherConfig()
        self.conversation_history = []
        self.session_id = int(time.time())
        self.max_history = 10  # Keep only recent conversation
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Simple system prompt that adapts naturally
        self.system_prompt = """You are a loving, patient parent helping your child learn to talk and communicate. Your child is just beginning to learn language, starting from basic sounds and gradually developing more complex speech.

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
    
    def _make_api_call(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """Make API call to the local LLM with retry logic."""
        
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
        
        return None
    
    def _build_messages(self, new_user_message: str = None) -> List[Dict]:
        """Build message list with system prompt and recent history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
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
        """Respond to student's attempt at communication."""
        
        # Build messages with conversation history
        messages = self._build_messages(student_attempt)
        
        response = self._make_api_call(messages)
        
        if response:
            # Store the exchange
            self.conversation_history.append({"role": "user", "content": student_attempt})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Trim history to keep it manageable
            self._trim_history()
            
            self.logger.info(f"Student: '{student_attempt}' -> Teacher: '{response}'")
            return response
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
        Simple stage indicator based on conversation length.
        This maintains compatibility with plastic_learner.py logging.
        """
        conversation_length = len(self.conversation_history)
        
        if conversation_length < 20:
            return "beginning"
        elif conversation_length < 100:
            return "developing"
        elif conversation_length < 300:
            return "expanding"
        else:
            return "conversational"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def reset_conversation(self):
        """Reset the conversation state."""
        self.conversation_history = []
        self.session_id = int(time.time())
        self.logger.info("Conversation reset")
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file."""
        conversation_data = {
            "session_id": self.session_id,
            "current_stage": self.current_stage,
            "history": self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        self.logger.info(f"Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from file."""
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            
            self.session_id = conversation_data.get("session_id", int(time.time()))
            self.conversation_history = conversation_data.get("history", [])
            
            # Trim loaded history to max size
            self._trim_history()
            
            self.logger.info(f"Conversation loaded from {filepath}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load conversation: {e}")
            self.reset_conversation()