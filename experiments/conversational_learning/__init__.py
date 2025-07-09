"""
Conversational Learning Experiment

Two approaches to language learning with Hebbian SNNs:

1. ContinualLearner: Real-time learning during every conversation (RECOMMENDED)
   - No pre-training phase
   - Network learns and grows through experience
   - Biologically-inspired vocabulary learning
   - True continual learning

2. ConversationalTrainer: Traditional training with teacher-student setup
   - Separate training and inference phases
   - More like conventional ML training

The ContinualLearner approach better matches the goal of learning during 
interaction rather than pre-training.
"""

from .continual_learner import ContinualLearner, LearnerConfig
from .training.trainer import ConversationalTrainer, TrainingConfig