# Conversational Learning with Hebbian SNNs

This experiment implements **continual learning** for language through biological neural networks. Unlike traditional AI that requires pre-training, this system learns during every conversation.

## Key Innovation: Learn-as-you-go

**Traditional Approach:**
- Pre-train on massive datasets
- Fixed knowledge after training
- Separate training/inference phases
- Can't adapt to new information

**Our Approach:**
- Start with minimal knowledge
- Learn during every conversation
- No separate training phase
- Continuously adapt and grow

## Two Implementations

### 1. ContinualLearner (Recommended) 🧠

Real-time learning during conversations:

```python
from experiments.conversational_learning import ContinualLearner

learner = ContinualLearner()
response = learner.process_input_and_respond("Hello!")  # Learning happens here!
```

**Features:**
- Starts nearly blank
- Vocabulary grows through exposure
- Network capacity expands as needed
- Real-time Hebbian weight updates
- Biologically-inspired tokenization

### 2. ConversationalTrainer (Traditional)

Teacher-student training setup:

```python
from experiments.conversational_learning import ConversationalTrainer

trainer = ConversationalTrainer(config)
trainer.train()  # Separate training phase
```

## Quick Start

### 1. Start your local LLM
```bash
# Make sure your local LLM is running on localhost:1234
# We tested with gemma-3-27b-it-qat
```

### 2. Chat with the learner
```bash
cd experiments/conversational_learning
python chat_with_learner.py
```

### 3. Watch it learn!
- The network starts knowing almost nothing
- Vocabulary grows as you talk
- Neural activity patterns emerge
- Responses improve through experience

## Biological Inspiration

### Tokenization
Instead of pre-defined tokens, the system learns syllable-like patterns:
- Starts with basic phonemes
- Learns common combinations 
- Builds hierarchical representations
- Adapts vocabulary through usage

### Neural Architecture
- **Columnar organization**: Like cortical columns
- **Hebbian learning**: "Neurons that fire together, wire together"
- **Temporal sequences**: Memory across time
- **Winner-take-all**: Biological attention mechanism

### Learning Process
- **Real-time**: Updates during every interaction
- **Local**: No backpropagation, just local Hebbian rules
- **Adaptive**: Network grows and specializes
- **Continuous**: Never stops learning

## Example Session

```
🤖 Teacher: Hi baby! Say mama. Mama!
You: hi there
🧠 [Learning...]
🤖 Learner: hi
📈 Learned 15 patterns from 1 interactions

You: how are you?
🧠 [Learning...]  
🤖 Learner: good hi
📈 Learned 23 patterns from 2 interactions

You: what is your name?
🧠 [Learning...]
🤖 Learner: mm name good
📈 Learned 31 patterns from 3 interactions
```

## Key Differences from Pre-training

| Aspect | Traditional | Our Approach |
|--------|-------------|--------------|
| **Learning** | Pre-training phase | During every conversation |
| **Knowledge** | Fixed after training | Continuously growing |
| **Vocabulary** | Pre-defined tokens | Learned through exposure |
| **Adaptation** | Requires retraining | Real-time adaptation |
| **Memory** | Static weights | Dynamic, experience-based |
| **Efficiency** | Needs massive data | Learns from small interactions |

## Performance Characteristics

- **Speed**: Fast inference, learning happens in parallel
- **Memory**: Grows with experience, not pre-allocated
- **Generalization**: Emerges from biological principles
- **Robustness**: Graceful degradation, continuous adaptation

## Files Structure

```
conversational_learning/
├── continual_learner.py          # Main continual learning system
├── chat_with_learner.py         # Interactive demo
├── utils/
│   ├── biologically_inspired_tokenizer.py  # Learned vocabulary
│   └── llm_teacher.py           # Teacher interface
├── models/
│   └── sequence_snn.py          # Columnar Hebbian network
└── training/
    └── trainer.py               # Traditional training approach
```

## Configuration

```python
config = LearnerConfig(
    initial_vocab_size=50,        # Start small
    max_vocab_size=2000,          # Can grow to this
    learning_rate=0.05,           # Higher for faster adaptation
    hebbian_strength=1.0,         # Biological learning strength
    teacher_api_url="http://localhost:1234/v1/chat/completions"
)
```

## Research Goals

This experiment explores whether:
1. **Biological learning** (Hebbian) can match gradient descent efficiency
2. **Continual learning** eliminates need for pre-training
3. **Real-time adaptation** enables truly interactive AI
4. **Emergent vocabulary** is more efficient than pre-defined tokens
5. **Neural growth** can handle increasing complexity

## Future Directions

- **Multi-modal learning**: Vision + language
- **Episodic memory**: Remember specific conversations
- **Social learning**: Multiple agents learning together
- **Curiosity-driven exploration**: Self-motivated learning
- **Meta-learning**: Learning how to learn better