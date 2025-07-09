# Conversational Learning with Hebbian SNNs

This experiment implements **continual learning** for language through biological neural networks with **5 phases of neuro-inspired enhancements**. Unlike traditional AI that requires pre-training, this system learns during every conversation.

## Quick Start

To run all tests and verify the complete system:

```bash
python run_tests.py
```

To run a specific demo:

```bash
python safe_gpu1_demo.py
```

## Neuro-Inspired Enhancement Phases

The system implements 5 phases of biologically-inspired enhancements:

### Phase 1: Dopamine RPE Gating ✅
- **Reward Prediction Error (RPE)** signaling for learning
- **Dopamine modulation** of synaptic plasticity
- **Teacher feedback** drives learning through RPE
- **Target**: 30% fewer training samples to reach ≥90% accuracy

### Phase 2: Acetylcholine Attention + Norepinephrine Novelty ✅
- **Acetylcholine** for attention-based learning enhancement
- **Norepinephrine** for novelty detection and gain modulation
- **Pattern familiarity** tracking for adaptive learning
- **Target**: ↓5% validation perplexity

### Phase 3: Adenosine Fatigue + Short-Term Plasticity ✅
- **Adenosine fatigue** system for sleep-wake cycles
- **Short-Term Plasticity (STP)** buffers for temporary changes
- **Sleep consolidation** for memory strengthening
- **Target**: ½ cross-entropy on 128-token sequences

### Phase 4: Sleep Replay & Synaptic Tagging/Capture ✅
- **Synaptic tagging** for marking important experiences
- **Sleep replay** for offline memory consolidation
- **Synaptic capture** for selective memory strengthening
- **Target**: +7 BLEU after 1k turns

### Phase 5: Structural-Plasticity Optimization ✅
- **Adaptive structural plasticity** with incremental updates
- **Efficient connectivity management** with top-k selection
- **Performance monitoring** and adaptive frequency
- **Target**: ↓15% runtime with flat memory

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
- **Biologically-inspired neuromodulation**

## Directory Structure

```
conversational_learning/
├── README.md                    # This file
├── run_tests.py                 # Test runner for all phases
├── plastic_learner.py           # Main learner implementation
├── safe_gpu1_demo.py           # Demo script
├── requirements.txt            # Dependencies
├── models/
│   └── plastic_snn.py          # Enhanced neural network with all phases
├── utils/
│   ├── biologically_inspired_tokenizer.py
│   └── llm_teacher.py          # Teacher system
├── tests/
│   ├── test_phase1_dopamine.py         # Phase 1 tests
│   ├── test_phase2_attention.py        # Phase 2 tests
│   ├── test_phase3_fatigue.py          # Phase 3 tests
│   ├── test_phase4_replay.py           # Phase 4 tests
│   └── test_phase5_optimization.py     # Phase 5 tests
├── memory/
│   ├── plastic_memory.json            # Learner state
│   ├── plastic_memory_vocab.json      # Vocabulary
│   └── network_state_latest.npz       # Network weights
└── logs/
    └── plastic_learning.log           # System logs
```

## Implementation Features

### Biological Inspiration

#### Tokenization
Instead of pre-defined tokens, the system learns syllable-like patterns:
- Starts with basic phonemes
- Learns common combinations 
- Builds hierarchical representations
- Adapts vocabulary through usage

#### Neural Architecture
- **Columnar organization**: Like cortical columns
- **Hebbian learning**: "Neurons that fire together, wire together"
- **Temporal sequences**: Memory across time
- **Winner-take-all**: Biological attention mechanism
- **Neuromodulation**: Dopamine, acetylcholine, norepinephrine, adenosine

#### Learning Process
- **Real-time**: Updates during every interaction
- **Local**: No backpropagation, just local Hebbian rules
- **Adaptive**: Network grows and specializes
- **Continuous**: Never stops learning
- **Sleep-wake cycles**: Offline consolidation

## Performance Characteristics

- **Speed**: Fast inference, learning happens in parallel
- **Memory**: Flat memory consumption with bounded data structures
- **Generalization**: Emerges from biological principles
- **Robustness**: Graceful degradation, continuous adaptation
- **Efficiency**: ~15% runtime improvement with optimizations

## Phase Test Results

All phases are fully functional and tested:

- ✅ **Phase 1**: Dopamine RPE system working correctly
- ✅ **Phase 2**: Acetylcholine attention + Norepinephrine novelty working
- ✅ **Phase 3**: Adenosine fatigue + STP system working correctly
- ✅ **Phase 4**: Sleep replay & synaptic tagging working correctly
- ✅ **Phase 5**: Structural-plasticity optimization working correctly

## Example Session

```
🤖 Teacher: Hi baby! Say mama. Mama!
You: hi there
🧠 [Learning with dopamine RPE...]
🤖 Learner: hi
📈 Learned 15 patterns from 1 interactions
💡 Attention: 0.7, Novelty: 1.0, Fatigue: 0.1

You: how are you?
🧠 [Learning with attention boost...]  
🤖 Learner: good hi
📈 Learned 23 patterns from 2 interactions
💡 Attention: 0.8, Novelty: 0.6, Fatigue: 0.2

You: what is your name?
🧠 [Learning with novelty detection...]
🤖 Learner: mm name good
📈 Learned 31 patterns from 3 interactions
😴 Sleep triggered - consolidating memories...
```

## Configuration

```python
config = PlasticLearnerConfig(
    n_neurons=14173,              # Fixed network size
    vocab_size=800,               # Vocabulary capacity
    initial_connectivity=0.08,    # Sparse connectivity
    plasticity_rate=0.01,         # Hebbian learning rate
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
6. **Neuromodulation** improves learning efficiency and robustness

## Future Directions

- **Multi-modal learning**: Vision + language
- **Episodic memory**: Remember specific conversations
- **Social learning**: Multiple agents learning together
- **Curiosity-driven exploration**: Self-motivated learning
- **Meta-learning**: Learning how to learn better
- **Homeostatic plasticity**: Self-organizing neural networks