## Biologically-Inspired Hebbian LLM Technical Spec

### High-Level Architecture:
- Fully recurrent, associative spiking neural network (SNN)
- Pure Hebbian learning via spike-timing dependent plasticity (STDP)
- Continuous inference-time learning, no training/inference separation

### Neuron Types:
- **Sensory Neurons (Input):**
  - Sparse encoding of linguistic tokens (subword/word-level)
  - Spike upon token recognition
- **Output Neurons:**
  - Map activity to linguistic outputs (words/tokens)
  - Neurons correspond explicitly to output vocabulary tokens
- **Associative Neurons:**
  - Recurrent internal neurons
  - Plastic connectivity to sensory, output, and each other
- **Inhibitory Interneurons:**
  - Stabilize network, enforce sparse coding
  - Regulate activity across network

### Learning Mechanisms:
- **Hebbian Learning via STDP:**
  - Strengthen connections if postsynaptic neuron spikes shortly after presynaptic neuron (Δt ≈ 20ms)
  - Weaken connections for out-of-sync spikes

- **Synaptic Homeostasis:**
  - Neurons dynamically adjust firing thresholds to maintain average firing rate (target ≈ 5-15%)
  - Prevents runaway excitation and promotes sparsity

- **Neuromodulation:**
  - Global modulatory signals triggered by novelty or surprise detection
  - Temporarily increase synaptic plasticity, accelerating learning in novel contexts

### Initial Network Configuration:
- **Innate Connectivity:**
  - Small number (~1%) of neurons have weak initial bias towards linguistic primitives (common words/subwords)
  - Primarily random initial sparse connectivity (~5-10% neurons interconnected)

- **Developmental Phase:**
  - Initial "infancy" period: exposure to simple, short linguistic inputs (e.g., short sentences, high-frequency tokens)
  - Neuromodulation aggressively tuned during this phase for rapid structural emergence

### Continuous Learning and Memory Consolidation:
- **Online Learning:**
  - Every inference event (input exposure) updates connections via STDP

- **Sleep Consolidation:**
  - Periodic offline phases (scheduled or after activity threshold reached)
  - Replays recent patterns at reduced plasticity
  - Prunes weak or rarely-used synapses
  - Reinforces strong associations to form stable attractors (abstract representations)

### Sparse Connectivity and Modularity:
- **Sparsity:**
  - Average neuron connectivity ≤ 10% of network neurons
  - Dynamically enforced through pruning during consolidation phases

- **Modularity:**
  - Emergent modules specialize by linguistic function (syntax, semantics, temporal context)
  - Primarily local connections within modules, fewer global connections ("bridge neurons") to ensure global coherence

### Handling Linguistic Complexity:
- **Hierarchical Representation:**
  - Neurons operate at varying timescales (fast neurons encode immediate linguistic units, slow neurons aggregate longer sequences)

- **Long-Range Dependencies:**
  - Reinforced via recurrent connectivity and varied delays

- **Abstract Concepts:**
  - Formed via attractor dynamics, stabilized during sleep consolidation phases

### Input/Output Interface:
- **Input:**
  - Text tokenized into linguistic features activating sensory neurons
- **Output:**
  - Highest activated output neuron (post-inference stabilization) selected as predicted token/word

### Evaluation Metrics:
- **Behavioral:** Next-word prediction, coherent sentence generation, basic Q&A tasks
- **Internal Representations:**
  - Analysis of neuronal activation clusters
  - Validate formation of meaningful linguistic categories
- **Continual Learning:**
  - Test adaptation to new vocabulary and grammar over time
  - Monitor stability of previously learned associations

### Computational Implementation:
- **Simulation:**
  - Event-driven spiking neural network (SNN) simulation
  - Time resolution: ~1 ms per tick
- **Hardware:**
  - GPU-based parallelization or neuromorphic chips recommended (e.g., Loihi, SpiNNaker)

### Prototype Scale Recommendation:
- **Neuron Count:**
  - Initial small-scale proof-of-concept: ~10,000-50,000 neurons
  - Sparse connections (~500-5,000 connections/neuron)

### Deliverables:
- **Implementation:** Minimal SNN simulator capable of STDP, neuromodulation, sensory/output neuron management, and sleep consolidation
- **Evaluation Suite:** Framework to assess linguistic coherence, continuous learning performance, and neuronal activity analysis

### Integrated Training Flow

The training process is organized into five key phases that work together to create a synergistic learning experience for the Hebbian LLM. Each phase leverages one or more of the components you specified, ensuring they contribute to the model’s development.

#### 1. Input Phase: Learning from Raw Source Material
- **What Happens**: The Hebbian LLM is exposed to raw source material, such as text corpora or datasets of natural language. This could include books, articles, or conversational data.
- **How It Works**: As the raw data is processed, sensory neurons in the Hebbian LLM are activated, producing spikes that travel through the network. Based on the timing of these spikes, Spike-Timing-Dependent Plasticity (STDP)—a core Hebbian learning rule—adjusts the synaptic weights between neurons.
- **Purpose**: This phase allows the Hebbian LLM to learn basic linguistic patterns and structures directly from the raw data, mimicking an "infancy" stage where foundational knowledge is built through exposure.
- **Component Used**: Raw source material.

#### 2. Reference Phase: Guidance from the LLM
- **What Happens**: A pre-trained LLM (e.g., GPT-3 or a similar model) processes the same input data as the Hebbian LLM and generates a "gold standard" response, such as a predicted next word, a completed sentence, or an answer to a question.
- **How It Works**: This reference LLM acts as a teacher, providing high-quality outputs that the Hebbian LLM can use as a target to aim for.
- **Purpose**: By offering accurate and contextually appropriate responses, the reference LLM provides a benchmark that guides the Hebbian LLM’s learning, ensuring it aligns with established linguistic norms.
- **Component Used**: LLM for reference and question answering.

#### 3. Comparison and Adjustment: Bridging the Gap
- **What Happens**: The Hebbian LLM’s output (e.g., a generated response or prediction) is compared to the reference LLM’s output.
- **How It Works**: Any differences between the two outputs trigger neuromodulatory signals within the Hebbian LLM. These signals temporarily increase plasticity in the parts of the network responsible for the error, encouraging faster adaptation of synaptic weights to reduce the discrepancy.
- **Purpose**: This step integrates supervised-like guidance into the Hebbian framework, helping the continuously learning AI refine its understanding and improve its performance based on the reference LLM’s expertise.
- **Components Used**: Continuously learning AI (Hebbian LLM) and LLM for reference.

#### 4. Consolidation Phase: Stabilizing Knowledge
- **What Happens**: Periodically, the Hebbian LLM enters a "sleep consolidation" phase, inspired by biological processes.
- **How It Works**: During this phase, the model replays recent activity patterns, strengthening important synaptic connections and pruning weaker ones. This process stabilizes learned linguistic patterns and prevents overfitting to recent inputs.
- **Purpose**: Consolidation ensures that the knowledge gained from raw source material and reference LLM guidance is retained and abstracted into robust representations, supporting long-term learning.
- **Component Used**: Continuously learning AI (Hebbian LLM).

#### 5. Evaluation and Feedback Loop: Continuous Improvement
- **What Happens**: The Hebbian LLM’s performance is regularly assessed using tasks like question answering, next-word prediction, or sentence generation, with the reference LLM providing the benchmark for correctness.
- **How It Works**: Performance metrics (e.g., accuracy, coherence) are calculated. If the Hebbian LLM struggles with certain areas (e.g., complex grammar), the training process is adjusted—perhaps by providing more targeted raw source material or increasing reliance on the reference LLM for those specific cases.
- **Purpose**: This feedback loop ensures the training remains effective and adaptive, allowing the Hebbian LLM to progressively improve and eventually rely less on the reference LLM as it matures.
- **Components Used**: Continuously learning AI (Hebbian LLM) and LLM for reference.

---

### How It All Fits Together
- **Raw Source Material** provides the foundational data for unsupervised learning, letting the Hebbian LLM discover patterns naturally through STDP.
- **The Reference LLM** acts as a teacher, offering high-quality examples and corrections to guide the Hebbian LLM toward better performance.
- **The Continuously Learning Hebbian LLM** ties everything together, processing inputs, adjusting based on feedback, and consolidating knowledge over time.

This flow operates as a cycle: the Hebbian LLM processes raw data, refines its outputs with help from the reference LLM, consolidates what it’s learned, and uses feedback to improve continuously. Over time, as its linguistic abilities strengthen, it can depend less on the reference LLM, becoming a standalone, adaptive AI.

