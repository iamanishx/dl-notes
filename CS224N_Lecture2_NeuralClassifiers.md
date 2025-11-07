# CS224N Lecture 2: Neural Classifiers & Advanced Word Vectors

## Lecture Overview
**Continuation**: Word vectors, word senses, neural network classifiers
**Goal**: Understand word embedding papers (Word2Vec, GloVe) and neural foundations

## Word2Vec Algorithm Family

### Two Main Variants
1. **Skip-gram**: Predict context words from center word
   - More natural for language modeling
   - Better performance in practice
   
2. **Continuous Bag of Words (CBOW)**: Predict center word from context bag
   - Faster training
   - Similar results

### Negative Sampling Optimization

#### Problem with Naive Softmax
- **Computational Cost**: Σ_w exp(u_w · v_c) requires full vocabulary iteration
- **Example**: 100K vocab = 100K dot products per update

#### Solution: Binary Classification
```
# Instead of: P(context|center) via softmax
# Use: Binary logistic regression

Positive: (center, actual_context) → sigmoid(u_o · v_c) ≈ 1
Negative: (center, random_word) → sigmoid(u_r · v_c) ≈ 0
```

#### Loss Function
```
J = -log σ(u_o · v_c) - Σ_{k=1}^K log σ(-u_k · v_c)
```

#### Sampling Strategy
- **Unigram Distribution**: P(w) ∝ count(w)^(3/4)
- **Rationale**: Dampen frequency differences, sample rare words more often

## Co-occurrence Matrix Methods

### Traditional Approach
```python
# Example corpus: "I like deep learning. I enjoy flying."
# Window size = 1

        I  like  deep  learning  enjoy  flying
I       0    2     0        0       0
like    2    0     1        0       0
deep    0    1     0        0       0
learning 0    0     1        1       0
enjoy   0    0     0        1       0
flying  0    0     0        1       0
```

### Problems with Count Methods
1. **Dimensionality**: |V| × |V| matrix (500K × 500K)
2. **Sparsity**: Most entries = 0
3. **Noise**: Random corpus variations affect results
4. **Frequency Bias**: Common words dominate

### SVD Limitations
- **Assumption**: Normally distributed errors
- **Reality**: Word counts follow Zipf's distribution
- **Solution**: Scale counts (log, cap, remove function words)

## GloVe: Global Vectors

### Key Innovation
**Unify**: Matrix methods (efficient statistics) + Neural methods (better performance)

### Mathematical Foundation
**Core Insight**: Meaning components = ratios of co-occurrence probabilities

```
# Example: solid-gas spectrum
P(solid|ice) / P(solid|steam) ≈ 10
P(gas|ice) / P(gas|steam) ≈ 0.1
P(water|ice) / P(water|steam) ≈ 1
```

### Objective Function
```
J = Σ_{i,j} f(X_{ij}) (u_i^T v_j - log X_{ij})^2
```

- **f(X_{ij})**: Weighting function (diminishes extreme counts)
- **Log-bilinear**: u_i^T v_j approximates log co-occurrence
- **Bias terms**: Handle word frequency differences

### Performance Results
| Model              | Semantic | Syntactic | Total |
|--------------------|----------|------------|-------|
| SVD (raw)         | 7.3%     | -          | 7.3%  |
| SVD (scaled)       | 60.1%    | -          | 60.1%  |
| Word2Vec (skip-gram)| 68.4%    | 61.4%      | 64.9%  |
| GloVe              | 71.3%    | 68.1%      | 69.7%  |

## Word Vector Evaluation

### Intrinsic Evaluation
**Direct measurement** of vector quality

#### 1. Analogy Tasks
```python
# Semantic: man → king as woman → ____
analogy('man', 'king', 'woman') → 'queen'

# Syntactic: tall → tallest as long → ____
analogy('tall', 'tallest', 'long') → 'longest'
```

**Exclusion Rule**: Don't allow input words in answer set

#### 2. Similarity Correlation
- **Human Judgments**: Paired word similarity scores (0-10 scale)
- **Model Correlation**: Spearman's ρ between rankings
- **Examples**: 
  - tiger-cat: high similarity
  - computer-internet: medium similarity  
  - stock-jaguar: low similarity

### Extrinsic Evaluation
**Real task performance** with word vectors as features

#### Example: Named Entity Recognition
```python
# Baseline: Discrete word features
# Enhanced: Word vectors + baseline
# Result: Significant F1 score improvement
```

**Advantages**:
- Direct task relevance
- Real-world performance measure

**Disadvantages**:
- Expensive to evaluate
- Confounding factors in full system

## Word Senses Challenge

### Polysemy Problem
**Example**: "pike" has multiple meanings:
1. **Weapon**: Sharp pointed staff
2. **Fish**: Elongated aquatic animal  
3. **Road**: Turnpike/highway

### Single Vector Limitation
- **Averaging Effect**: Multiple meanings blur together
- **Context Loss**: Can't distinguish "bank" (river vs. financial)

### Potential Solutions
1. **Contextual Embeddings**: BERT, ELMo
2. **Sense Clusters**: Multiple vectors per word
3. **Dynamic Computation**: Context-dependent representations

## Neural Network Foundations

### Classification Review
**Traditional**: Feature engineering + linear models
**Neural**: Learn features automatically

### Key Differences
1. **Representation Learning**: Hierarchical feature discovery
2. **Non-linearity**: Activation functions
3. **End-to-End**: Direct input→output mapping

### Neural Network Components
- **Input Layer**: Word vectors or embeddings
- **Hidden Layers**: Feature transformations
- **Output Layer**: Task-specific predictions
- **Loss Function**: Optimization target
- **Backpropagation**: Gradient-based learning

## Practical Implementation

### Vector Dimensions
- **25D**: Poor performance
- **50D**: Reasonable baseline
- **100D**: Good performance
- **300D**: Sweet spot (diminishing returns beyond)
- **1000D**: Maximum practical range

### Training Data Effects
| Data Source        | Semantic Score | Notes |
|-------------------|---------------|---------|
| Google News (1B)   | 64.9%         | News-specific |
| Wikipedia (6B)     | 76.2%         | Encyclopedia knowledge |
| Common Crawl (42B) | 81.5%         | Broad web knowledge |

### Optimization Insights
- **SGD Benefits**: Faster + better generalization
- **Batch Size**: Trade-off speed vs. stability
- **Learning Rate**: Critical hyperparameter
- **Regularization**: Prevent overfitting

## Key Takeaways

1. **Algorithm Trade-offs**: Speed vs. performance, simplicity vs. accuracy
2. **Evaluation Matters**: Both intrinsic and extrinsic metrics needed
3. **Context is King**: Word meaning depends on usage patterns
4. **Neural Advantage**: Automatic feature learning from raw data
5. **Scaling Laws**: More data → better vectors (with diminishing returns)

---

#LectureNotes #CS224N #NeuralNetworks #WordEmbeddings #NLP