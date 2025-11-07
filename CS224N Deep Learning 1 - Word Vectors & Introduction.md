# CS224N Deep Learning 1: Word Vectors & Introduction

## Course Overview

**Instructor**: Christopher Manning (Stanford)
**Focus**: Natural Language Processing with Deep Learning

### Three Primary Goals

1. **Foundations of Deep Learning for NLP**: Basics → Key methods (RNNs, attention, transformers)
2. **Big Picture Understanding of Human Language**: Why language is difficult for computers to understand/produce
3. **Practical Implementation**: Build systems in PyTorch for major NLP problems (word meaning, dependency parsing, machine translation, QA)

## Human Language & Word Meaning

### The Power of Language

- **Recent Evolution**: Language developed ~100K-1M years ago (blink of an eye in evolutionary terms)
- **Writing**: Only ~5,000 years old - enabled knowledge preservation across time/space
- **Key Insight**: Communication, not physical attributes, became humanity's competitive advantage

### Language as Social System

> "Language isn't a formal system, language is glorious chaos. You can never know for sure what any words will mean to anyone. All you can do is try to get better at guessing how your words affect people."

- **Challenge**: Language is constructed, interpreted, and constantly evolving
- **NLP Goal**: Build computational systems that better guess word meanings and effects

## Traditional NLP Limitations

### Discrete Symbol Problem

- **One-hot Vectors**: Each word = separate dimension (500K+ dimensions for vocabulary)
- **Orthogonality Issue**: "motel" ⊥ "hotel" - no natural similarity measure
- **WordNet Limitations**:
  - Lacks nuance (proficient ≠ good in all contexts)
  - Incomplete/Outdated (missing modern terms like "ninja" as programmer)
  - No similarity measure (fantastic ≈ great but not synonyms)

## Distributional Semantics Revolution

### Core Principle

> "You shall know a word by the company it keeps" - J.R. Firth

**Key Insight**: Word meaning = context words that frequently appear nearby

### Word2Vec Algorithm

#### Model Architecture

- **Two Vectors per Word**:
  - `v_w`: Center word vector
  - `u_w`: Context word vector
  - _Rationale_: Simplifies math/optimization (avoids w·w terms)

#### Probability Calculation

```
P(o|c) = exp(u_o · v_c) / Σ_w exp(u_w · v_c)
```

- **Dot Product**: Natural similarity measure
- **Softmax**: Converts scores to probability distribution

#### Objective Function

```
J(θ) = -(1/T) Σ_t Σ_{-m≤j≤m, j≠0} log P(w_{t+j} | w_t)
```

#### Gradient Descent

- **Stochastic Gradient Descent (SGD)**: Update per center word, not entire corpus
- **Learning Rate (α)**: Critical hyperparameter
  - Too small: Slow convergence
  - Too large: Divergence or oscillation

## Mathematical Deep Dive

### Gradient Calculation for Center Vectors

∂/∂v_c log P(o|c) = u_o - Σ_x P(x|c) · u_x

**Interpretation**:

- **Observed** (u_o): Push toward actual context words
- **Expected** (Σ_x P(x|c) · u_x): Pull away from weighted average of all words

### Optimization Insights

- **Sparse Updates**: Each window affects ~11 words only
- **Row vs Column Vectors**: PyTorch uses row vectors for memory efficiency
- **Non-convex Landscape**: Complex networks have multiple local minima

## Word Vector Properties

### Amazing Capabilities

1. **Semantic Clustering**: Similar words group together
2. **Analogical Reasoning**: vector(king) - vector(man) + vector(woman) ≈ vector(queen)
3. **Cross-lingual Patterns**: country-capital relationships
4. **Syntactic Regularities**: tall-tallest, long-longest

### Dimensionality

- **Sweet Spot**: 300 dimensions (empirically determined)
- **Trade-offs**:
  - <25: Poor performance
  - > 300: Diminishing returns
  - 1000: Maximum useful range

## Historical Context

### Pre-Deep Learning Era

- **Co-occurrence Matrices**: Huge, sparse, high-dimensional
- **SVD**: Linear algebra approach, poor with raw counts
- **LSA/COALS**: Early attempts at count-based methods

### Word2Vec Breakthrough (2013)

- **Efficiency**: Iterative learning vs. matrix operations
- **Performance**: Superior to count-based methods
- **Scalability**: Works with billions of words

## Practical Applications

### Demo Results

```python
# Similarity examples
most_similar('croissant') → ['brioche', 'baguette', 'focaccia']
most_similar('usa') → ['canada', 'america', 'united_states']

# Analogies
analogy('man', 'king', 'woman') → 'queen'
analogy('australia', 'canberra', 'france') → 'paris'
```

## Key Takeaways

1. **Distributed Representation**: Meaning spread across dimensions, not localized
2. **Context is Meaning**: Word vectors learn from surrounding words
3. **Simple → Complex**: Basic algorithm captures sophisticated relationships
4. **Foundation**: Word vectors enable all modern NLP systems

---

#LectureNotes #CS224N #WordVectors #DeepLearning #NLP
