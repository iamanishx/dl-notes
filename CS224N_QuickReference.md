# CS224N Quick Reference Guide

## Word2Vec Algorithm Summary

### Skip-gram with Negative Sampling

```python
# Forward pass
center_vec = embeddings[center_word]
context_vec = embeddings[context_word]
score = dot(center_vec, context_vec)
prob = sigmoid(score)

# Loss calculation
positive_loss = -log(sigmoid(dot(center, positive_context)))
negative_loss = Σ_{neg} -log(sigmoid(-dot(center, negative_word)))
total_loss = positive_loss + negative_loss

# Gradient update
center_grad = (context_vec - prob * context_vec) + Σ_neg (prob_neg * negative_word_vec)
context_grad = (center_vec - prob * center_vec)
```

### Key Hyperparameters

- **Window Size**: 5-10 words (typical)
- **Vector Dimension**: 100-300 (sweet spot)
- **Negative Samples**: 5-15 per positive
- **Learning Rate**: 0.001-0.01 (adaptive best)

## GloVe Algorithm Summary

### Objective Function

```python
J = Σ_{i,j=1}^V f(X_{ij}) (u_i^T v_j + b_i + b_j - log X_{ij})^2

# Components
- X_{ij}: Co-occurrence count
- f(X_{ij}): Weighting function (min(max(X_{ij}, 100)^0.75 / X_{ij})
- u_i, v_j: Word vectors (two separate sets)
- b_i, b_j: Bias terms
```

### Training Process

1. Build co-occurrence matrix from corpus
2. Apply weighting function
3. Optimize with AdaGrad or similar
4. Average center and context vectors

## Neural Network Components

### Common Activation Functions

```python
# Sigmoid (binary classification)
σ(x) = 1 / (1 + exp(-x))

# ReLU (hidden layers)
ReLU(x) = max(0, x)

# Softmax (multi-class)
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

### Backpropagation Basics

```python
# Chain rule application
∂L/∂w = ∂L/∂a × ∂a/∂w

# For embedding layer
∂L/∂embedding = ∂L/∂output × output_weights^T
```

## Evaluation Metrics

### Intrinsic

```python
# Analogy accuracy
def analogy_accuracy(model, test_set):
    correct = 0
    for (a, b, c, d) in test_set:
        predicted = most_similar(a - b + c)
        if predicted == d:
            correct += 1
    return correct / len(test_set)

# Similarity correlation
def spearman_correlation(model_scores, human_scores):
    return scipy.stats.spearmanr(model_scores, human_scores)
```

### Extrinsic

```python
# Named Entity Recognition example
def evaluate_ner(embeddings, test_data):
    # Baseline: CRF with word features
    # Enhanced: CRF + embedding features
    baseline_f1 = train_and_evaluate(baseline_features)
    enhanced_f1 = train_and_evaluate(baseline_features + embeddings)
    return enhanced_f1 - baseline_f1
```

## Optimization Techniques

### Learning Rate Schedules

```python
# Step decay
lr_t = lr_0 / (1 + decay_rate * t)

# Exponential decay
lr_t = lr_0 * decay_rate^t

# Adaptive methods (Adam)
m_t = β1 * m_{t-1} + (1-β1) * grad_t
v_t = β2 * v_{t-1} + (1-β2) * grad_t^2
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

### Regularization Methods

```python
# L2 regularization
loss_with_reg = original_loss + λ * Σ_w ||w||^2

# Dropout
def dropout(x, p):
    mask = (random() > p)
    return x * mask / (1 - p)

# Early stopping
if validation_loss > best_loss - patience:
    break
```

## Practical Tips

### Memory Efficiency

- **Sparse Updates**: Only update words in current window
- **Batch Processing**: Balance speed vs. memory
- **Gradient Checkpointing**: Save intermediate states

### Numerical Stability

- **Gradient Clipping**: ||∇J|| ≤ max_norm
- **Mixed Precision**: FP16 for inference, FP32 for training
- **Loss Scaling**: Normalize across batch size

### Data Preprocessing

```python
# Subword tokenization (fastText)
def tokenize(text):
    return [char_ngram for char_ngram in text]

# Frequency filtering
vocab = {w for w in vocab if count[w] >= min_freq}

# Context window balancing
def sample_context(center_idx, window_size):
    # Dynamic window size based on word frequency
    return dynamic_window_sample(center_idx, window_size)
```

## Common Pitfalls & Solutions

### Out-of-Vocabulary Words

**Problem**: Unseen words at test time
**Solutions**:

- Subword models (fastText, BPE)
- Random initialization with training
- UNK token with learned representation

### Polysemy Handling

**Problem**: One vector per word type
**Solutions**:

- Contextual embeddings (BERT, ELMo)
- Sense clustering
- Multi-prototype models

### Training Instability

**Problem**: Loss explosion or divergence
**Solutions**:

- Learning rate reduction
- Gradient clipping
- Batch normalization
- Better initialization (Xavier/He)

---

#CS224N #Reference #DeepLearning #PracticalGuide
