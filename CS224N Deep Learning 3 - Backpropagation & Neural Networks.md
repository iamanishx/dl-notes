# CS224N Deep Learning 3: Backpropagation & Neural Networks

## Lecture Overview

**Focus**: Mathematical foundations of neural network training
**Key Topics**: Matrix calculus, Jacobians, backpropagation algorithm, computational graphs

## Why Python for Deep Learning?

- **Ecosystem**: Rich scientific computing libraries (NumPy, SciPy, PyTorch)
- **Rapid Prototyping**: Quick iteration and experimentation
- **Industry Standard**: Most research and production systems use Python
- **Educational Value**: Lower barrier to entry for deep learning

## Course Setup & Python Basics

### Development Environment

```python
# Essential libraries for NLP/DL
import numpy as np           # Numerical computing
import torch                    # Deep learning framework
import matplotlib.pyplot as plt # Visualization

# For advanced operations
from scipy import stats
from sklearn.metrics import accuracy_score
```

### NumPy Fundamentals

```python
# Vector operations
vectors = np.random.randn(1000, 300)  # 1000 words, 300D
similarities = vectors @ vectors.T        # Cosine similarity

# Matrix operations
weights = np.random.randn(300, 100)     # 300D → 100D
biases = np.random.randn(100)           # Bias terms
activations = vectors @ weights + biases   # Linear transformation
```

### PyTorch Introduction

```python
# Tensor creation
word_vectors = torch.randn(50000, 300, requires_grad=True)  # Trainable parameters

# Automatic differentiation
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(word_vectors.parameters(), lr=0.01)

# Computational graph
loss.backward()  # Automatic gradient computation
optimizer.step()  # Parameter updates
```

## Language Modeling Basics

### Probability Distributions

```python
# Categorical distribution
def categorical_distribution(probs):
    return probs / np.sum(probs)

# Word frequency distribution (Zipf's law)
def zipf_distribution(vocab_size, exponent=1.0):
    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / (ranks ** exponent)
    return probs / np.sum(probs)
```

### Information Theory

```python
# Entropy (uncertainty measure)
def entropy(prob_distribution):
    return -np.sum(prob_distribution * np.log2(prob_distribution + 1e-10))

# Cross-entropy
def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-10))
```

## Matrix Calculus for Neural Networks

### Forward Propagation

```python
def forward_pass(x, W1, b1, W2, b2):
    """Two-layer neural network"""
    # Layer 1: Linear transformation
    h1 = x @ W1.T + b1  # (batch_size, hidden_dim)
    
    # Layer 2: Non-linearity + Linear
    z1 = torch.relu(h1) @ W2.T + b2  # (batch_size, output_dim)
    
    return z1
```

### Backward Propagation

```python
def backward_pass(x, y, W1, b1, W2, b2):
    """Compute gradients for two-layer network"""
    batch_size = x.shape[0]
    
    # Output layer gradients
    dz1 = (z1 - y) / batch_size  # (batch_size, output_dim)
    
    # Hidden layer 2 gradients
    dh1 = (dz1 @ W2) * torch.relu_derivative(h1)  # Chain rule
    dW2 = (h1.T @ dz1) / batch_size
    db2 = torch.mean(dz1, dim=0)  # Bias gradient
    
    # Hidden layer 1 gradients
    dz1_input = (dz1 @ W2.T) * torch.relu_derivative(h1)  # Chain rule
    dW1 = (x.T @ dz1_input) / batch_size
    db1 = torch.mean(dz1_input, dim=0)  # Bias gradient
    
    return dW1, db1, dW2, db2
```

### Jacobian Matrices

```python
def compute_jacobian(network_fn, x, params):
    """Numerical Jacobian computation"""
    eps = 1e-8
    jacobian = np.zeros((len(params), x.shape[-1]))
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = network_fn(x, params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = network_fn(x, params_minus)
        
        jacobian[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return jacobian
```

## Computational Graphs

### Forward Propagation Visualization

```python
def build_computation_graph(layers):
    """Build directed acyclic graph for forward pass"""
    graph = {}
    
    # Input node
    graph['input'] = {'type': 'input', 'outputs': []}
    
    for i, layer in enumerate(layers):
        layer_name = f'layer_{i}'
        
        if i == 0:
            # Connect input to first layer
            graph['input']['outputs'].append(layer_name)
        else:
            # Connect layers sequentially
            prev_layer = f'layer_{i-1}'
            graph[prev_layer]['outputs'].append(layer_name)
            graph[layer_name] = {
                'type': 'operation',
                'operation': 'linear' if i < len(layers)-1 else 'nonlinear',
                'inputs': [f'layer_{i-1}'],
                'outputs': [],
                'params': [f'W_{i}', f'b_{i}']
            }
    
    return graph
```

### Gradient Flow Visualization

```python
def visualize_gradients(network_fn, x, y, params):
    """Visualize gradient flow through network"""
    gradients = {}
    
    # Compute gradients at each layer
    for i in range(len(params)):
        layer_grad = f'grad_layer_{i}'
        gradients[layer_grad] = compute_layer_gradient(i, x, y, params)
    
    # Visualize gradient magnitudes
    grad_magnitudes = {k: np.linalg.norm(v) for k, v in gradients.items()}
    
    return gradients, grad_magnitudes
```

## Practical Python Tips

### Efficient Matrix Operations

```python
# Vectorized operations (avoid loops)
def batch_matrix_multiply(X, W):
    """Efficient batch matrix multiplication"""
    return X @ W  # (batch_size, input_dim) @ (input_dim, hidden_dim)

# Broadcasting for bias addition
def add_bias(pre_activation, bias):
    """Add bias with proper broadcasting"""
    return pre_activation + bias[np.newaxis, :]  # (hidden_dim,) + (1, hidden_dim)
```

### Numerical Stability

```python
# Gradient clipping
def clip_gradients(gradients, max_norm=1.0):
    """Prevent gradient explosion"""
    total_norm = np.sqrt(sum(np.linalg.norm(g)**2 for g in gradients.values())
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for k in gradients.keys():
            gradients[k] *= scale
    
    return gradients

# Learning rate scheduling
def learning_rate_schedule(initial_lr, epoch, decay_rate=0.95):
    """Exponential decay"""
    return initial_lr * (decay_rate ** epoch)
```

### Memory Management

```python
# Checkpointing
def save_checkpoint(model, optimizer, epoch, loss):
    """Save model state"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Gradient accumulation for small batches
def accumulate_gradients(gradients_list):
    """Accumulate gradients from multiple steps"""
    accumulated = {}
    for grads in gradients_list:
        for k, v in grads.items():
            if k not in accumulated:
                accumulated[k] = v.copy()
            else:
                accumulated[k] += v
    return accumulated
```

## Advanced Neural Concepts

### Universal Approximation Theorem

**Key Insight**: Neural networks can approximate any continuous function given sufficient hidden units

**Mathematical Foundation**:

- **Cybenko's Theorem**: Universal approximation capabilities
- **Stone-Weierstrass Theorem**: One hidden layer sufficient for boolean functions
- **Kolmogorov's Theorem**: Two hidden layers sufficient for any function

### Regularization Theory

```python
# L2 regularization
def l2_regularization(weights, lambda_reg=0.01):
    """L2 penalty on large weights"""
    return lambda_reg * np.sum(weights**2)

# Dropout (during training)
def dropout_layer(activations, p=0.5, training=True):
    """Random neuron dropout"""
    if training:
        mask = (np.random.rand(*activations.shape) > p) / (1 - p)
        return activations * mask
    else:
        # During inference: scale by dropout probability
        return activations * p
```

### Optimization Theory

```python
# Convergence analysis
def analyze_convergence(loss_history):
    """Analyze training convergence"""
    losses = np.array(loss_history)
    
    # Moving average
    window_size = min(100, len(losses) // 4)
    moving_avg = np.convolve(losses, np.ones(window_size) / window_size)[window_size//2:]
    
    # Gradient norm analysis
    grad_norms = [np.linalg.norm(g) for g in gradient_history]
    
    return {
        'final_loss': losses[-1],
        'moving_average': moving_avg[-1] if len(moving_avg) > 0 else losses[-1],
        'convergence_rate': (losses[0] - losses[-1]) / losses[0],
        'gradient_stability': np.std(grad_norms[-100:]) if len(grad_norms) > 100 else None
    }
```

## Key Mathematical Insights

### Chain Rule Mastery

**Multi-layer Networks**: Gradient flows through multiple transformations
**Parameter Sharing**: Gradients must respect shared parameters across layers
**Computational Efficiency**: Vectorized operations vs. explicit loops

### Backpropagation Intuition

**Error Signals**: Gradients tell us how to adjust each parameter
**Learning Dynamics**: Small steps in parameter space following negative gradients
**Convergence Landscape**: Non-convex optimization with multiple local minima

## Implementation Best Practices

### Debugging Neural Networks

```python
# Gradient checking
def check_gradients(network_fn, x, y, params, epsilon=1e-7):
    """Numerical gradient verification"""
    numerical_grads = {}
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        f_plus = network_fn(x, params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= epsilon
        f_minus = network_fn(x, params_minus)
        
        numerical_grad = (f_plus - f_minus) / (2 * epsilon)
        analytical_grad = compute_analytical_gradient(i, x, y, params)
        
        # Relative error
        relative_error = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(analytical_grad)
        numerical_grads[f'param_{i}'] = numerical_grad
    
    return numerical_grads, relative_error

# Loss landscape visualization
def plot_loss_surface(loss_fn, x_range, y_range):
    """3D visualization of loss surface"""
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[loss_fn(x, y) for x, y in zip(X.ravel(), Y.ravel())])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Loss')
    plt.title('Neural Network Loss Surface')
    plt.show()
```

## Key Takeaways

1. **Mathematical Foundation**: Matrix calculus enables efficient neural network computation
2. **Computational Graphs**: Forward/backward propagation as systematic gradient flow
3. **Jacobian Matrices**: Efficient way to compute all parameter gradients simultaneously
4. **Numerical Stability**: Essential for stable training of deep networks
5. **Python Ecosystem**: Rich toolchain for practical deep learning implementation

---

# LectureNotes #CS224N #Backpropagation #NeuralNetworks #Python #MatrixCalculus
