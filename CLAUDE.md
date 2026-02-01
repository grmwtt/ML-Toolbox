# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is an **educational ML toolbox** containing from-scratch NumPy implementations of 21 fundamental machine learning algorithms. Each implementation is in a Jupyter notebook with theory, code, diagnostics, visualizations, use cases, and sklearn comparison.

**Key principle**: All core algorithms use **only NumPy** (no sklearn for implementation). Sklearn is used only for datasets and validation comparison.

## Repository Architecture

### Directory Organization

```
ML-Toolbox/
├── supervised/
│   ├── classification/  # 7 notebooks: logistic_regression, naive_bayes, decision_trees,
│   │                    #              random_forest, svm, knn, neural_network
│   └── regression/      # 6 notebooks: linear_regression, polynomial_regression,
│                        #              ridge_regression, lasso_regression,
│                        #              elastic_net, gradient_boosting
├── unsupervised/
│   ├── clustering/      # 4 notebooks: kmeans, hierarchical_clustering, dbscan,
│   │                    #              gaussian_mixture_models
│   └── dimensionality_reduction/  # 4 notebooks: pca, lda, tsne, autoencoders
├── utils/               # Shared utilities (all NumPy-based)
│   ├── evaluation_metrics.py    # ClassificationMetrics, RegressionMetrics, ClusteringMetrics
│   ├── preprocessing.py         # Scalers, encoders, train_test_split, etc.
│   ├── visualization.py         # MLVisualizer, DiagnosticPlots classes
│   └── cross_validation.py      # KFold, GridSearchCV, learning_curve, etc.
└── data/
    └── sample_datasets.py       # Dataset loaders with metadata (uses sklearn.datasets)
```

### Utility Module Design Pattern

All utility modules follow this pattern:
- **Static methods** in classes for organization (e.g., `ClassificationMetrics.accuracy()`)
- **Type hints** for all function signatures
- **NumPy-only** implementations (no sklearn/scipy except where noted)
- **Comprehensive docstrings** with formulas

Example usage:
```python
from utils import StandardScaler, ClassificationMetrics, KFold
from data import load_breast_cancer

# Load data with metadata
X, y, info = load_breast_cancer()

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kfold.split(X, y):
    # Training logic
    pass
```

### Notebook Structure (Standardized)

Every algorithm notebook has exactly **7 sections**:

1. **Theory Section** - LaTeX math, assumptions, complexity analysis
2. **Implementation from Scratch** - NumPy-only class with fit(), predict(), score()
3. **Training & Optimization** - Real dataset from sklearn.datasets
4. **Diagnostics & Evaluation** - Metrics, learning curves, convergence analysis
5. **Visualizations** - Decision boundaries, feature importance, residual plots, etc.
6. **Use Cases & Guidelines** - When to use/avoid, pros/cons tables, parameter selection
7. **Comparison with sklearn** - Validation, performance benchmarking

**Critical**: When creating new notebooks, maintain this exact structure for consistency.

## Common Development Tasks

### Running Notebooks
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Navigate to any notebook and run cells sequentially
```

### Validating Notebook Structure
```bash
# Check all notebooks are valid JSON
python3 -c "
import json, os
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.ipynb'):
            with open(os.path.join(root, f)) as nb:
                json.load(nb)
                print(f'✓ {os.path.join(root, f)}')
"
```

### Testing Utility Modules
```python
# Example: Test preprocessing module
from utils.preprocessing import StandardScaler, train_test_split
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Test scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

# Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
assert len(X_train) + len(X_test) == len(X)
```

## Important Implementation Details

### CPU Optimization
All implementations are optimized for **CPU-only execution**:
- Small hidden layer sizes for neural networks (e.g., [64, 32])
- Limited iterations/estimators (e.g., n_estimators=10-50 for Random Forest)
- Small sample sizes for O(n²) algorithms like t-SNE (300-500 samples)
- Vectorized NumPy operations wherever possible

### Numerical Stability
Key techniques used throughout:
- **Log-space calculations** for probabilities (e.g., Naive Bayes)
- **Cholesky decomposition** instead of matrix inversion (e.g., Linear Regression)
- **Gradient clipping** in neural networks
- **Regularization** to prevent singular matrices (e.g., LDA adds small epsilon to covariance)

### Algorithm-Specific Notes

**Coordinate Descent** (Lasso, Elastic Net):
- Precompute column norms for efficiency
- Use soft-thresholding operator: `sign(x) * max(|x| - lambda, 0)`

**EM Algorithm** (GMM):
- Initialize with K-Means or random
- Use Cholesky for covariance matrix stability
- Track log-likelihood for convergence

**Tree-based methods** (Decision Trees, Random Forest, Gradient Boosting):
- Recursive splitting with depth tracking
- Feature importance via total decrease in impurity
- Support for both Gini and Entropy

**Neural Networks**:
- Xavier/He weight initialization
- Multiple optimizers (SGD, Momentum, Adam)
- Early stopping with validation set

## Dataset Usage Guidelines

From `data/sample_datasets.py`:
- All loaders return `(X, y, info)` tuple where `info` is a dict with metadata
- Use `print_dataset_info(info)` for formatted output
- Choose datasets based on algorithm type (see function docstrings)

**Algorithm → Dataset mapping**:
- Logistic Regression, SVM, Neural Networks → `load_breast_cancer()` (binary, 30 features)
- KNN, Naive Bayes, Decision Trees → `load_iris()` (multiclass, 4 features)
- Regression algorithms → `load_diabetes()` or `load_california_housing()` (subset)
- Clustering → `make_blobs()`, `make_moons()`, `make_circles()` with custom generators
- Dimensionality reduction → `load_digits()` (64 features, good for PCA/t-SNE)

## Quick Reference Tables

The README contains decision tables for algorithm selection:
- **Classification**: Logistic Regression, Naive Bayes, Decision Trees, Random Forest, SVM, KNN, Neural Network
- **Regression**: Linear, Polynomial, Ridge, Lasso, Elastic Net, Gradient Boosting
- **Clustering**: K-Means, Hierarchical, DBSCAN, GMM
- **Dimensionality Reduction**: PCA, LDA, t-SNE, Autoencoders

Each table shows "Best For" and "Avoid When" scenarios - refer to these when helping users choose algorithms.

## Code Style Conventions

- **Type hints** on all function/method signatures
- **Docstrings** follow Google style with formulas where applicable
- **Class naming**: `AlgorithmNameScratch` or just `AlgorithmName` for from-scratch implementations
- **Variable naming**: `X` for features, `y` for labels, `n_samples`, `n_features`, `n_classes`
- **Private methods**: Prefix with `_` (e.g., `_compute_distances()`)
- **Verbose output**: Optional `verbose` parameter, print progress every N iterations

## Known Limitations & Trade-offs

1. **No production optimization**: These are educational implementations. Sklearn is 10-100x faster.
2. **No GPU support**: All CPU-based using NumPy.
3. **Limited spatial indexing**: KNN/DBSCAN use brute force O(n²) without KD-trees.
4. **No sparse matrix support**: All dense NumPy arrays.
5. **Memory constraints**: Some algorithms (t-SNE, Hierarchical Clustering) limited to small datasets.

When helping users, always clarify that this is for **learning**, not production deployment.
