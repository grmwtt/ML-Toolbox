# Machine Learning Toolbox

Custom implementations of machine learning algorithms from scratch for educational purposes.

## Overview

This repository contains implementations of fundamental ML algorithms with:
- Mathematical foundations and derivations
- From-scratch implementations using NumPy
- Diagnostic tools and evaluation metrics
- Visualization of results and decision boundaries
- Comparison with sklearn implementations
- Use cases and practical guidelines

## Repository Structure

```
ML-Toolbox/
├── supervised/
│   ├── classification/
│   │   ├── logistic_regression.ipynb    # Binary & Multiclass classification
│   │   ├── naive_bayes.ipynb            # Gaussian, Multinomial, Bernoulli
│   │   ├── decision_trees.ipynb         # ID3, CART with Gini/Entropy
│   │   ├── random_forest.ipynb          # Ensemble of decision trees
│   │   ├── svm.ipynb                    # Support Vector Machines
│   │   ├── knn.ipynb                    # K-Nearest Neighbors
│   │   └── neural_network.ipynb         # Multi-Layer Perceptron
│   └── regression/
│       ├── linear_regression.ipynb      # OLS and Gradient Descent
│       ├── polynomial_regression.ipynb  # Polynomial features
│       ├── ridge_regression.ipynb       # L2 Regularization
│       ├── lasso_regression.ipynb       # L1 Regularization
│       ├── elastic_net.ipynb            # Combined L1+L2
│       └── gradient_boosting.ipynb      # Gradient Boosting Regressor
├── unsupervised/
│   ├── clustering/
│   │   ├── kmeans.ipynb                 # K-Means with k-means++ init
│   │   ├── hierarchical_clustering.ipynb # Agglomerative clustering
│   │   ├── dbscan.ipynb                 # Density-based clustering
│   │   └── gaussian_mixture_models.ipynb # GMM with EM algorithm
│   └── dimensionality_reduction/
│       ├── pca.ipynb                    # Principal Component Analysis
│       ├── lda.ipynb                    # Linear Discriminant Analysis
│       ├── tsne.ipynb                   # t-SNE visualization
│       └── autoencoders.ipynb           # Neural network autoencoders
├── utils/
│   ├── __init__.py
│   ├── evaluation_metrics.py            # Classification, Regression, Clustering metrics
│   ├── preprocessing.py                 # Scalers, Encoders, Data transforms
│   ├── visualization.py                 # Plotting utilities
│   └── cross_validation.py              # CV utilities and grid search
├── data/
│   ├── __init__.py
│   └── sample_datasets.py               # Dataset loaders and generators
└── requirements.txt
```

## Notebook Structure

Each notebook follows a consistent structure:

### 1. Theory Section
- Mathematical formulation with LaTeX equations
- Intuitive explanation of the algorithm
- Assumptions and limitations
- Time and space complexity analysis

### 2. Implementation
- Custom implementation from scratch using NumPy
- Clear, commented code with docstrings
- Modular design for reusability

### 3. Diagnostics & Evaluation
- Training curves (loss, accuracy over iterations)
- Learning curves (performance vs training size)
- Validation curves (hyperparameter tuning)
- Confusion matrices and classification reports
- Feature importance analysis (where applicable)
- Residual plots (for regression)

### 4. Visualizations
- Decision boundaries (2D and 3D)
- Feature distributions
- Model predictions vs actual values
- ROC curves and precision-recall curves
- Interactive plots using plotly

### 5. Use Cases & Guidelines
- When to use this algorithm
- When NOT to use this algorithm
- Types of data it works best with
- Pros and cons
- Common pitfalls and how to avoid them

### 6. Comparison with sklearn
- Side-by-side comparison with sklearn implementation
- Verification of correctness

## Quick Reference: When to Use Each Algorithm

### Classification

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| Logistic Regression | Binary/multiclass, interpretability needed, baseline | Non-linear relationships |
| Naive Bayes | Text classification, spam detection, fast training | Feature dependencies exist |
| Decision Trees | Interpretable rules, feature importance | High dimensions, prone to overfit |
| Random Forest | High accuracy, feature importance, robust | Need interpretability, memory limited |
| SVM | High-dimensional data, clear margins | Very large datasets, probability estimates |
| KNN | Simple, no training needed, non-parametric | Large datasets, high dimensions |
| Neural Network | Complex patterns, large data | Small datasets, need interpretability |

### Regression

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| Linear Regression | Linear relationships, interpretability | Non-linear, multicollinearity |
| Polynomial Regression | Known polynomial relationship | High degree (overfitting) |
| Ridge Regression | Multicollinearity, many features | Need sparse solutions |
| Lasso Regression | Feature selection, sparse models | All features important |
| Elastic Net | Correlated features, grouping effect | Simple problems |
| Gradient Boosting | Structured data, high accuracy | Need interpretability |

### Clustering

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| K-Means | Spherical clusters, known K | Non-spherical, varying densities |
| Hierarchical | Unknown K, need hierarchy | Large datasets (O(n²)) |
| DBSCAN | Arbitrary shapes, noise detection | Varying densities |
| GMM | Soft clustering, elliptical shapes | Too many components |

### Dimensionality Reduction

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| PCA | Linear relationships, preprocessing | Non-linear structure |
| LDA | Classification preprocessing | Unsupervised tasks |
| t-SNE | Visualization (2D/3D) | Need to project new data |
| Autoencoders | Non-linear, feature learning | Small datasets |

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ML-Toolbox

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Running Notebooks

```bash
jupyter notebook
```

Navigate to any algorithm notebook and run cells sequentially.

### Using the Utility Modules

```python
# Import utilities
from utils import StandardScaler, cross_val_score, ClassificationMetrics
from data import load_breast_cancer, make_blobs

# Load data
X, y, info = load_breast_cancer()

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate with cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
```

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Log Loss
- Confusion Matrix

### Regression
- MSE, RMSE, MAE
- R², Adjusted R²
- Mean Absolute Percentage Error (MAPE)

### Clustering
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Within-Cluster Sum of Squares (Elbow Method)

## Implementation Checklist

### Supervised Learning - Classification
- [x] Logistic Regression (Binary & Multiclass)
- [x] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [x] Decision Trees (CART with Gini/Entropy)
- [x] Random Forest
- [x] Support Vector Machines
- [x] k-Nearest Neighbors
- [x] Neural Network (Basic MLP)

### Supervised Learning - Regression
- [x] Linear Regression (OLS, Gradient Descent)
- [x] Polynomial Regression
- [x] Ridge Regression (L2)
- [x] Lasso Regression (L1)
- [x] Elastic Net
- [x] Gradient Boosting

### Unsupervised Learning - Clustering
- [x] K-Means
- [x] Hierarchical Clustering
- [x] DBSCAN
- [x] Gaussian Mixture Models

### Unsupervised Learning - Dimensionality Reduction
- [x] Principal Component Analysis (PCA)
- [x] Linear Discriminant Analysis (LDA)
- [x] t-SNE
- [x] Autoencoders

## Key Features

- **No sklearn for core algorithms** - Everything implemented using NumPy
- **Comprehensive testing** - Each implementation compared against sklearn
- **Educational focus** - Clear explanations and step-by-step derivations
- **Practical examples** - Real datasets with meaningful interpretations
- **Modular code** - Reusable components across different algorithms
- **CPU-optimized** - All implementations run efficiently without GPU

## References

- Pattern Recognition and Machine Learning - Bishop
- The Elements of Statistical Learning - Hastie, Tibshirani, Friedman
- Machine Learning - Tom Mitchell
- Deep Learning - Ian Goodfellow

---

**Note**: This is a learning-focused repository. For production use, prefer optimized libraries like scikit-learn, TensorFlow, or PyTorch.
