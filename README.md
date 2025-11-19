# Machine Learning Toolbox

A comprehensive collection of machine learning algorithms implemented from scratch with detailed explanations, diagnostics, and visualizations.

## 📚 Overview

This repository contains custom implementations of fundamental machine learning algorithms, each accompanied by:
- Mathematical foundations and intuition
- Step-by-step implementation from scratch
- Diagnostic tools and evaluation metrics
- Visualization of results and decision boundaries
- Comparison with sklearn implementations
- Use cases and appropriate data types

## 🗂️ Repository Structure

```
ml-toolbox/
├── supervised/
│   ├── classification/
│   │   ├── logistic_regression.ipynb
│   │   ├── naive_bayes.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── random_forest.ipynb
│   │   ├── svm.ipynb
│   │   ├── knn.ipynb
│   │   └── neural_network.ipynb
│   └── regression/
│       ├── linear_regression.ipynb
│       ├── polynomial_regression.ipynb
│       ├── ridge_regression.ipynb
│       ├── lasso_regression.ipynb
│       ├── elastic_net.ipynb
│       └── gradient_boosting.ipynb
├── unsupervised/
│   ├── clustering/
│   │   ├── kmeans.ipynb
│   │   ├── hierarchical_clustering.ipynb
│   │   ├── dbscan.ipynb
│   │   └── gaussian_mixture_models.ipynb
│   └── dimensionality_reduction/
│       ├── pca.ipynb
│       ├── lda.ipynb
│       ├── tsne.ipynb
│       └── autoencoders.ipynb
├── utils/
│   ├── evaluation_metrics.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── cross_validation.py
└── data/
    └── sample_datasets.py
```

## 🔧 Each Notebook Contains

### 1. **Theory Section**
- Mathematical formulation
- Intuitive explanation
- Assumptions and limitations
- Time and space complexity

### 2. **Implementation**
- Custom implementation from scratch using NumPy
- Clear, commented code with docstrings
- Modular design for reusability

### 3. **Diagnostics & Evaluation**
- Training curves (loss, accuracy over iterations)
- Learning curves (performance vs training size)
- Validation curves (hyperparameter tuning)
- Confusion matrices and classification reports
- Feature importance analysis (where applicable)
- Residual plots (for regression)

### 4. **Visualizations**
- Decision boundaries (2D and 3D)
- Feature distributions
- Model predictions vs actual values
- ROC curves and precision-recall curves
- Interactive plots using plotly

### 5. **Use Cases & Guidelines**
- When to use this algorithm
- Types of data it works best with
- Pros and cons
- Common pitfalls and how to avoid them

## 🎯 Learning Objectives

Each implementation helps understand:
- The math behind the algorithm
- How to code it from scratch
- How to evaluate and diagnose model performance
- When and why to use specific algorithms
- How to interpret results

## 📊 Evaluation Metrics Implemented

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Log Loss
- Cohen's Kappa
- Matthews Correlation Coefficient

### Regression
- MSE, RMSE, MAE
- R², Adjusted R²
- Mean Absolute Percentage Error (MAPE)
- Explained Variance Score

### Clustering
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Within-Cluster Sum of Squares (Elbow Method)

## 🚀 Getting Started

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Navigate to any algorithm notebook
4. Run cells sequentially to see implementation and results

## 📝 Implementation Checklist

### Supervised Learning - Classification
- [ ] Logistic Regression (Binary & Multiclass)
- [ ] Naive Bayes (Gaussian, Multinomial, Bernoulli)
- [ ] Decision Trees (ID3, C4.5, CART)
- [ ] Random Forest
- [ ] Support Vector Machines
- [ ] k-Nearest Neighbors
- [ ] Neural Network (Basic MLP)

### Supervised Learning - Regression
- [ ] Linear Regression (OLS, Gradient Descent)
- [ ] Polynomial Regression
- [ ] Ridge Regression (L2)
- [ ] Lasso Regression (L1)
- [ ] Elastic Net
- [ ] Gradient Boosting

### Unsupervised Learning - Clustering
- [ ] K-Means
- [ ] Hierarchical Clustering
- [ ] DBSCAN
- [ ] Gaussian Mixture Models

### Unsupervised Learning - Dimensionality Reduction
- [ ] Principal Component Analysis (PCA)
- [ ] Linear Discriminant Analysis (LDA)
- [ ] t-SNE
- [ ] Autoencoders

## 💡 Key Features

- **No sklearn for core algorithms** - Everything implemented using NumPy
- **Comprehensive testing** - Each implementation tested against sklearn
- **Educational focus** - Clear explanations and step-by-step derivations
- **Practical examples** - Real datasets with meaningful interpretations
- **Modular code** - Reusable components across different algorithms

## 📚 Resources & References

- Pattern Recognition and Machine Learning - Bishop
- The Elements of Statistical Learning - Hastie, Tibshirani, Friedman
- Machine Learning - Tom Mitchell
- Deep Learning - Ian Goodfellow

---

**Note**: This is a learning-focused repository. For production use, prefer optimized libraries like scikit-learn, TensorFlow, or PyTorch.
