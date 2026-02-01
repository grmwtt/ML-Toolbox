"""
Sample Datasets Module
======================
Functions to load and generate sample datasets for ML experiments.
Uses sklearn datasets and synthetic data generation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn import datasets


# =============================================================================
# CLASSIFICATION DATASETS
# =============================================================================

def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Breast Cancer Wisconsin dataset.

    Binary classification: Malignant vs Benign tumors.
    569 samples, 30 features.

    Best for: Logistic Regression, SVM, Neural Networks
    """
    data = datasets.load_breast_cancer()
    info = {
        'name': 'Breast Cancer Wisconsin',
        'task': 'Binary Classification',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_names': list(data.target_names),
        'description': 'Predict if a tumor is malignant or benign based on cell features.'
    }
    return data.data, data.target, info


def load_iris() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Iris dataset.

    Multiclass classification: 3 species of iris flowers.
    150 samples, 4 features.

    Best for: KNN, Decision Trees, Naive Bayes
    """
    data = datasets.load_iris()
    info = {
        'name': 'Iris',
        'task': 'Multiclass Classification (3 classes)',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_names': list(data.target_names),
        'description': 'Classify iris flowers into 3 species based on petal and sepal measurements.'
    }
    return data.data, data.target, info


def load_wine() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Wine recognition dataset.

    Multiclass classification: 3 wine cultivars.
    178 samples, 13 features.

    Best for: Random Forest, SVM, Decision Trees
    """
    data = datasets.load_wine()
    info = {
        'name': 'Wine',
        'task': 'Multiclass Classification (3 classes)',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_names': [f'Class {i}' for i in data.target_names],
        'description': 'Classify wines into 3 cultivars based on chemical analysis.'
    }
    return data.data, data.target, info


def load_digits() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Digits dataset.

    Multiclass classification: Handwritten digits 0-9.
    1797 samples, 64 features (8x8 images).

    Best for: Neural Networks, SVM, KNN
    """
    data = datasets.load_digits()
    info = {
        'name': 'Digits',
        'task': 'Multiclass Classification (10 classes)',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': [f'pixel_{i}' for i in range(64)],
        'target_names': [str(i) for i in range(10)],
        'description': 'Recognize handwritten digits (0-9) from 8x8 pixel images.'
    }
    return data.data, data.target, info


def make_classification_data(n_samples: int = 1000, n_features: int = 20,
                            n_informative: int = 10, n_classes: int = 2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic classification data.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features
    n_classes : int
        Number of classes
    random_state : int
        Random seed
    """
    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative - 2,
        n_classes=n_classes,
        random_state=random_state,
        n_clusters_per_class=1
    )
    info = {
        'name': 'Synthetic Classification',
        'task': f'{"Binary" if n_classes == 2 else "Multiclass"} Classification',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'n_classes': n_classes,
        'description': 'Synthetic dataset for testing classification algorithms.'
    }
    return X, y, info


def make_moons(n_samples: int = 1000, noise: float = 0.1,
              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate two interleaving half circles (moons).

    Good for demonstrating non-linear decision boundaries.
    """
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise,
                               random_state=random_state)
    info = {
        'name': 'Moons',
        'task': 'Binary Classification (Non-linear)',
        'n_samples': n_samples,
        'n_features': 2,
        'description': 'Two interleaving half circles. Good for non-linear classifiers.'
    }
    return X, y, info


def make_circles(n_samples: int = 1000, noise: float = 0.1,
                factor: float = 0.5, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a large circle containing a smaller circle.

    Good for demonstrating RBF kernels and non-linear boundaries.
    """
    X, y = datasets.make_circles(n_samples=n_samples, noise=noise,
                                 factor=factor, random_state=random_state)
    info = {
        'name': 'Circles',
        'task': 'Binary Classification (Non-linear)',
        'n_samples': n_samples,
        'n_features': 2,
        'description': 'Concentric circles. Linear classifiers fail here.'
    }
    return X, y, info


# =============================================================================
# REGRESSION DATASETS
# =============================================================================

def load_california_housing() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the California Housing dataset.

    Regression: Predict median house value.
    20,640 samples, 8 features.

    Best for: Linear Regression, Gradient Boosting, Neural Networks
    """
    data = datasets.fetch_california_housing()
    info = {
        'name': 'California Housing',
        'task': 'Regression',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_name': 'Median House Value ($100,000s)',
        'description': 'Predict median house value in California districts.'
    }
    return data.data, data.target, info


def load_diabetes() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the Diabetes dataset.

    Regression: Predict disease progression.
    442 samples, 10 features.

    Best for: Linear/Ridge/Lasso Regression
    """
    data = datasets.load_diabetes()
    info = {
        'name': 'Diabetes',
        'task': 'Regression',
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_name': 'Disease Progression',
        'description': 'Predict diabetes progression based on patient measurements.'
    }
    return data.data, data.target, info


def make_regression_data(n_samples: int = 1000, n_features: int = 10,
                        n_informative: int = 5, noise: float = 10.0,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic regression data.
    """
    X, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    info = {
        'name': 'Synthetic Regression',
        'task': 'Regression',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'noise': noise,
        'description': 'Synthetic dataset for testing regression algorithms.'
    }
    return X, y, info


def make_polynomial_data(n_samples: int = 200, degree: int = 3,
                        noise: float = 5.0, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate polynomial regression data.

    y = sum(coef_i * x^i) + noise
    """
    np.random.seed(random_state)
    X = np.sort(np.random.uniform(-3, 3, n_samples)).reshape(-1, 1)

    # Generate random coefficients
    coefs = np.random.randn(degree + 1)
    y = sum(coefs[i] * X.flatten() ** i for i in range(degree + 1))
    y += np.random.normal(0, noise, n_samples)

    info = {
        'name': 'Polynomial Data',
        'task': 'Polynomial Regression',
        'n_samples': n_samples,
        'degree': degree,
        'coefficients': coefs.tolist(),
        'description': f'Polynomial data of degree {degree} with noise.'
    }
    return X, y, info


def make_nonlinear_data(n_samples: int = 500, noise: float = 0.5,
                       function: str = 'sin', random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate non-linear regression data.

    Parameters:
    -----------
    function : str
        'sin', 'exp', 'log', or 'complex'
    """
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)

    if function == 'sin':
        y = np.sin(X.flatten()) + np.random.normal(0, noise, n_samples)
        desc = 'y = sin(x) + noise'
    elif function == 'exp':
        X = np.linspace(0, 3, n_samples).reshape(-1, 1)
        y = np.exp(X.flatten()) + np.random.normal(0, noise * 5, n_samples)
        desc = 'y = exp(x) + noise'
    elif function == 'log':
        X = np.linspace(0.1, 10, n_samples).reshape(-1, 1)
        y = np.log(X.flatten()) + np.random.normal(0, noise, n_samples)
        desc = 'y = log(x) + noise'
    elif function == 'complex':
        y = (np.sin(X.flatten()) * np.exp(-0.1 * X.flatten()) +
             0.5 * np.cos(2 * X.flatten()) + np.random.normal(0, noise, n_samples))
        desc = 'y = sin(x)*exp(-0.1x) + 0.5*cos(2x) + noise'
    else:
        raise ValueError(f"Unknown function: {function}")

    info = {
        'name': f'Nonlinear Data ({function})',
        'task': 'Nonlinear Regression',
        'n_samples': n_samples,
        'function': function,
        'description': desc
    }
    return X, y, info


# =============================================================================
# CLUSTERING DATASETS
# =============================================================================

def make_blobs(n_samples: int = 500, n_features: int = 2,
              centers: int = 3, cluster_std: float = 1.0,
              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate isotropic Gaussian blobs for clustering.

    Best for: K-Means, GMM
    """
    X, y = datasets.make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    info = {
        'name': 'Gaussian Blobs',
        'task': 'Clustering',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_clusters': centers,
        'description': 'Isotropic Gaussian blobs. Ideal for K-Means clustering.'
    }
    return X, y, info


def make_varied_blobs(n_samples: int = 500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate blobs with varied cluster sizes and densities.

    Good for testing adaptive clustering methods.
    """
    np.random.seed(random_state)

    # Create clusters with different sizes and variances
    n1, n2, n3 = n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)

    X1 = np.random.randn(n1, 2) * 0.5 + np.array([0, 0])
    X2 = np.random.randn(n2, 2) * 1.5 + np.array([5, 5])
    X3 = np.random.randn(n3, 2) * 0.8 + np.array([2, 8])

    X = np.vstack([X1, X2, X3])
    y = np.array([0] * n1 + [1] * n2 + [2] * n3)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    info = {
        'name': 'Varied Blobs',
        'task': 'Clustering',
        'n_samples': n_samples,
        'n_features': 2,
        'n_clusters': 3,
        'description': 'Clusters with different sizes and densities.'
    }
    return X, y, info


def make_nested_clusters(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate nested/concentric clusters.

    Good for DBSCAN and spectral clustering.
    """
    X, y = datasets.make_circles(n_samples=n_samples, factor=0.5,
                                 noise=0.05, random_state=random_state)

    # Add a third cluster
    np.random.seed(random_state)
    n_extra = n_samples // 4
    X_extra = np.random.randn(n_extra, 2) * 0.3 + np.array([3, 0])

    X = np.vstack([X, X_extra])
    y = np.concatenate([y, np.full(n_extra, 2)])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    info = {
        'name': 'Nested Clusters',
        'task': 'Clustering',
        'n_samples': len(X),
        'n_features': 2,
        'n_clusters': 3,
        'description': 'Nested circles plus an outlier cluster. K-Means fails here.'
    }
    return X, y, info


def make_elongated_clusters(n_samples: int = 500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate elongated (non-spherical) clusters.

    Good for demonstrating GMM advantages over K-Means.
    """
    np.random.seed(random_state)

    n_per_cluster = n_samples // 3

    # Elongated cluster 1 (diagonal)
    X1 = np.random.randn(n_per_cluster, 2)
    X1 = X1 @ np.array([[2, 1], [1, 0.5]]) + np.array([0, 0])

    # Elongated cluster 2 (horizontal)
    X2 = np.random.randn(n_per_cluster, 2)
    X2 = X2 @ np.array([[3, 0], [0, 0.5]]) + np.array([8, 2])

    # Compact cluster 3
    X3 = np.random.randn(n_samples - 2 * n_per_cluster, 2) * 0.5 + np.array([4, -3])

    X = np.vstack([X1, X2, X3])
    y = np.array([0] * n_per_cluster + [1] * n_per_cluster +
                 [2] * (n_samples - 2 * n_per_cluster))

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    info = {
        'name': 'Elongated Clusters',
        'task': 'Clustering',
        'n_samples': n_samples,
        'n_features': 2,
        'n_clusters': 3,
        'description': 'Non-spherical clusters. GMM outperforms K-Means here.'
    }
    return X, y, info


def make_density_clusters(n_samples: int = 500, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate clusters with varying densities plus noise points.

    Ideal for DBSCAN.
    """
    np.random.seed(random_state)

    # Dense cluster
    n1 = n_samples // 3
    X1 = np.random.randn(n1, 2) * 0.3 + np.array([0, 0])

    # Sparse cluster
    n2 = n_samples // 3
    X2 = np.random.randn(n2, 2) * 1.5 + np.array([5, 5])

    # Noise points
    n_noise = n_samples - n1 - n2
    X_noise = np.random.uniform(-3, 10, (n_noise, 2))

    X = np.vstack([X1, X2, X_noise])
    y = np.array([0] * n1 + [1] * n2 + [-1] * n_noise)  # -1 for noise

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    info = {
        'name': 'Density Clusters',
        'task': 'Clustering',
        'n_samples': n_samples,
        'n_features': 2,
        'n_clusters': 2,
        'n_noise': n_noise,
        'description': 'Varying density clusters with noise points. Ideal for DBSCAN.'
    }
    return X, y, info


# =============================================================================
# DIMENSIONALITY REDUCTION DATASETS
# =============================================================================

def make_swiss_roll(n_samples: int = 1000, noise: float = 0.5,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate Swiss roll dataset.

    3D manifold that can be unrolled to 2D.
    Good for testing manifold learning (t-SNE, autoencoders).
    """
    X, y = datasets.make_swiss_roll(n_samples=n_samples, noise=noise,
                                    random_state=random_state)
    info = {
        'name': 'Swiss Roll',
        'task': 'Dimensionality Reduction',
        'n_samples': n_samples,
        'n_features': 3,
        'intrinsic_dim': 2,
        'description': '3D manifold that unfolds to 2D. Tests manifold learning.'
    }
    return X, y, info


def make_s_curve(n_samples: int = 1000, noise: float = 0.1,
                random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate S-curve dataset.

    Similar to Swiss roll but S-shaped.
    """
    X, y = datasets.make_s_curve(n_samples=n_samples, noise=noise,
                                 random_state=random_state)
    info = {
        'name': 'S-Curve',
        'task': 'Dimensionality Reduction',
        'n_samples': n_samples,
        'n_features': 3,
        'intrinsic_dim': 2,
        'description': '3D S-curve that can be flattened to 2D.'
    }
    return X, y, info


def make_high_dim_clusters(n_samples: int = 500, n_features: int = 50,
                          n_informative: int = 10, n_clusters: int = 5,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate high-dimensional data with embedded clusters.

    Good for testing PCA and other dimensionality reduction.
    """
    X, y = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative - 5,
        n_classes=n_clusters,
        n_clusters_per_class=1,
        random_state=random_state
    )
    info = {
        'name': 'High-Dim Clusters',
        'task': 'Dimensionality Reduction',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_informative': n_informative,
        'n_clusters': n_clusters,
        'description': f'High-dimensional data ({n_features}D) with {n_clusters} embedded clusters.'
    }
    return X, y, info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dataset_summary() -> str:
    """
    Return a summary of all available datasets.
    """
    summary = """
    AVAILABLE DATASETS
    ==================

    CLASSIFICATION:
    ---------------
    - load_breast_cancer(): Binary classification, 569 samples, 30 features
    - load_iris(): Multiclass (3), 150 samples, 4 features
    - load_wine(): Multiclass (3), 178 samples, 13 features
    - load_digits(): Multiclass (10), 1797 samples, 64 features
    - make_classification_data(): Synthetic, customizable
    - make_moons(): Non-linear binary, 2D
    - make_circles(): Non-linear binary, 2D

    REGRESSION:
    -----------
    - load_california_housing(): 20,640 samples, 8 features
    - load_diabetes(): 442 samples, 10 features
    - make_regression_data(): Synthetic, customizable
    - make_polynomial_data(): Polynomial relationship
    - make_nonlinear_data(): Various non-linear functions

    CLUSTERING:
    -----------
    - make_blobs(): Gaussian blobs, ideal for K-Means
    - make_varied_blobs(): Different sizes/densities
    - make_nested_clusters(): Concentric circles + blob
    - make_elongated_clusters(): Non-spherical, good for GMM
    - make_density_clusters(): Varying density + noise, ideal for DBSCAN

    DIMENSIONALITY REDUCTION:
    -------------------------
    - make_swiss_roll(): 3D manifold
    - make_s_curve(): 3D S-curve
    - make_high_dim_clusters(): High-dimensional with embedded clusters
    """
    return summary


def print_dataset_info(info: Dict[str, Any]) -> None:
    """
    Pretty print dataset information.
    """
    print("\n" + "=" * 50)
    print(f"Dataset: {info.get('name', 'Unknown')}")
    print("=" * 50)
    print(f"Task: {info.get('task', 'Unknown')}")
    print(f"Samples: {info.get('n_samples', 'Unknown')}")
    print(f"Features: {info.get('n_features', 'Unknown')}")
    if 'target_names' in info:
        print(f"Classes: {info['target_names']}")
    if 'feature_names' in info and len(info['feature_names']) <= 10:
        print(f"Features: {info['feature_names']}")
    print(f"\nDescription: {info.get('description', 'No description')}")
    print("=" * 50)
