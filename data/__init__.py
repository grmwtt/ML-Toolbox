"""
Data Package
============
Sample datasets for the ML Toolbox.
"""

from .sample_datasets import (
    # Classification
    load_breast_cancer, load_iris, load_wine, load_digits,
    make_classification_data, make_moons, make_circles,
    # Regression
    load_california_housing, load_diabetes,
    make_regression_data, make_polynomial_data, make_nonlinear_data,
    # Clustering
    make_blobs, make_varied_blobs, make_nested_clusters,
    make_elongated_clusters, make_density_clusters,
    # Dimensionality reduction
    make_swiss_roll, make_s_curve, make_high_dim_clusters,
    # Utilities
    get_dataset_summary, print_dataset_info
)

__all__ = [
    'load_breast_cancer', 'load_iris', 'load_wine', 'load_digits',
    'make_classification_data', 'make_moons', 'make_circles',
    'load_california_housing', 'load_diabetes',
    'make_regression_data', 'make_polynomial_data', 'make_nonlinear_data',
    'make_blobs', 'make_varied_blobs', 'make_nested_clusters',
    'make_elongated_clusters', 'make_density_clusters',
    'make_swiss_roll', 'make_s_curve', 'make_high_dim_clusters',
    'get_dataset_summary', 'print_dataset_info'
]
