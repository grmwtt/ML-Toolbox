"""
Preprocessing Utilities Module
==============================
Custom implementations of common preprocessing techniques.
"""

import numpy as np
from typing import Optional, Tuple, List, Union


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    z = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for scaling.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to transform
        """
        X = np.array(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a given range [min, max].

    x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max to be used for scaling.
        """
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)

        data_range = self.data_max_ - self.data_min_
        # Avoid division by zero
        data_range[data_range == 0] = 1.0

        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features."""
        X = np.array(X)
        return X * self.scale_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling."""
        return (X - self.min_) / self.scale_


class RobustScaler:
    """
    Scale features using statistics robust to outliers.

    Uses the median and IQR instead of mean and std.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """Compute median and IQR."""
        X = np.array(X)
        self.center_ = np.median(X, axis=0)

        q_min, q_max = self.quantile_range
        q = np.percentile(X, [q_min, q_max], axis=0)
        self.scale_ = q[1] - q[0]
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features."""
        X = np.array(X)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class Normalizer:
    """
    Normalize samples individually to unit norm.
    """

    def __init__(self, norm: str = 'l2'):
        """
        Parameters:
        -----------
        norm : str
            'l1', 'l2', or 'max'
        """
        self.norm = norm

    def fit(self, X: np.ndarray) -> 'Normalizer':
        """Fit (no-op for Normalizer)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize each sample."""
        X = np.array(X, dtype=float)

        if self.norm == 'l1':
            norms = np.abs(X).sum(axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        elif self.norm == 'max':
            norms = np.abs(X).max(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class LabelEncoder:
    """
    Encode labels as integers.
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """Find unique classes."""
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform labels to integers."""
        return np.array([self.class_to_idx_[label] for label in y])

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Transform integers back to labels."""
        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """
    Encode categorical features as one-hot vectors.
    """

    def __init__(self, sparse: bool = False):
        self.sparse = sparse
        self.categories_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """Find unique categories for each feature."""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.categories_ = [np.unique(X[:, i]) for i in range(self.n_features_)]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform categorical features to one-hot."""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        n_categories = sum(len(cats) for cats in self.categories_)

        result = np.zeros((n_samples, n_categories))

        col_idx = 0
        for feature_idx, categories in enumerate(self.categories_):
            for sample_idx in range(n_samples):
                cat_idx = np.where(categories == X[sample_idx, feature_idx])[0]
                if len(cat_idx) > 0:
                    result[sample_idx, col_idx + cat_idx[0]] = 1
            col_idx += len(categories)

        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.
    """

    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X: np.ndarray) -> 'PolynomialFeatures':
        """Compute number of output features."""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features."""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Start with bias term if requested
        if self.include_bias:
            output = [np.ones((n_samples, 1))]
        else:
            output = []

        # Add original features
        output.append(X)

        # Add polynomial features
        for d in range(2, self.degree + 1):
            if self.interaction_only:
                # Only interaction terms
                from itertools import combinations
                for combo in combinations(range(n_features), d):
                    term = np.prod([X[:, i:i+1] for i in combo], axis=0)
                    output.append(term)
            else:
                # All polynomial terms
                from itertools import combinations_with_replacement
                for combo in combinations_with_replacement(range(n_features), d):
                    term = np.prod([X[:, i:i+1] for i in combo], axis=0)
                    output.append(term)

        result = np.hstack(output)
        self.n_output_features_ = result.shape[1]
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                    random_state: Optional[int] = None,
                    shuffle: bool = True,
                    stratify: Optional[np.ndarray] = None) -> Tuple:
    """
    Split data into training and test sets.

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels/targets
    test_size : float
        Proportion of data for test set
    random_state : int
        Random seed
    shuffle : bool
        Whether to shuffle before splitting
    stratify : np.ndarray
        If provided, data is split in a stratified fashion

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    if stratify is not None:
        # Stratified split
        train_indices = []
        test_indices = []

        for label in np.unique(stratify):
            label_indices = np.where(stratify == label)[0]

            if shuffle:
                np.random.shuffle(label_indices)

            n_label_test = int(len(label_indices) * test_size)
            test_indices.extend(label_indices[:n_label_test])
            train_indices.extend(label_indices[n_label_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    else:
        # Regular split
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def handle_missing_values(X: np.ndarray, strategy: str = 'mean',
                         fill_value: Optional[float] = None) -> np.ndarray:
    """
    Handle missing values in the data.

    Parameters:
    -----------
    X : np.ndarray
        Data with potential missing values (NaN)
    strategy : str
        'mean', 'median', 'most_frequent', or 'constant'
    fill_value : float
        Value to use when strategy='constant'

    Returns:
    --------
    X with missing values filled
    """
    X = np.array(X, dtype=float)

    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if not mask.any():
            continue

        if strategy == 'mean':
            fill = np.nanmean(X[:, col])
        elif strategy == 'median':
            fill = np.nanmedian(X[:, col])
        elif strategy == 'most_frequent':
            values, counts = np.unique(X[~mask, col], return_counts=True)
            fill = values[np.argmax(counts)]
        elif strategy == 'constant':
            fill = fill_value if fill_value is not None else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        X[mask, col] = fill

    return X


def remove_outliers(X: np.ndarray, y: Optional[np.ndarray] = None,
                   method: str = 'iqr', threshold: float = 1.5) -> Tuple:
    """
    Remove outliers from data.

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels (optional)
    method : str
        'iqr' (Interquartile Range) or 'zscore'
    threshold : float
        For IQR: multiplier for IQR. For zscore: number of standard deviations.

    Returns:
    --------
    X_cleaned, y_cleaned (or just X_cleaned if y is None)
    """
    X = np.array(X)

    if method == 'iqr':
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)

    elif method == 'zscore':
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = np.all(z_scores < threshold, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    if y is not None:
        return X[mask], np.array(y)[mask]
    return X[mask]


def balance_classes(X: np.ndarray, y: np.ndarray,
                   strategy: str = 'oversample',
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution.

    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    strategy : str
        'oversample' (duplicate minority) or 'undersample' (reduce majority)
    random_state : int
        Random seed

    Returns:
    --------
    X_balanced, y_balanced
    """
    if random_state is not None:
        np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)

    if strategy == 'oversample':
        target_count = counts.max()
        X_balanced, y_balanced = [], []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            n_samples = len(cls_indices)

            if n_samples < target_count:
                # Oversample
                additional = np.random.choice(cls_indices,
                                            target_count - n_samples,
                                            replace=True)
                cls_indices = np.concatenate([cls_indices, additional])

            X_balanced.append(X[cls_indices])
            y_balanced.append(y[cls_indices])

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)

    elif strategy == 'undersample':
        target_count = counts.min()
        X_balanced, y_balanced = [], []

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            selected = np.random.choice(cls_indices, target_count, replace=False)

            X_balanced.append(X[selected])
            y_balanced.append(y[selected])

        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Shuffle the result
    indices = np.random.permutation(len(y_balanced))
    return X_balanced[indices], y_balanced[indices]
