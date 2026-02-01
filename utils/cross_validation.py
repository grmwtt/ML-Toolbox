"""
Cross-Validation Utilities Module
=================================
Custom implementations of cross-validation techniques.
"""

import numpy as np
from typing import Optional, Tuple, List, Generator, Callable


class KFold:
    """
    K-Fold cross-validation iterator.

    Provides train/test indices to split data into k consecutive folds.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None):
        """
        Parameters:
        -----------
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle data before splitting
        random_state : int
            Random seed for shuffling
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator:
        """
        Generate indices to split data into training and test set.

        Parameters:
        -----------
        X : np.ndarray
            Data to split
        y : np.ndarray
            Ignored, exists for API compatibility

        Yields:
        -------
        train_indices, test_indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([indices[:current],
                                          indices[current + fold_size:]])
            yield train_indices, test_indices
            current += fold_size

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold cross-validation iterator.

    Ensures each fold has the same proportion of classes as the full dataset.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> Generator:
        """
        Generate indices for stratified k-fold split.

        Parameters:
        -----------
        X : np.ndarray
            Data to split
        y : np.ndarray
            Target labels (required for stratification)

        Yields:
        -------
        train_indices, test_indices
        """
        y = np.array(y)
        classes = np.unique(y)

        # Get indices for each class
        class_indices = {c: np.where(y == c)[0] for c in classes}

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for c in classes:
                np.random.shuffle(class_indices[c])

        # Split each class into folds
        class_fold_indices = {}
        for c in classes:
            n_class_samples = len(class_indices[c])
            fold_sizes = np.full(self.n_splits, n_class_samples // self.n_splits, dtype=int)
            fold_sizes[:n_class_samples % self.n_splits] += 1

            current = 0
            class_fold_indices[c] = []
            for fold_size in fold_sizes:
                class_fold_indices[c].append(
                    class_indices[c][current:current + fold_size]
                )
                current += fold_size

        # Yield train/test indices for each fold
        for fold in range(self.n_splits):
            test_indices = np.concatenate([class_fold_indices[c][fold] for c in classes])
            train_indices = np.concatenate([
                np.concatenate([class_fold_indices[c][f] for f in range(self.n_splits) if f != fold])
                for c in classes
            ])

            if self.shuffle:
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class LeaveOneOut:
    """
    Leave-One-Out cross-validation iterator.

    Each sample is used once as test while all others are training.
    """

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator:
        """
        Generate indices for leave-one-out split.
        """
        n_samples = len(X)
        for i in range(n_samples):
            test_indices = np.array([i])
            train_indices = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])
            yield train_indices, test_indices

    def get_n_splits(self, X: np.ndarray) -> int:
        """Return number of splits (equals number of samples)."""
        return len(X)


class TimeSeriesSplit:
    """
    Time Series cross-validation iterator.

    Provides train/test indices for time series data, where test set
    is always in the future relative to training set.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Parameters:
        -----------
        n_splits : int
            Number of splits
        gap : int
            Number of samples to exclude between train and test
        """
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator:
        """
        Generate indices for time series split.
        """
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                test_end = n_samples

            train_indices = np.arange(0, test_start - self.gap)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


def cross_val_score(model, X: np.ndarray, y: np.ndarray,
                   cv: int = 5, scoring: str = 'accuracy',
                   shuffle: bool = True, random_state: Optional[int] = None) -> np.ndarray:
    """
    Evaluate model using cross-validation.

    Parameters:
    -----------
    model : object
        Model with fit and predict (or predict_proba) methods
    X : np.ndarray
        Features
    y : np.ndarray
        Target labels
    cv : int
        Number of folds
    scoring : str
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'rmse', 'mae', 'r2'
    shuffle : bool
        Whether to shuffle data
    random_state : int
        Random seed

    Returns:
    --------
    scores : np.ndarray
        Array of scores for each fold
    """
    # Determine if classification or regression
    is_classification = scoring in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Create appropriate splitter
    if is_classification:
        kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    else:
        kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

    scores = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone model (simple approach - create new instance)
        model_copy = model.__class__(**{
            k: v for k, v in model.__dict__.items()
            if not k.endswith('_') and not callable(v)
        })

        # Train
        model_copy.fit(X_train, y_train)

        # Score
        if scoring == 'accuracy':
            pred = model_copy.predict(X_test)
            score = np.mean(pred == y_test)
        elif scoring == 'precision':
            pred = model_copy.predict(X_test)
            tp = np.sum((y_test == 1) & (pred == 1))
            fp = np.sum((y_test == 0) & (pred == 1))
            score = tp / (tp + fp) if (tp + fp) > 0 else 0
        elif scoring == 'recall':
            pred = model_copy.predict(X_test)
            tp = np.sum((y_test == 1) & (pred == 1))
            fn = np.sum((y_test == 1) & (pred == 0))
            score = tp / (tp + fn) if (tp + fn) > 0 else 0
        elif scoring == 'f1':
            pred = model_copy.predict(X_test)
            tp = np.sum((y_test == 1) & (pred == 1))
            fp = np.sum((y_test == 0) & (pred == 1))
            fn = np.sum((y_test == 1) & (pred == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif scoring == 'roc_auc':
            if hasattr(model_copy, 'predict_proba'):
                proba = model_copy.predict_proba(X_test)
            else:
                proba = model_copy.predict(X_test)
            # Simple AUC calculation
            from .evaluation_metrics import ClassificationMetrics
            _, _, score = ClassificationMetrics.roc_auc(y_test, proba)
        elif scoring == 'mse':
            pred = model_copy.predict(X_test)
            score = -np.mean((y_test - pred) ** 2)  # Negative for consistency
        elif scoring == 'rmse':
            pred = model_copy.predict(X_test)
            score = -np.sqrt(np.mean((y_test - pred) ** 2))
        elif scoring == 'mae':
            pred = model_copy.predict(X_test)
            score = -np.mean(np.abs(y_test - pred))
        elif scoring == 'r2':
            pred = model_copy.predict(X_test)
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")

        scores.append(score)

    return np.array(scores)


def cross_val_predict(model, X: np.ndarray, y: np.ndarray,
                     cv: int = 5, method: str = 'predict') -> np.ndarray:
    """
    Generate cross-validated predictions.

    Parameters:
    -----------
    model : object
        Model with fit and predict methods
    X : np.ndarray
        Features
    y : np.ndarray
        Target labels
    cv : int
        Number of folds
    method : str
        'predict' or 'predict_proba'

    Returns:
    --------
    predictions : np.ndarray
        Cross-validated predictions for each sample
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    predictions = np.zeros(len(y))

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Clone and fit model
        model_copy = model.__class__(**{
            k: v for k, v in model.__dict__.items()
            if not k.endswith('_') and not callable(v)
        })
        model_copy.fit(X_train, y_train)

        # Predict
        if method == 'predict_proba' and hasattr(model_copy, 'predict_proba'):
            predictions[test_idx] = model_copy.predict_proba(X_test)
        else:
            predictions[test_idx] = model_copy.predict(X_test)

    return predictions


class GridSearchCV:
    """
    Exhaustive search over specified parameter values.
    """

    def __init__(self, model, param_grid: dict, cv: int = 5,
                 scoring: str = 'accuracy', verbose: int = 0):
        """
        Parameters:
        -----------
        model : object
            Model to tune
        param_grid : dict
            Dictionary with parameter names as keys and lists of values
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric
        verbose : int
            Verbosity level
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GridSearchCV':
        """
        Run fit with all sets of parameters.
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = list(product(*param_values))

        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }

        best_score = -np.inf
        best_params = None
        best_model = None

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            if self.verbose > 0:
                print(f"Testing params {i+1}/{len(all_combinations)}: {params}")

            # Create model with these parameters
            model_copy = self.model.__class__(**params)

            # Cross-validate
            scores = cross_val_score(model_copy, X, y,
                                    cv=self.cv, scoring=self.scoring)

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                # Fit on full data
                best_model = self.model.__class__(**params)
                best_model.fit(X, y)

            if self.verbose > 0:
                print(f"  Score: {mean_score:.4f} (+/- {std_score:.4f})")

        # Calculate ranks
        scores_array = np.array(results['mean_test_score'])
        results['rank_test_score'] = np.argsort(np.argsort(-scores_array)) + 1

        self.cv_results_ = results
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator."""
        return self.best_estimator_.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score using the best estimator."""
        return self.best_estimator_.score(X, y)


def learning_curve(model, X: np.ndarray, y: np.ndarray,
                  train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                  cv: int = 5, scoring: str = 'accuracy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate learning curve data.

    Parameters:
    -----------
    model : object
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    train_sizes : np.ndarray
        Relative or absolute sizes of training set
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns:
    --------
    train_sizes_abs : np.ndarray
        Absolute sizes of training sets
    train_scores : np.ndarray
        Scores on training sets
    test_scores : np.ndarray
        Scores on test sets
    """
    n_samples = len(X)

    # Convert relative to absolute sizes if necessary
    if train_sizes.max() <= 1:
        train_sizes_abs = (train_sizes * n_samples).astype(int)
    else:
        train_sizes_abs = train_sizes.astype(int)

    train_scores_mean = []
    test_scores_mean = []

    for size in train_sizes_abs:
        # Use subset of data
        X_subset = X[:size]
        y_subset = y[:size]

        scores = cross_val_score(model, X_subset, y_subset,
                                cv=min(cv, size), scoring=scoring)
        test_scores_mean.append(np.mean(scores))

        # Train score on full subset
        model_copy = model.__class__(**{
            k: v for k, v in model.__dict__.items()
            if not k.endswith('_') and not callable(v)
        })
        model_copy.fit(X_subset, y_subset)
        train_pred = model_copy.predict(X_subset)

        if scoring == 'accuracy':
            train_scores_mean.append(np.mean(train_pred == y_subset))
        elif scoring in ['mse', 'rmse', 'mae', 'r2']:
            if scoring == 'mse':
                train_scores_mean.append(-np.mean((y_subset - train_pred) ** 2))
            elif scoring == 'rmse':
                train_scores_mean.append(-np.sqrt(np.mean((y_subset - train_pred) ** 2)))
            elif scoring == 'mae':
                train_scores_mean.append(-np.mean(np.abs(y_subset - train_pred)))
            elif scoring == 'r2':
                ss_res = np.sum((y_subset - train_pred) ** 2)
                ss_tot = np.sum((y_subset - np.mean(y_subset)) ** 2)
                train_scores_mean.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

    return train_sizes_abs, np.array(train_scores_mean), np.array(test_scores_mean)


def validation_curve(model, X: np.ndarray, y: np.ndarray,
                    param_name: str, param_range: np.ndarray,
                    cv: int = 5, scoring: str = 'accuracy') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate validation curve data for a specific hyperparameter.

    Parameters:
    -----------
    model : object
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    param_name : str
        Name of hyperparameter to vary
    param_range : np.ndarray
        Values of the hyperparameter
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns:
    --------
    train_scores : np.ndarray
        Scores on training sets for each parameter value
    test_scores : np.ndarray
        Scores on test sets for each parameter value
    """
    train_scores = []
    test_scores = []

    base_params = {
        k: v for k, v in model.__dict__.items()
        if not k.endswith('_') and not callable(v)
    }

    for param_value in param_range:
        params = base_params.copy()
        params[param_name] = param_value

        model_copy = model.__class__(**params)
        scores = cross_val_score(model_copy, X, y, cv=cv, scoring=scoring)

        # Train score
        model_copy.fit(X, y)
        train_pred = model_copy.predict(X)
        if scoring == 'accuracy':
            train_score = np.mean(train_pred == y)
        else:
            train_score = np.mean(scores)  # Approximation

        train_scores.append(train_score)
        test_scores.append(np.mean(scores))

    return np.array(train_scores), np.array(test_scores)
