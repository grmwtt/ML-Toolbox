"""
Utils Package
=============
Utility modules for the ML Toolbox.
"""

from .evaluation_metrics import ClassificationMetrics, RegressionMetrics, ClusteringMetrics
from .preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
    LabelEncoder, OneHotEncoder, PolynomialFeatures,
    train_test_split, handle_missing_values, remove_outliers, balance_classes
)
from .cross_validation import (
    KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit,
    cross_val_score, cross_val_predict, GridSearchCV,
    learning_curve, validation_curve
)
from .visualization import MLVisualizer, DiagnosticPlots

__all__ = [
    'ClassificationMetrics', 'RegressionMetrics', 'ClusteringMetrics',
    'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer',
    'LabelEncoder', 'OneHotEncoder', 'PolynomialFeatures',
    'train_test_split', 'handle_missing_values', 'remove_outliers', 'balance_classes',
    'KFold', 'StratifiedKFold', 'LeaveOneOut', 'TimeSeriesSplit',
    'cross_val_score', 'cross_val_predict', 'GridSearchCV',
    'learning_curve', 'validation_curve',
    'MLVisualizer', 'DiagnosticPlots'
]
