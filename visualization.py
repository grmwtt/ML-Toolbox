"""
Visualization Utilities Module
==============================
Common visualization functions for ML algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MLVisualizer:
    """
    Visualization utilities for machine learning models.
    """
    
    @staticmethod
    def setup_style():
        """
        Set up consistent plotting style.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    @staticmethod
    def plot_decision_boundary_2d(model, X: np.ndarray, y: np.ndarray, 
                                 title: str = "Decision Boundary",
                                 feature_names: Optional[List[str]] = None,
                                 h: float = 0.02) -> None:
        """
        Plot decision boundary for 2D data.
        
        Parameters:
        -----------
        model : object
            Model with predict or predict_proba method
        X : np.ndarray
            Feature matrix (n_samples, 2)
        y : np.ndarray
            Target labels
        title : str
            Plot title
        feature_names : List[str]
            Names of features for axis labels
        h : float
            Step size in mesh
        """
        if X.shape[1] != 2:
            raise ValueError("This function only works with 2D data")
        
        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predict on mesh
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            if len(Z.shape) > 1 and Z.shape[1] > 1:
                Z = Z[:, 1]  # Use probability of positive class
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Contour plot for decision boundary
        if hasattr(model, 'predict_proba'):
            contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu, levels=20)
            plt.colorbar(contour, label='Probability')
            # Add decision boundary line
            plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        else:
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Scatter plot for data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                            edgecolor='black', s=50, alpha=0.8)
        
        # Labels
        if feature_names:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        
        plt.title(title)
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                           train_sizes: Optional[np.ndarray] = None,
                           title: str = "Learning Curves",
                           ylabel: str = "Score") -> None:
        """
        Plot learning curves to diagnose bias/variance.
        
        Parameters:
        -----------
        train_scores : List[float]
            Training scores for each training size
        val_scores : List[float]
            Validation scores for each training size
        train_sizes : np.ndarray
            Actual training set sizes (optional)
        title : str
            Plot title
        ylabel : str
            Y-axis label
        """
        plt.figure(figsize=(10, 6))
        
        if train_sizes is None:
            train_sizes = range(1, len(train_scores) + 1)
        
        plt.plot(train_sizes, train_scores, 'o-', color='blue', 
                label='Training Score', linewidth=2, markersize=8)
        plt.plot(train_sizes, val_scores, 'o-', color='red',
                label='Validation Score', linewidth=2, markersize=8)
        
        plt.xlabel('Training Set Size')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add shaded regions for interpretation
        plt.fill_between(train_sizes, train_scores, val_scores, 
                        alpha=0.1, color='gray')
        
        # Add interpretation text
        gap = np.mean(np.array(train_scores) - np.array(val_scores))
        if gap > 0.1:
            plt.text(0.02, 0.02, 'High Variance (Overfitting)', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        elif np.mean(val_scores) < 0.7:
            plt.text(0.02, 0.02, 'High Bias (Underfitting)', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        else:
            plt.text(0.02, 0.02, 'Good Fit', 
                    transform=plt.gca().transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            cmap: str = 'Blues') -> None:
        """
        Plot confusion matrix heatmap.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        class_names : List[str]
            Names of classes
        title : str
            Plot title
        cmap : str
            Colormap
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                      title: str = "ROC Curve") -> None:
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        fpr : np.ndarray
            False positive rates
        tpr : np.ndarray
            True positive rates
        auc : float
            Area under curve
        title : str
            Plot title
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Add interpretation zones
        plt.fill_between([0, 1], [0, 1], 1, alpha=0.1, color='green')
        plt.text(0.6, 0.4, 'Better than\nRandom', fontsize=10, alpha=0.5)
        plt.text(0.4, 0.6, 'Worse than\nRandom', fontsize=10, alpha=0.5)
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importances: np.ndarray, 
                              feature_names: Optional[List[str]] = None,
                              title: str = "Feature Importance",
                              top_n: int = 20) -> None:
        """
        Plot feature importance bar chart.
        
        Parameters:
        -----------
        importances : np.ndarray
            Feature importance values
        feature_names : List[str]
            Names of features
        title : str
            Plot title
        top_n : int
            Number of top features to show
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Plot") -> None:
        """
        Plot residuals for regression diagnostics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        title : str
            Plot title
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Residuals vs Fitted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Residuals')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_clustering_results(X: np.ndarray, labels: np.ndarray,
                              centers: Optional[np.ndarray] = None,
                              title: str = "Clustering Results") -> None:
        """
        Plot clustering results for 2D data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, 2)
        labels : np.ndarray
            Cluster labels
        centers : np.ndarray
            Cluster centers (optional)
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            raise ValueError("This function only works with 2D data")
        
        plt.figure(figsize=(10, 8))
        
        # Plot points
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1], c=[color], 
                       label=f'Cluster {label}', s=50, alpha=0.7)
        
        # Plot centers if provided
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='black', 
                       marker='x', s=200, linewidths=3, label='Centers')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def create_interactive_3d_plot(X: np.ndarray, y: np.ndarray,
                                  title: str = "3D Visualization") -> go.Figure:
        """
        Create interactive 3D scatter plot using plotly.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, 3)
        y : np.ndarray
            Labels or values for coloring
        title : str
            Plot title
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Interactive 3D figure
        """
        if X.shape[1] != 3:
            raise ValueError("This function requires 3D data")
        
        fig = go.Figure(data=[go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Value")
            ),
            text=[f"Point {i}<br>Value: {y[i]:.2f}" for i in range(len(y))],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Feature 3'
            ),
            width=800,
            height=600
        )
        
        return fig


class DiagnosticPlots:
    """
    Diagnostic plots for model evaluation and comparison.
    """
    
    @staticmethod
    def plot_model_comparison(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            metric_func: callable, metric_name: str = "Score") -> None:
        """
        Compare multiple models on train and test sets.
        
        Parameters:
        -----------
        models : dict
            Dictionary of {model_name: model_object}
        X_train, y_train : Training data
        X_test, y_test : Test data
        metric_func : callable
            Function to compute metric (e.g., accuracy_score)
        metric_name : str
            Name of the metric for labeling
        """
        model_names = list(models.keys())
        train_scores = []
        test_scores = []
        
        for name, model in models.items():
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_scores.append(metric_func(y_train, train_pred))
            test_scores.append(metric_func(y_test, test_pred))
        
        # Create bar plot
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Model Comparison - {metric_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_hyperparameter_tuning(param_values: List, train_scores: List[float],
                                  val_scores: List[float], param_name: str,
                                  title: str = "Hyperparameter Tuning") -> None:
        """
        Plot validation curves for hyperparameter tuning.
        
        Parameters:
        -----------
        param_values : List
            Values of the hyperparameter
        train_scores : List[float]
            Training scores for each parameter value
        val_scores : List[float]
            Validation scores for each parameter value
        param_name : str
            Name of the hyperparameter
        title : str
            Plot title
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(param_values, train_scores, 'o-', label='Training Score', linewidth=2)
        plt.plot(param_values, val_scores, 'o-', label='Validation Score', linewidth=2)
        
        # Mark best validation score
        best_idx = np.argmax(val_scores)
        plt.scatter(param_values[best_idx], val_scores[best_idx], 
                   color='red', s=100, zorder=5, 
                   label=f'Best: {param_values[best_idx]}')
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Use log scale if values span multiple orders of magnitude
        if max(param_values) / min(param_values) > 100:
            plt.xscale('log')
        
        plt.show()
