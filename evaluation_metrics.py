"""
Evaluation Metrics Module
==========================
Custom implementations of common ML evaluation metrics.
"""

import numpy as np
from typing import Optional, Union, List, Tuple


class ClassificationMetrics:
    """
    Classification metrics implemented from scratch.
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        n_classes: Optional[int] = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Returns:
            Matrix where element (i,j) is the number of samples with 
            true label i and predicted label j.
        """
        if n_classes is None:
            n_classes = max(np.max(y_true), np.max(y_pred)) + 1
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1
        
        return cm
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                           average: str = 'binary') -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Parameters:
        -----------
        average : str
            'binary' for binary classification
            'macro' for multiclass (average across classes)
            'micro' for multiclass (global calculation)
        """
        if average == 'binary':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
        
        elif average == 'macro':
            classes = np.unique(np.concatenate([y_true, y_pred]))
            precisions, recalls, f1s = [], [], []
            
            for c in classes:
                y_true_c = (y_true == c).astype(int)
                y_pred_c = (y_pred == c).astype(int)
                p, r, f = ClassificationMetrics.precision_recall_f1(y_true_c, y_pred_c, 'binary')
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            
            return np.mean(precisions), np.mean(recalls), np.mean(f1s)
        
        elif average == 'micro':
            cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
            tp = np.sum(np.diag(cm))
            fp = np.sum(cm) - tp
            fn = fp  # In micro averaging, FP and FN are equal
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate ROC curve and AUC score.
        
        Returns:
            fpr : False Positive Rates
            tpr : True Positive Rates
            auc : Area Under Curve
        """
        # Sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        # Get unique thresholds
        thresholds = np.unique(y_scores)
        
        # Calculate TPR and FPR for each threshold
        tpr_list = [0]
        fpr_list = [0]
        
        for threshold in thresholds[::-1]:
            y_pred = (y_scores >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        tpr_list.append(1)
        fpr_list.append(1)
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr_list, fpr_list)
        
        return np.array(fpr_list), np.array(tpr_list), auc
    
    @staticmethod
    def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                 epsilon: float = 1e-15) -> float:
        """
        Calculate logarithmic loss (cross-entropy).
        
        LogLoss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)]
        """
        # Clip probabilities to avoid log(0)
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        return -np.mean(y_true * np.log(y_pred_proba) + 
                       (1 - y_true) * np.log(1 - y_pred_proba))


class RegressionMetrics:
    """
    Regression metrics implemented from scratch.
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error.
        
        MSE = 1/n * Σ(y_true - y_pred)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        RMSE = √MSE
        """
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        MAE = 1/n * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Coefficient of Determination (R²).
        
        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = Σ(y_true - y_pred)²
            SS_tot = Σ(y_true - y_mean)²
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    @staticmethod
    def adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, 
                   n_features: int) -> float:
        """
        Adjusted R² score.
        
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
        where:
            n = number of samples
            p = number of features
        """
        n = len(y_true)
        r2 = RegressionMetrics.r2_score(y_true, y_pred)
        
        return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, 
            epsilon: float = 1e-15) -> float:
        """
        Mean Absolute Percentage Error.
        
        MAPE = 100/n * Σ|((y_true - y_pred) / y_true)|
        """
        # Avoid division by zero
        mask = np.abs(y_true) > epsilon
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


class ClusteringMetrics:
    """
    Clustering metrics implemented from scratch.
    """
    
    @staticmethod
    def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Silhouette Coefficient.
        
        Silhouette = (b - a) / max(a, b)
        where:
            a = mean intra-cluster distance
            b = mean nearest-cluster distance
        """
        n_samples = len(X)
        n_clusters = len(np.unique(labels))
        
        if n_clusters == 1:
            return 0
        
        silhouette_vals = []
        
        for i in range(n_samples):
            # Current sample's cluster
            curr_cluster = labels[i]
            
            # Calculate mean intra-cluster distance
            same_cluster_mask = labels == curr_cluster
            if np.sum(same_cluster_mask) > 1:
                a = np.mean(np.linalg.norm(X[same_cluster_mask] - X[i], axis=1))
            else:
                a = 0
            
            # Calculate mean nearest-cluster distance
            b = float('inf')
            for cluster in np.unique(labels):
                if cluster != curr_cluster:
                    other_cluster_mask = labels == cluster
                    mean_dist = np.mean(np.linalg.norm(X[other_cluster_mask] - X[i], axis=1))
                    b = min(b, mean_dist)
            
            # Calculate silhouette coefficient for this sample
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_vals.append(s)
        
        return np.mean(silhouette_vals)
    
    @staticmethod
    def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin Index.
        Lower values indicate better clustering.
        
        DB = 1/n * Σ max(R_ij)
        where R_ij = (s_i + s_j) / d_ij
        """
        n_clusters = len(np.unique(labels))
        cluster_centers = []
        cluster_dispersions = []
        
        # Calculate cluster centers and dispersions
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_points = X[cluster_mask]
            center = np.mean(cluster_points, axis=0)
            dispersion = np.mean(np.linalg.norm(cluster_points - center, axis=1))
            
            cluster_centers.append(center)
            cluster_dispersions.append(dispersion)
        
        # Calculate Davies-Bouldin index
        db_index = 0
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    d_ij = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    if d_ij > 0:
                        ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / d_ij
                        max_ratio = max(max_ratio, ratio)
            
            db_index += max_ratio
        
        return db_index / n_clusters if n_clusters > 0 else 0
    
    @staticmethod
    def calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz Index (Variance Ratio Criterion).
        Higher values indicate better clustering.
        
        CH = [B/(k-1)] / [W/(n-k)]
        where:
            B = between-cluster dispersion
            W = within-cluster dispersion
            k = number of clusters
            n = number of samples
        """
        n_samples = len(X)
        n_clusters = len(np.unique(labels))
        
        if n_clusters == 1:
            return 0
        
        # Overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Calculate between-cluster and within-cluster dispersion
        between_dispersion = 0
        within_dispersion = 0
        
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_points = X[cluster_mask]
            n_cluster_samples = len(cluster_points)
            
            if n_cluster_samples > 0:
                cluster_mean = np.mean(cluster_points, axis=0)
                
                # Between-cluster dispersion
                between_dispersion += n_cluster_samples * np.sum((cluster_mean - overall_mean) ** 2)
                
                # Within-cluster dispersion
                within_dispersion += np.sum((cluster_points - cluster_mean) ** 2)
        
        # Calculate index
        if within_dispersion == 0:
            return 0
        
        return (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: Optional[List[str]] = None) -> None:
    """
    Print a formatted classification report.
    """
    cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in classes]
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    
    for i, class_name in enumerate(class_names):
        y_true_c = (y_true == classes[i]).astype(int)
        y_pred_c = (y_pred == classes[i]).astype(int)
        precision, recall, f1 = ClassificationMetrics.precision_recall_f1(y_true_c, y_pred_c, 'binary')
        support = np.sum(y_true == classes[i])
        
        print(f"{class_name:<15} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10}")
    
    print("-"*60)
    
    # Overall metrics
    accuracy = ClassificationMetrics.accuracy(y_true, y_pred)
    macro_p, macro_r, macro_f1 = ClassificationMetrics.precision_recall_f1(y_true, y_pred, 'macro')
    
    print(f"\n{'Accuracy:':<20} {accuracy:.3f}")
    print(f"{'Macro Avg F1:':<20} {macro_f1:.3f}")
    print("="*60)
