import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under ROC Curve.

    Args:
        y_true: (N,) binary labels (0=real, 1=fake)
        y_score: (N,) predicted probabilities for class 1 (fake)

    Returns:
        AUC value
    """
    return float(roc_auc_score(y_true, y_score))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Equal Error Rate.

    EER is the point where false positive rate equals false negative rate.

    Args:
        y_true: (N,) binary labels (0=real, 1=fake)
        y_score: (N,) predicted probabilities for class 1 (fake)

    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    return eer
