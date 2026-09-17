import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.

    y_pred : array-like
        Binary predictions.

    y_prob : array-like
        Predicted probability for the positive class.

    Returns
    -------
    dict
        Classification metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
    }

    return metrics


def calculate_confusion_matrix(
    y_true,
    y_pred,
):
    """
    Calculate binary confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred,
    )