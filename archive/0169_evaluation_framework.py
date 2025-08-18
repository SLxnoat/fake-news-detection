from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Iterable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ArrayLike = Union[np.ndarray, List[int], List[str]]

@dataclass
class EvaluationResult:
    accuracy: float
    f1: float
    confusion_matrix: np.ndarray
    labels: List[Union[int, str]]
    report_text: Optional[str] = None
    auc: Optional[float] = None  # For binary or multiclass probability models
    summary: Optional[Dict[str, float]] = None


def _ensure_labels(y_true: ArrayLike, labels: Optional[Iterable] = None) -> List:
    if labels is not None:
        return list(labels)
    # consistent order for confusion matrix
    return list(np.unique(y_true))


def evaluate_model(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
    return_report: bool = True,
    y_proba: Optional[np.ndarray] = None,  # shape (n_samples,) for binary or (n_samples, n_classes)
    multi_class_auc_strategy: str = "ovr",
) -> EvaluationResult:

    
    labels_list = _ensure_labels(y_true, labels)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)

    report_txt = classification_report(y_true, y_pred, labels=labels_list) if return_report else None

    auc = None
    if y_proba is not None:
        # Handle AUC for binary or multiclass
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                # Binary: y_proba is prob of positive class
                # We must map y_true to {0,1}. Try to infer positive class as the max label if numeric,
                # otherwise last in labels_list.
                # Convert to 0/1 using label order in labels_list
                if len(labels_list) != 2:
                    raise ValueError("Binary AUC requested but number of classes != 2.")
                positive_class = labels_list[-1]
                y_true_bin = np.array([1 if y == positive_class else 0 for y in y_true])
                proba_pos = y_proba if y_proba.ndim == 1 else y_proba.ravel()
                auc = roc_auc_score(y_true_bin, proba_pos)
            else:
                # Multiclass AUC
                auc = roc_auc_score(y_true, y_proba, multi_class=multi_class_auc_strategy)
        except Exception:
            # If it fails (e.g., single class in y_true), just skip AUC
            auc = None

