from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, Union, List

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ArrayLike = Union[np.ndarray, List[int], List[str]]


def run_stratified_kfold(
    model,
    X,
    y,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: Iterable[str] = ("accuracy", "f1_weighted"),
    n_jobs: Optional[int] = -1,
    return_train_score: bool = False,
) -> Dict:

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=skf,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
    )

    # Summarize metrics (keys look like 'test_accuracy', 'test_f1_weighted', ...)
    means = {}
    stds = {}
    for key, vals in results.items():
        if key.startswith("test_") or (return_train_score and key.startswith("train_")):
            means[key] = float(np.mean(vals))
            stds[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return {
        "raw": results,      # full per-fold arrays
        "means": means,      # average per metric
        "stds": stds,        # std dev per metric
        "cv": skf,
    }


def oof_predictions_and_metrics(
    model,
    X,
    y,
    *,
    average: str = "weighted",
    labels: Optional[Iterable] = None,
) -> Dict:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mdl = clone(model)
    y_pred_oof = cross_val_predict(mdl, X, y, cv=skf, n_jobs=-1, method="predict")

    labels_list = list(labels) if labels is not None else list(np.unique(y))
    acc = accuracy_score(y, y_pred_oof)
    f1 = f1_score(y, y_pred_oof, average=average)
    cm = confusion_matrix(y, y_pred_oof, labels=labels_list)

    return {
        "y_pred_oof": y_pred_oof,
        "labels": labels_list,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "summary": {"accuracy": acc, f1" + f"_{average}": f1},
    }
