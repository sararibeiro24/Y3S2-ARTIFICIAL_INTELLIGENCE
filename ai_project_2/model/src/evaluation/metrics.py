from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Report:
    pr_auc: float
    average_precision: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    recall_at_p90: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return asdict(self)


def recall_at_precision(y_true, scores, target_precision: float = 0.90) -> float:
    # In a fraud-review workflow analyst capacity is the real constraint;
    # "what fraction of fraud do we catch while keeping precision above X"
    # is the metric the business cares about, more than F1.
    p, r, _ = precision_recall_curve(y_true, scores)
    mask = p >= target_precision
    if not mask.any():
        return 0.0
    return float(r[mask].max())


def evaluate(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> Report:
    y_true = np.asarray(y_true)
    y_pred = (scores >= threshold).astype(int)
    p, r, _ = precision_recall_curve(y_true, scores)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Report(
        pr_auc=float(np.trapezoid(np.flip(p), np.flip(r))),
        average_precision=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        recall_at_p90=recall_at_precision(y_true, scores, 0.90),
        threshold=float(threshold),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
