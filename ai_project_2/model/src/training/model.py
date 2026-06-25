from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from model.src.config import XGB_PARAMS


def class_weight(y: pd.Series | np.ndarray) -> float:
    # scale_pos_weight = N_negative / N_positive.
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    return neg / pos


def fit_supervised(X: pd.DataFrame, y: pd.Series, **overrides) -> XGBClassifier:
    params = {**XGB_PARAMS, "scale_pos_weight": class_weight(y), **overrides}
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model
