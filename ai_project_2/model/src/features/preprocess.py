from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.src.config import (
    ENGINEERED_FLAGS,
    ENGINEERED_NUMERIC,
    NUMERIC_COLUMNS,
    TARGET,
    TYPE_COLUMN,
)

_CATEGORICAL = [TYPE_COLUMN]
_NUMERIC = NUMERIC_COLUMNS + ENGINEERED_NUMERIC
_FLAGS = ENGINEERED_FLAGS


def build_preprocessor() -> ColumnTransformer:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("type_ohe", ohe, _CATEGORICAL),
            ("scale", StandardScaler(), _NUMERIC),
            ("flags", "passthrough", _FLAGS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def feature_columns() -> list[str]:
    return _CATEGORICAL + _NUMERIC + _FLAGS


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    cols = [c for c in feature_columns() if c in data.columns]
    y = data[TARGET] if TARGET in data.columns else None
    return data[cols].copy(), y


def fit_preprocessor(labeled: pd.DataFrame, unlabeled: pd.DataFrame) -> ColumnTransformer:
    pre = build_preprocessor()
    X_lab, _ = split_xy(labeled)
    X_unl, _ = split_xy(unlabeled)
    pre.fit(pd.concat([X_lab, X_unl], axis=0, ignore_index=True))
    return pre


def transform(pre: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    X, _ = split_xy(data)
    return pre.transform(X)
