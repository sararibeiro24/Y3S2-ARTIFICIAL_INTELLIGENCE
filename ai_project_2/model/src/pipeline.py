from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.src.config import RAW_FRAUD_CSV, TARGET
from model.src.data.clean import clean
from model.src.data.encode import encode_transaction_type
from model.src.data.load import load_data
from model.src.data.split import Splits, load_splits, make_splits, save_splits
from model.src.evaluation.explain import gain_importance, shap_summary
from model.src.evaluation.metrics import Report, evaluate
from model.src.evaluation.report import comparison_table, save_metrics_json
from model.src.features.preprocess import fit_preprocessor, split_xy, transform
from model.src.training.model import fit_supervised
from model.src.training.model_io import save


@dataclass
class TrainedArtifacts:
    preprocessor: Any
    model: Any
    threshold: float
    feature_names: list[str]


def prepare() -> Splits:
    df = load_data(RAW_FRAUD_CSV)
    df = clean(df)
    df = encode_transaction_type(df)
    splits = make_splits(df)
    save_splits(splits)
    print(
        f"splits: test={len(splits.test):,} val={len(splits.val):,} "
        f"labeled={len(splits.labeled_train):,} unlabeled={len(splits.unlabeled_train):,}"
    )
    print(f"val fraud rate:  {splits.val[TARGET].mean():.5f}")
    print(f"test fraud rate: {splits.test[TARGET].mean():.5f}")
    return splits


def train(splits: Splits | None = None) -> TrainedArtifacts:
    splits = splits or load_splits()

    pre = fit_preprocessor(splits.labeled_train, splits.unlabeled_train)
    _, y_lab = split_xy(splits.labeled_train)

    X_lab = transform(pre, splits.labeled_train)
    model = fit_supervised(X_lab, y_lab)

    art = TrainedArtifacts(
        preprocessor=pre,
        model=model,
        threshold=0.5,
        feature_names=list(X_lab.columns),
    )
    save(art, "model.joblib")
    return art


def evaluate_all(art: TrainedArtifacts, splits: Splits | None = None) -> dict[str, Report]:
    splits = splits or load_splits()
    _, y_test = split_xy(splits.test)

    X_test = transform(art.preprocessor, splits.test)
    scores = art.model.predict_proba(X_test)[:, 1]
    reports = {"xgb": evaluate(y_test, scores, threshold=art.threshold)}

    table = comparison_table(reports)
    print(table.to_string(index=False))

    save_metrics_json(reports)

    try:
        shap_summary(art.model, X_test)
    except Exception as e:
        print(f"shap failed ({e}); writing gain-based importance instead")
    gi = gain_importance(art.model, art.feature_names)
    print("top features by gain:")
    print(gi.head(10).to_string())

    return reports


def run_all() -> dict[str, Report]:
    splits = prepare()
    art = train(splits)
    return evaluate_all(art, splits)
