from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model.src.config import (
    LABELED_PER_CLASS,
    NUMERIC_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET,
    TEST_FRACTION,
    USE_TEMPORAL_SPLIT,
    VAL_FRACTION,
)
from model.src.features.engineer import engineer


@dataclass
class Splits:
    test: pd.DataFrame
    val: pd.DataFrame
    labeled_train: pd.DataFrame
    unlabeled_train: pd.DataFrame


def _scrub_numeric(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=NUMERIC_COLUMNS)


def _temporal_cutoffs(data: pd.DataFrame, val_frac: float, test_frac: float) -> tuple[int, int]:
    test_cut = int(np.quantile(data["step"], 1.0 - test_frac))
    val_cut = int(np.quantile(data["step"], 1.0 - test_frac - val_frac))
    return val_cut, test_cut


def _stratified_random_split(
    data: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split as sk_split

    train_pool, test = sk_split(
        data, test_size=test_frac, stratify=data[TARGET], random_state=seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    train_pool, val = sk_split(
        train_pool,
        test_size=rel_val,
        stratify=train_pool[TARGET],
        random_state=seed,
    )
    return train_pool, val, test


def _temporal_split(
    data: pd.DataFrame, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_cut, test_cut = _temporal_cutoffs(data, val_frac, test_frac)
    train_pool = data[data["step"] <= val_cut]
    val = data[(data["step"] > val_cut) & (data["step"] <= test_cut)]
    test = data[data["step"] > test_cut]
    return train_pool, val, test


def _sample_labeled(
    train_pool: pd.DataFrame, labeled_per_class: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraud = train_pool[train_pool[TARGET] == 1]
    nonfraud = train_pool[train_pool[TARGET] == 0]
    n_fraud = min(labeled_per_class, len(fraud))
    n_nonfraud = min(labeled_per_class, len(nonfraud))
    fraud_lab = fraud.sample(n=n_fraud, random_state=seed)
    nonfraud_lab = nonfraud.sample(n=n_nonfraud, random_state=seed)
    labeled = (
        pd.concat([fraud_lab, nonfraud_lab])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    remaining = train_pool.drop(pd.concat([fraud_lab, nonfraud_lab]).index)
    return labeled, remaining


def make_splits(
    data: pd.DataFrame,
    test_frac: float = TEST_FRACTION,
    val_frac: float = VAL_FRACTION,
    labeled_per_class: int = LABELED_PER_CLASS,
    seed: int = RANDOM_SEED,
    temporal: bool = USE_TEMPORAL_SPLIT,
) -> Splits:
    if temporal:
        train_pool, val, test = _temporal_split(data, val_frac, test_frac)
    else:
        train_pool, val, test = _stratified_random_split(data, val_frac, test_frac, seed)

    labeled, remaining = _sample_labeled(train_pool, labeled_per_class, seed)

    unlabeled = remaining.drop(columns=[TARGET]).reset_index(drop=True)

    test = engineer(_scrub_numeric(test).reset_index(drop=True))
    val = engineer(_scrub_numeric(val).reset_index(drop=True))
    labeled = engineer(_scrub_numeric(labeled))
    unlabeled = engineer(_scrub_numeric(unlabeled))

    return Splits(test=test, val=val, labeled_train=labeled, unlabeled_train=unlabeled)


def save_splits(splits: Splits, out_dir: Path = PROCESSED_DATA_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "test": out_dir / "test.parquet",
        "val": out_dir / "val.parquet",
        "labeled_train": out_dir / "labeled_train.parquet",
        "unlabeled_train": out_dir / "unlabeled_train.parquet",
    }
    splits.test.to_parquet(paths["test"], index=False)
    splits.val.to_parquet(paths["val"], index=False)
    splits.labeled_train.to_parquet(paths["labeled_train"], index=False)
    splits.unlabeled_train.to_parquet(paths["unlabeled_train"], index=False)
    return paths


def load_splits(in_dir: Path = PROCESSED_DATA_DIR) -> Splits:
    return Splits(
        test=pd.read_parquet(in_dir / "test.parquet"),
        val=pd.read_parquet(in_dir / "val.parquet"),
        labeled_train=pd.read_parquet(in_dir / "labeled_train.parquet"),
        unlabeled_train=pd.read_parquet(in_dir / "unlabeled_train.parquet"),
    )
