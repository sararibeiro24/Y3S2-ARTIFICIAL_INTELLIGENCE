from __future__ import annotations

import argparse

from model.src.analysis.distributions import (
    plot_labeled_feature_boxplots,
    run_all as plot_distributions,
)
from model.src.config import RAW_FRAUD_CSV
from model.src.data.clean import clean
from model.src.data.encode import encode_transaction_type
from model.src.data.load import load_data
from model.src.data.split import load_splits
from model.src.pipeline import evaluate_all, prepare, run_all, train
from model.src.training.model_io import load as load_artifacts


def cmd_eda(_args: argparse.Namespace) -> None:
    df = load_data(RAW_FRAUD_CSV)
    df = clean(df)
    df = encode_transaction_type(df)
    for p in plot_distributions(df):
        print(f"saved {p}")
    try:
        splits = load_splits()
        rows = splits.labeled_train
    except FileNotFoundError:
        rows = df
    for p in plot_labeled_feature_boxplots(rows):
        print(f"saved {p}")


def cmd_prep(_args: argparse.Namespace) -> None:
    prepare()


def cmd_train(_args: argparse.Namespace) -> None:
    train()


def cmd_eval(_args: argparse.Namespace) -> None:
    art = load_artifacts("model.joblib")
    evaluate_all(art)


def cmd_all(_args: argparse.Namespace) -> None:
    run_all()


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud")
    sub = parser.add_subparsers(dest="cmd")
    for name, fn in [
        ("eda", cmd_eda),
        ("prep", cmd_prep),
        ("train", cmd_train),
        ("eval", cmd_eval),
        ("all", cmd_all),
    ]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)

    args = parser.parse_args()
    if args.cmd is None:
        cmd_all(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
