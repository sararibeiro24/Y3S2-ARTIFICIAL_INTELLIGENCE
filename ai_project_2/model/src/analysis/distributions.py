from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea

from matplotlib.ticker import FuncFormatter

from model.src.config import BOXPLOT_COLUMNS, FIGURES_DIR, TARGET, TYPE_ENCODING


def _save(fig_path: Path) -> Path:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    return fig_path


def _type_labels() -> list[str]:
    return [name for name, _ in sorted(TYPE_ENCODING.items(), key=lambda kv: kv[1])]


def plot_class_balance(data: pd.DataFrame, target: str = "isFraud") -> Path:
    counts = data[target].value_counts().sort_index()
    ax = sea.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), palette="viridis", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.title("Class Balance (isFraud)")
    plt.xlabel("isFraud")
    plt.ylabel("count")
    plt.yscale("log")
    return _save(FIGURES_DIR / "class_balance.png")


def plot_type_counts(data: pd.DataFrame) -> Path:
    counts = data["type"].value_counts().sort_index()
    labels = _type_labels()
    sea.barplot(x=[labels[i] for i in counts.index], y=counts.values, hue=[labels[i] for i in counts.index], palette="mako", legend=False)
    plt.title("Transaction Type Counts")
    plt.xlabel("type")
    plt.ylabel("count")
    plt.xticks(rotation=20)
    return _save(FIGURES_DIR / "type_counts.png")


def plot_fraud_rate_by_type(data: pd.DataFrame) -> Path:
    rates = data.groupby("type")["isFraud"].mean().sort_index()
    labels = _type_labels()
    ax = sea.barplot(x=[labels[i] for i in rates.index], y=rates.values, hue=[labels[i] for i in rates.index], palette="rocket", legend=False)
    for i, v in enumerate(rates.values):
        ax.text(i, v, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("type")
    plt.ylabel("fraud rate")
    plt.xticks(rotation=20)
    return _save(FIGURES_DIR / "fraud_rate_by_type.png")


def plot_amount_by_class(data: pd.DataFrame) -> Path:
    df = data[["isFraud", "amount"]].copy()
    df["log_amount"] = np.log1p(df["amount"])
    sea.boxplot(data=df, x="isFraud", y="log_amount", hue="isFraud", palette="Set2", legend=False)
    plt.title("Amount Distribution by Class (log1p)")
    plt.xlabel("isFraud")
    plt.ylabel("log1p(amount)")
    return _save(FIGURES_DIR / "amount_by_class.png")

def plot_labeled_feature_boxplots(data: pd.DataFrame, prefix: str = "labeled") -> list[Path]:
    sea.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
    paths: list[Path] = []
    for col in BOXPLOT_COLUMNS:
        ax = sea.boxplot(data=data, x=TARGET, y=col, hue=TARGET, palette="Set2", showfliers=False, legend=False)
        ax.set_title(f"normal {col} vs Fraud", fontsize=13, fontweight="bold")
        ax.set_xlabel("Is Fraud?")
        ax.set_ylabel(col)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        paths.append(_save(FIGURES_DIR / f"{prefix}_{col}_box.png"))
    return paths


def plot_correlation(data: pd.DataFrame) -> Path:
    sea.heatmap(data.corr(), fmt=".2f", annot=True, cmap="coolwarm")
    plt.subplots_adjust(left=0.1)
    plt.title("Correlation Plot")
    return _save(FIGURES_DIR / "correlation_plot.png")


def run_all(data: pd.DataFrame) -> list[Path]:
    return [
        plot_class_balance(data),
        plot_type_counts(data),
        plot_fraud_rate_by_type(data),
        plot_amount_by_class(data),
        plot_correlation(data),
    ]
