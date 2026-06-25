from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.src.config import FIGURES_DIR


def shap_summary(model, X: pd.DataFrame, out_path: Path | None = None, sample: int = 5000) -> Path:
    import shap

    if len(X) > sample:
        X = X.sample(sample, random_state=0)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    out_path = out_path or (FIGURES_DIR / "shap_summary.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def gain_importance(model, feature_names: list[str]) -> pd.Series:
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")

    if all(k.startswith("f") and k[1:].isdigit() for k in raw):
        importances = {feature_names[int(k[1:])]: v for k, v in raw.items()}
    else:
        importances = raw
    return pd.Series(importances).sort_values(ascending=False)
