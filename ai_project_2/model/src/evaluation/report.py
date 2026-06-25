from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.src.config import METRICS_DIR
from model.src.evaluation.metrics import Report


def save_metrics_json(reports: dict[str, Report], path: Path | None = None) -> Path:
    path = path or (METRICS_DIR / "evaluation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: rep.to_dict() for name, rep in reports.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def comparison_table(reports: dict[str, Report]) -> pd.DataFrame:
    rows = []
    for name, rep in reports.items():
        d = rep.to_dict()
        d["model"] = name
        rows.append(d)
    cols = [
        "model", "pr_auc", "average_precision", "roc_auc",
        "precision", "recall", "f1", "recall_at_p90",
        "threshold", "tp", "fp", "fn", "tn",
    ]
    return pd.DataFrame(rows)[cols]
