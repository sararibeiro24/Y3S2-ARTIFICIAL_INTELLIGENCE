from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from model.src.config import MODELS_DIR


def save(obj: Any, name: str, out_dir: Path = MODELS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    joblib.dump(obj, path)
    return path


def load(name: str, in_dir: Path = MODELS_DIR) -> Any:
    return joblib.load(in_dir / name)
