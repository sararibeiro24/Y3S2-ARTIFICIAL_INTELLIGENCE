import pandas as pd
from pathlib import Path


def load_data(file_path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
