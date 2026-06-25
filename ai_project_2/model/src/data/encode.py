import pandas as pd

from model.src.config import TYPE_ENCODING


def encode_transaction_type(data: pd.DataFrame, mapping: dict[str, int] = TYPE_ENCODING) -> pd.DataFrame:
    data = data.copy()
    data["type"] = data["type"].replace(mapping).astype("int8")
    return data
