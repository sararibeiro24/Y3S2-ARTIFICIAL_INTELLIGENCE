import pandas as pd

from model.src.config import DROP_COLUMNS


def drop_missing(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates()


def drop_identifier_columns(data: pd.DataFrame, columns: list[str] = DROP_COLUMNS) -> pd.DataFrame:
    return data.drop(columns=columns)


def clean(data: pd.DataFrame) -> pd.DataFrame:
    data = drop_missing(data)
    data = drop_duplicates(data)
    data = drop_identifier_columns(data)
    return data
