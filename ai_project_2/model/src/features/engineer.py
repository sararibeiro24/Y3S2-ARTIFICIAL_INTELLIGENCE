import numpy as np
import pandas as pd

from model.src.config import TYPE_ENCODING

# PaySim `step` is hours since simulation start, so it decomposes naturally
# into hour-of-day and day-of-week.
_HOURS_PER_DAY = 24
_HIGH_AMOUNT_QUANTILE = 0.99

_TRANSFER_CODE = TYPE_ENCODING["TRANSFER"]
_CASH_OUT_CODE = TYPE_ENCODING["CASH_OUT"]


def add_balance_errors(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["errorBalanceOrig"] = (
        data["newbalanceOrig"] + data["amount"] - data["oldbalanceOrg"]
    )
    data["errorBalanceDest"] = (
        data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
    )
    return data


def add_zero_balance_flags(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["orig_zero_after"] = (data["newbalanceOrig"] == 0).astype("int8")
    data["dest_zero_before"] = (data["oldbalanceDest"] == 0).astype("int8")
    data["dest_zero_after"] = (data["newbalanceDest"] == 0).astype("int8")
    return data


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    step = data["step"].astype("int64")
    data["hour_of_day"] = (step % _HOURS_PER_DAY).astype("int16")
    data["day_of_week"] = ((step // _HOURS_PER_DAY) % 7).astype("int16")
    return data


def add_amount_features(
    data: pd.DataFrame, high_threshold: float | None = None
) -> pd.DataFrame:
    data = data.copy()
    data["amount_log"] = np.log1p(data["amount"])
    threshold = (
        high_threshold
        if high_threshold is not None
        else data["amount"].quantile(_HIGH_AMOUNT_QUANTILE)
    )
    data["is_high_amount"] = (data["amount"] >= threshold).astype("int8")
    return data


def add_type_flags(data: pd.DataFrame) -> pd.DataFrame:
    # In the source PaySim paper fraud only occurs in TRANSFER and CASH_OUT.
    data = data.copy()
    data["is_transfer_or_cashout"] = (
        data["type"].isin([_TRANSFER_CODE, _CASH_OUT_CODE]).astype("int8")
    )
    return data


def engineer(data: pd.DataFrame) -> pd.DataFrame:
    data = add_balance_errors(data)
    data = add_zero_balance_flags(data)
    data = add_time_features(data)
    data = add_amount_features(data)
    data = add_type_flags(data)
    return data
