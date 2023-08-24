from __future__ import annotations

from typing import Any
from typing import cast
from typing import Tuple

import pandas as pd

# ===================
# TIME SERIES
# ===================


def average(df: pd.DataFrame) -> pd.DataFrame:
    return df - df.mean()


def detect_time_step(df: pd.DataFrame) -> pd.Timedelta:
    """
    return the median time step of a dataframe
    """
    ts = cast(pd.Timedelta, df.index.to_series().diff().median())
    if pd.isna(ts):
        msg = "Couldn't detect time step!"
        raise ValueError(msg)
    else:
        return ts


def calculate_span(df: pd.DataFrame) -> Any:
    return df.index.to_series().diff().sum()


def calculate_completeness(df: pd.DataFrame) -> Any:
    """
    return the completeness of a dataframe in %
    """
    data_avail_ratio = 1 - df.resample("60min").mean().isna().sum() / len(df.resample("60min").mean())
    return data_avail_ratio * 100


# ======================
# DATA SANITY FUNCTIONS
# ======================
def correct_unit(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    flag = False
    for i in range(int(len(df) / 1000)):
        if df.iloc[i * 1000 : (i + 1) * 1000].std() > 50:
            # most probably signal is in mm
            df.iloc[i * 1000 : (i + 1) * 1000] = df.iloc[i * 1000 : (i + 1) * 1000] / 1000
            flag = True
        elif df.iloc[i * 1000 : (i + 1) * 1000].std() > 5:
            # most probably signal is in cm
            df.iloc[i * 1000 : (i + 1) * 1000] = df.iloc[i * 1000 : (i + 1) * 1000] / 100
            flag = True
    return df, flag


def detect_gaps(
    df: pd.DataFrame,
) -> Tuple[pd.Series[pd.Timedelta], pd.Series[pd.Timedelta], pd.Series[pd.Timedelta]]:
    # Take the diff of the first column (drop 1st row since it's undefined)
    deltas = df.index.to_series().diff()[1:]
    gaps = deltas[(deltas > pd.Timedelta(10, "min"))]
    small_gaps = gaps[gaps < pd.Timedelta(1, "hours")]
    big_gaps = gaps[gaps > pd.Timedelta(1, "hours")]
    return gaps, small_gaps, big_gaps
