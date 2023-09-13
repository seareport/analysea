from __future__ import annotations

from typing import Any
from typing import cast
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags

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


def completeness(df: Union[pd.DataFrame, pd.Series[Any]]) -> Any:
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


def json_format(d: Dict[Any, Any]) -> Dict[Any, Any]:
    for key, value in d.items():
        if isinstance(value, dict):
            json_format(value)  # Recurse into nested dictionaries
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()  # Convert NumPy array to list
        elif isinstance(value, pd.Timestamp):
            d[key] = value.strftime("%Y-%m-%d %H:%M:%S")  # Convert pandas Timestamp to string
        elif isinstance(value, pd.Timedelta):
            d[key] = str(value)  # Convert pandas Timedelta to string
    return d


def nd_format(d: Dict[Any, Any]) -> Dict[Any, Any]:
    for key, value in d.items():
        if isinstance(value, dict):
            nd_format(value)  # Recurse into nested dictionaries
        elif isinstance(value, list):
            d[key] = np.array(value)  # Convert NumPy array to list
    return d


# Function to calculate correlation https://gist.github.com/FerusAndBeyond
def correlation(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> Any:
    shortest = min(x.shape[0], y.shape[0])
    return np.corrcoef(x[:shortest], y[:shortest])[0, 1]


def shift_for_maximum_correlation(
    x: npt.NDArray[Any], y: npt.NDArray[Any], time: npt.NDArray[Any]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    correlation = correlate(x, y)
    lags = correlation_lags(x.size, y.size)
    lag = lags[np.argmax(correlation)]
    print(f"Best lag: {lag}")
    if lag < 0:
        y = y[abs(lag) :]
        x = x[: -abs(lag)]
        time = time[: -abs(lag)]
    else:
        time = time[lag:]
        x = x[lag:]
        y = y[:-lag]
    return x, y, time
