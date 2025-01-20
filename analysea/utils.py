from __future__ import annotations

import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags

from analysea.spikes import remove_outliers


# ===================
# TIME SERIES
# ===================
def resample(df: pd.DataFrame, t_rsp: int = 30) -> pd.DataFrame:
    """
    Resample a pandas dataframe to a new time interval.

    @param df (pd.DataFrame): The input dataframe.
    @param t_rsp (int): optional, The target resample period in minutes
    by default 30.

    @returns (pd.DataFrame): The resampled dataframe.
    """
    ts = df.resample(f"{t_rsp}min").mean().shift(freq=f"{int(t_rsp/2)}min")
    return ts


def interpolate(df: pd.DataFrame, t_rsp: int = 30) -> pd.DataFrame:
    """
    This function resamples a pandas dataframe to a new time interval
    using linear interpolation.

    It uses analysea's detect_time_step function to interpolate only
    between holes in the data and not extrapolate "flat areas" in the
    signal

    @param df (pd.DataFrame): The input dataframe.
    @param t_rsp (int): optional, The target resample period in minutes
    by default 30.

    @returns (pd.DataFrame): The interpolated dataframe.
    """
    time_step = detect_time_step(df)
    n_interp = int(t_rsp * 60 / time_step.total_seconds())
    ts = df.interpolate(method="linear", limit=n_interp)
    return ts


def detect_splits(sr: pd.Series, max_gap: pd.Timedelta) -> pd.DatetimeIndex:
    split_points = pd.DatetimeIndex([sr.index[0], sr.index[-1]])
    condition = sr.index.to_series().diff() > max_gap
    for i, point in enumerate(sr[condition].index, 1):
        split_points = split_points.insert(i, point)
    return split_points


def split_series(sr: pd.Series, max_gap: pd.Timedelta = pd.Timedelta(hours=24)) -> Iterator[pd.Series]:
    """
    Splits a pandas series into segments without overlapping gaps larger than max_gap.

    @param sr (pd.Series): The input series.
    @param max_gap (pd.Timedelta): The maximum allowed gap between two segments.

    @returns: Iterator[pd.Series]: An iterator of segments.
    """
    for start, stop in itertools.pairwise(detect_splits(sr=sr, max_gap=max_gap)):
        segment = sr[start:stop]
        yield segment[:-1]


def calc_stats(segments: list[pd.Series]) -> pd.DataFrame:
    data = []
    for i, segment in enumerate(segments):
        ss = dict(
            start=segment.index[0],
            end=segment.index[-1],
            duration=segment.index[-1] - segment.index[0],
            scount=segment.count(),
            smean=segment.mean(),
            sstd=segment.std(),
            smin=segment.min(),
            s01=segment.quantile(0.01),
            s10=segment.quantile(0.10),
            s25=segment.quantile(0.25),
            s50=segment.quantile(0.50),
            s75=segment.quantile(0.75),
            s90=segment.quantile(0.90),
            s99=segment.quantile(0.99),
            smax=segment.max(),
            sskewness=segment.skew(),
            skurtosis=segment.kurtosis(),
        )
        data.append(ss)
    stats = pd.DataFrame(data)
    return stats


def cleanup(
    ts: pd.Series,
    clip_limits: Optional[Tuple[float, float]] = None,
    kurtosis: float = 2.0,
    remove_flats: bool = False,
    despike: bool = True,
    demean: bool = True,
) -> pd.DataFrame:
    """
    This function cleans up a time series by removing outliers,
    detecting and removing flat areas, and removing steps.

    @param ts (pd.Series): The input time series.
    @param clip_limits tuple[float, float]: Optional, The lower and upper
      bounds for outlier detection. If None, outlier detection is not performed.
    @param kurtosis (float): The threshold for detecting outliers. If the absolute
      value of the kurtosis of a segment is less than this value, the segment is
      considered clean.
    @param remove_flats (bool): Whether to remove flat areas from the time series.
      If True, flat areas are detected by comparing the difference in consecutive values.
    @param despike (bool): Whether to remove outliers using the provided clip_limits.
      If True and clip_limits is not None, outlier detection is performed.
    @param demean (bool): Whether to demean the time series.
      If True, the mean of the time series is subtracted from each value.

    @returns (pd.DataFrame): The cleaned up time series.

    """
    # Check if the input is empty
    if ts.empty:
        return pd.DataFrame()
    if remove_flats:
        ts = ts[ts.diff() != 0]  # remove flat areas
    ss = ts[abs(ts.diff()) < ts.std()]  # remove steps
    if demean:
        ss = ss - ss.mean()
    if ss.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    for sensor in ss.columns:
        sr = ss[sensor]
        segments = [seg for seg in split_series(sr, max_gap=pd.Timedelta(hours=6)) if not seg.empty]
        for seg in segments:
            # remove outliers
            if despike and clip_limits:
                seg = remove_outliers(seg, *clip_limits)
            if seg.empty:
                continue
            stats = calc_stats([seg])
            if abs(stats.iloc[0].skurtosis) < kurtosis:
                df = pd.concat([df, pd.DataFrame({sensor: seg}, index=seg.index)])

    return df


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
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    time: npt.NDArray[Any],
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
