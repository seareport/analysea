from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import utide

from analysea.utils import nd_format

ASTRO_PLOT = [
    "M2",
    "S2",
    "N2",
    "O1",
    "K2",
    "K1",
    "NU2",
    "Q1",
    "L2",
    "P1",
    "2N2",
    "M4",
    "MS4",
    "MM",
    "MU2",
    "SSA",
    "LDA2",
    "MF",
    "MSM",
    "MN4",
]

ASTRO_WRITE = [
    "M2",
    "S2",
    "N2",
    "O1",
    "K2",
    "K1",
    "NU2",
    "Q1",
    "L2",
    "P1",
    "2N2",
    "M4",
    "MS4",
    "MM",
    "MU2",
    "SSA",
    "LDA2",
    "MF",
    "MSM",
    "MN4",
]


def get_const_amps_labels(
    keep: List[str], coef: Dict[Any, Any]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    recognise & plot only the main constituents
    """
    ix = []
    for c in keep:
        if c in coef["name"].tolist():
            i = coef["name"].tolist().index(c)
            ix.append(i)
    amps = np.append(coef["A"][np.sort(ix)], np.zeros(len(keep) - len(ix)))
    const = np.append(coef["name"][np.sort(ix)], np.empty(len(keep) - len(ix)))
    phases = np.append(coef["g"][np.sort(ix)], np.empty(len(keep) - len(ix)))
    return amps, const, phases


def circular_mean(angles: npt.NDArray[Any]) -> Any:
    angles_rad = np.deg2rad(angles)
    mean_x = np.mean(np.sin(angles_rad))
    mean_y = np.mean(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(mean_x, mean_y)
    return np.rad2deg(mean_angle_rad)


def demean_amps_phases(
    coefs: List[Dict[Any, Any]], keep_const: List[str]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    amps = np.zeros((len(keep_const), len(coefs)))
    phases = np.zeros((len(keep_const), len(coefs)))
    #
    for iyear, coef in enumerate(coefs):
        _amps, const, _phases = get_const_amps_labels(keep_const, coef)
        amps[:, iyear] = _amps
        phases[:, iyear] = _phases
    #
    mean_amps = np.apply_along_axis(np.mean, 1, amps)
    mean_phases = np.apply_along_axis(circular_mean, 1, phases)
    #
    return const, mean_amps, mean_phases  # ignore mypy


def calc_constituents(
    ts: pd.Series, resample_time: int = 10, lat: float = 0.0, **kwargs: Optional[Dict[str, Any]]
) -> Any:
    # resample to 10 min for a MUCH faster analysis
    # https://github.com/wesleybowman/UTide/issues/103
    h_rsmp = ts.resample(f"{resample_time}min").mean()

    ts = h_rsmp.index
    # shifting half the sampling time
    # > to recenter the averaged signal
    df = h_rsmp.shift(freq=f"{resample_time // 2}min")
    ts = df.index
    df = h_rsmp.iloc[:, 0].values
    coef = utide.solve(ts, df, lat=lat, **kwargs)
    return coef


def reconstruct_chunk(
    ts_chunk: pd.Series, constituents: Dict[Any, Any], **kwargs: Optional[Dict[Any, Any]]
) -> pd.Series:
    """
    Reconstruct a single chunk of time series data.
    This function is mainly aimed to reduce intense i/o loads (RAM usage)
    """

    tidal = utide.reconstruct(ts_chunk.index, nd_format(constituents), **kwargs)

    return pd.Series(data=ts_chunk.iloc[:, 0].values - tidal.h, index=ts_chunk.index)


def detide(
    ts: pd.Series[float],
    constituents: Dict[Any, Any] | None = None,
    lat: float = 0,
    resample_time: int = 10,
    split_period: int = 365,
    **kwargs: Optional[Dict[Any, Any]],
) -> pd.Series:
    """
    By default this function will split the time series into yearly chunks

    @param ts: time series to be processed
    @param constituents: constituents to be used in the analysis
    @param lat: latitude of the station
    @param resample_time: resample time in minutes
    @param split_period: period in days to split the time series into (default 365)
    @param kwargs: keyword arguments to be passed to utide.reconstruct
    @return: reconstructed time series
    """
    result_series = []
    date_ranges = pd.date_range(start=ts.index[0], end=ts.index[-1], freq=f"{split_period}D")
    # Split the time series into 6-month chunks (default), otherwise specify
    for start_date, end_date in zip(date_ranges[:-1], date_ranges[1:]):
        ts_chunk = ts[start_date:end_date]
        if constituents is None:
            constituents = calc_constituents(ts=ts, resample_time=resample_time, lat=lat, **kwargs)
        result_series.append(reconstruct_chunk(ts_chunk, constituents, **kwargs))

    # Concatenate the processed chunks
    return pd.concat(result_series)


def tide_analysis(
    ts: pd.Series[float], resample_time: int = 10, lat: float = 0.0, **kwargs: Optional[Dict[Any, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[Any]]:
    constituents = calc_constituents(ts=ts, lat=lat, resample_time=resample_time, **kwargs)
    tidal = utide.reconstruct(ts.index, constituents, **kwargs)
    tide = pd.Series(data=tidal.h, index=ts.index)
    surge = pd.Series(data=ts.iloc[0, :].values - tidal.h, index=ts.index)
    return tide, surge, constituents


def yearly_tide_analysis(
    h: pd.Series[float],
    resample_time: int = 10,
    split_period: int = 365,
    lat: int = 0,
    **kwargs: Optional[Dict[Any, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[npt.NDArray[Any]], List[int]]:
    log = kwargs.get("verbose", False)

    min_time = pd.Timestamp(h.index.min())
    max_time = pd.Timestamp(h.index.max())
    date_ranges = pd.date_range(start=min_time, end=max_time, freq=f"{split_period}D")

    tide = []
    surge = []
    coefs = []
    years = []

    for start, end in zip(date_ranges[:-1], date_ranges[1:]):
        signal = h[start:end]

        years.append(start.year)
        tide_tmp, surge_tmp, coef = tide_analysis(signal, lat=lat, resample_time=resample_time, **kwargs)
        coefs.append(coef)
        surge.append(surge_tmp)
        tide.append(tide_tmp)

        if log:
            print(f"  => Analyse year {start.year} ({start}-{end})")
            print(f"   +>  {len(tide)} / {len(h)} records done")

    return pd.concat(tide), pd.concat(surge), coefs, years
