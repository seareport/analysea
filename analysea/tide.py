from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import utide
from typing_extensions import Unpack

from .custom_types import UTideArgs
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


@dataclass
class TideAnalysisResults:
    tide: pd.DataFrame
    surge: pd.DataFrame
    coefs: list[dict[str, Any]]
    years: list[int]


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
    ts: pd.Series,
    resample_time: int = 30,
    resample_detide: bool = False,
    **kwargs: Unpack[UTideArgs],
) -> Any:
    """
    Calculate the tide constituents for a time series.

    Parameters
    ----------
    @param ts: (pd.Series) The time series to be processed.
    @param resample_time: (int) resample time in minutes, by default 30.
    @param resample_detide: (bool) resample the detided signal (for faster process)
        default is False
    **kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the utide.solve function.
        "lat" argument being the only one required.

    Returns
    -------
    np.ndarray
        The tide constituents.

    """
    if resample_detide:
        # resample to 30 min for a MUCH faster analysis
        # https://github.com/wesleybowman/UTide/issues/103
        ts = ts.resample(f"{resample_time}min").mean()

        # shifting half the sampling time to recenter the averaged signal
        ts = ts.shift(freq=f"{resample_time / 2}min")
    coef = utide.solve(ts.index, ts, **kwargs)
    return coef


def detide(
    ts: pd.Series[float],
    constituents: Dict[Any, Any] | None = None,
    resample_time: int = 30,
    chunk: int = 100000,
    resample_detide: bool = False,
    **kwargs: Unpack[UTideArgs],
) -> pd.Series:
    """
    By default this function will split the time series into yearly chunks

    @param ts: (pd.Series) The time series to be processed.
    @param constituents: (dict) constituents to be used in the analysis
    @param resample_time: (int) resample time in minutes, by default 30.
    @param chunk: (int) length of the chunks for splitting the time series (for faster process)
    @param resample_detide: (bool) resample the detided signal (for faster process)
        default is False
    @param kwargs: keyword arguments to be passed to calc constituents ("lat"
        argument being the only one required)
    @return: reconstructed time series
    """
    verbose = kwargs.get("verbose", False)
    result_series = []
    if constituents is None:
        constituents = calc_constituents(
            ts=ts, resample_time=resample_time, resample_detide=resample_detide, **kwargs
        )

    if resample_detide:
        ts = ts.resample(f"{resample_time}min").apply(np.nanmean)
        ts = ts.shift(freq=f"{resample_time / 2}min")

    for start in range(0, len(ts), chunk):
        end = min(start + chunk, len(ts))
        ts_chunk = ts.iloc[start:end]

        if not ts_chunk.empty:
            tidal = utide.reconstruct(ts_chunk.index, nd_format(constituents), verbose=verbose)
            storm_surge = ts_chunk - tidal.h
            result_series.append(storm_surge)
    return pd.concat(result_series)


def tide_analysis(
    ts: pd.Series[float],
    resample_time: int = 30,
    resample_detide: bool = False,
    **kwargs: Unpack[UTideArgs],
) -> TideAnalysisResults:
    """
    Perform a tide analysis on a time series.

    Parameters
    ----------
    @param ts: (pd.Series) The time series to be processed.
    @param resample_time: (int) resample time in minutes, by default 30.
    @param resample_detide: (bool) resample the detided signal (for faster process)
        default is False
    **kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the utide.solve function.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[Any]]
        A tuple containing the following elements:

        * tide : pd.DataFrame
            A dataframe containing the tide values.
        * surge : pd.DataFrame
            A dataframe containing the surge values.
        * constituents : npt.NDArray[Any]
            The constituents used in the analysis.

    """
    verbose = kwargs.get("verbose", False)
    constituents = calc_constituents(
        ts=ts, resample_time=resample_time, resample_detide=resample_detide, **kwargs
    )

    if resample_detide:
        ts = ts.resample(f"{resample_time}min").apply(np.nanmean)
        ts = ts.shift(freq=f"{resample_time / 2}min")

    tidal = utide.reconstruct(ts.index, constituents, verbose=verbose)
    tide = pd.Series(data=tidal.h, index=ts.index)
    surge = pd.Series(data=ts.values - tidal.h, index=ts.index)
    return TideAnalysisResults(
        tide=tide.to_frame(), surge=surge.to_frame(), coefs=[constituents], years=[ts.index.year[0]]
    )


def yearly_tide_analysis(
    h: pd.Series[float],
    resample_time: int = 30,
    split_period: int = 365,
    **kwargs: Unpack[UTideArgs],
) -> TideAnalysisResults:
    """
    Perform a tide analysis on a time series, split into yearly intervals.

    Parameters
    ----------
    h : pd.Series
        The time series to analyze.
    resample_time : int, optional
        The resample time in minutes, by default 30.
    split_period : int, optional
        The period in days to split the time series into, by default 365.
    **kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the utide.solve function.
        "lat" argument being the only one required.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[npt.NDArray[Any]], List[int]]
        A tuple containing the following elements:

        * tide : pd.DataFrame
            A dataframe containing the tide values.
        * surge : pd.DataFrame
            A dataframe containing the surge values.
        * coefs : List[npt.NDArray[Any]]
            The constituents used in the analysis for each year.
        * years : List[int]
            The years analyzed.

    """
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
        ta = tide_analysis(signal, resample_time=resample_time, **kwargs)
        coefs.append(ta.coefs[0])
        surge.append(ta.surge)
        tide.append(ta.tide)

        if log:
            print(f"  => Analyse year {start.year} ({start}-{end})")
            print(f"   +>  {len(tide)} / {len(h)} records done")

    return TideAnalysisResults(tide=pd.concat(tide), surge=pd.concat(surge), coefs=coefs, years=years)
