from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import utide

from analysea.utils import completeness
from analysea.utils import nd_format

# tidal analysis options
OPTS = {
    "conf_int": "linear",
    "constit": "auto",
    "method": "ols",  # ols is faster and good for missing data (Ponchaut et al., 2001)
    "order_constit": "frequency",
    "Rayleigh_min": 0.97,
    "lat": None,
    "verbose": False,
}  # careful if there is only one Nan parameter, the analysis crashes

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

def circular_mean(angles: npt.NDArray[Any]) -> npt.NDArray[Any]:
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
    vect = np.ones(len(coefs)) / len(coefs)
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
    kwargs: Dict[str, Any]
) -> npt.NDArray[Any]:
    # resample to 10 min for a MUCH faster analysis
    # https://github.com/wesleybowman/UTide/issues/103
    h_rsmp = ts.resample("10min").mean()
    ts = h_rsmp.index
    coef = utide.solve(ts, h_rsmp, **kwargs)
    return coef


def detide(
    ts: pd.Series[float], 
    kwargs: Dict[str, Any],
    constituents: dict[str, float] | None = None,
) -> pd.Series:
    if constituents is None:
        constituents = calc_constituents(ts=ts)
        tidal = utide.reconstruct(ts.index, nd_format(constituents), verbose=kwargs["verbose"])
    else : 
        tidal = utide.reconstruct(ts.index, nd_format(constituents), verbose=kwargs["verbose"])
    storm_surge = ts - tidal.h
    return storm_surge


def detide_yearly(h: pd.Series[float], 
                         split_period: int, 
                         kwargs: Dict[str, Any] = OPTS) -> pd.Series:
    
    log = kwargs.get("verbose", False)
    
    min_time = pd.Timestamp(h.index.min())
    max_time = pd.Timestamp(h.index.max())
    date_ranges = pd.date_range(start=min_time, end=max_time, freq=f'{split_period}D')
    surge = pd.DataFrame([])

    for start, end in zip(date_ranges[:-1], date_ranges[1:]):
        signal = h[start:end]
        if completeness(signal) > 70:
            surge_tmp = detide(signal, kwargs)
            surge = pd.concat([surge, surge_tmp])
        if log:
            print(f"  => Analyse year {start.year} ({start}-{end})")
            print(f"   +>  {len(surge)} / {len(h)} records done")

    return surge


def tide_analysis(
    ts: pd.Series[float], 
    kwargs: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[Any]]:
    constituents = calc_constituents(ts=ts, kwargs=kwargs)
    tidal = utide.reconstruct(ts.index, constituents, verbose=kwargs["verbose"])
    return pd.DataFrame(data=tidal.h, index=ts.index), ts - tidal.h, constituents


def yearly_tide_analysis(h: pd.Series[float], 
                         split_period: int, 
                         kwargs: Dict[str, Any] = OPTS) -> Tuple[pd.DataFrame, pd.DataFrame, List[npt.NDArray[Any]], List[int]]:
    
    log = kwargs.get("verbose", False)
    
    min_time = pd.Timestamp(h.index.min())
    max_time = pd.Timestamp(h.index.max())
    date_ranges = pd.date_range(start=min_time, end=max_time, freq=f'{split_period}D')
    
    tide = pd.DataFrame([])
    surge = pd.DataFrame([])
    coefs = []
    years = []

    for start, end in zip(date_ranges[:-1], date_ranges[1:]):
        signal = h[start:end]

        if completeness(signal) > 70:
            years.append(start.year)
            tide_tmp, surge_tmp, coef = tide_analysis(signal, kwargs)
            coefs.append(coef)
            tide = pd.concat([tide, tide_tmp])
            surge = pd.concat([surge, surge_tmp])
            
        if log:
            print(f"  => Analyse year {start.year} ({start}-{end})")
            print(f"   +>  {len(tide)} / {len(h)} records done")

    return tide, surge, coefs, years