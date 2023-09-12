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
    mean_amps = np.dot(amps, vect)
    mean_phases = np.dot(phases, vect)

    # do one more iteration, and drop value outside of standard deviation
    std_amps = np.std(amps, axis=1)
    amps = np.zeros((len(keep_const), len(coefs)))
    phases = np.zeros((len(keep_const), len(coefs)))
    #
    iM2 = const.tolist().index("M2")
    stdM2 = std_amps[iM2]
    MM2 = mean_amps[iM2]

    for iyear, coef in enumerate(coefs):
        _amps, const, _phases = get_const_amps_labels(keep_const, coef)
        if abs(_amps[iM2] - MM2) < stdM2:
            amps[:, iyear] = _amps
            phases[:, iyear] = _phases
        else:
            vect[iyear] = 0
    #
    mean_amps = np.dot(amps, vect) / vect.sum()
    mean_phases = np.dot(phases, vect)
    #
    return const, mean_amps, mean_phases  # ignore mypy


def tide_analysis(
    h: pd.Series[float], kwargs: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[Any]]:
    h = h - h.mean()  # ensuring mean
    # resample to 10 min for a MUCH faster analysis
    # https://github.com/wesleybowman/UTide/issues/103
    h_rsmp = h.resample("10min").mean()
    ts = h_rsmp.index
    coef = utide.solve(ts, h_rsmp, **kwargs)
    tidal = utide.reconstruct(h.index, coef, verbose=kwargs["verbose"])
    return pd.DataFrame(data=tidal.h, index=h.index), h - tidal.h, coef


def yearly_tide_analysis(
    h: pd.Series[float], split_period: int, kwargs: Dict[str, Any] = OPTS
) -> Tuple[pd.DataFrame, pd.DataFrame, List[npt.NDArray[Any]], List[int]]:
    if kwargs["verbose"]:
        log = True
    else:
        log = False
    #
    ts = h.index
    min_time = pd.Timestamp(ts.min())
    max_time = pd.Timestamp(ts.max())
    t_tmp = pd.Timestamp(min_time)
    n_years = int(np.floor((max_time - min_time).days / split_period))
    years = []
    tide = pd.DataFrame([])
    surge = pd.DataFrame([])
    coefs = []
    for i in range(n_years):
        if i == n_years - 1:
            signal = h.loc[lambda x: (x.index > t_tmp) & (x.index < max_time)]
            if completeness(signal) > 70:
                years.append(t_tmp.year)
                tide_tmp, surge_tmp, coef = tide_analysis(signal, kwargs)
                coefs.append(coef)
                tide = pd.concat([tide, tide_tmp])
                surge = pd.concat([surge, surge_tmp])
            t_tmp += pd.Timedelta(days=split_period)
        else:
            signal = h.loc[
                lambda x: (x.index > t_tmp) & (x.index < t_tmp + pd.Timedelta(days=split_period))
            ]
            if completeness(signal) > 70:
                years.append(t_tmp.year)
                tide_tmp, surge_tmp, coef = tide_analysis(signal, kwargs)
                coefs.append(coef)
                tide = pd.concat([tide, tide_tmp])
                surge = pd.concat([surge, surge_tmp])
            t_tmp += pd.Timedelta(days=split_period)
        if log:
            print("  => Analyse year", i, "of", n_years)
            print("   +> ", len(tide), "/", len(h), "records done")
    # tide = tide.reset_index().drop_duplicates(subset="index", keep="last").set_index("index")
    # surge = surge.reset_index().drop_duplicates(subset="index", keep="last").set_index("index")
    return tide, surge, coefs, years
