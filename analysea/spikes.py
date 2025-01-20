from __future__ import annotations

from typing import Any
from typing import Tuple

import numpy as np
import pandas as pd
import scipy as sp


def despike_prominence(df: pd.DataFrame, prominence: float) -> Tuple[Any, Any, pd.Series[Any]]:
    """
    input:
    @df: pd.Series
    @prominence: prominence of the spike: https://en.wikipedia.org/wiki/Topographic_prominence
    (recommendation: prominence = 3 * std. dev.: https://doi.org/10.3390/rs12233970 )

    output:
    @ipeaks: peaks indexes
    @peaks: peaks values
    """
    ipeaks, props = sp.signal.find_peaks(abs(df.interpolate()), prominence=prominence, width=1)
    peaks = df.iloc[ipeaks]
    res = pd.Series(data=df.values, index=df.index)
    # erase depending on the width of the spike
    for i, ip in enumerate(ipeaks):
        # width = props["widths"][i]
        left = int(props["left_ips"][i])
        right = int(props["right_ips"][i]) + 1
        res.iloc[left:right] = pd.NA
    return ipeaks, peaks, res


# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
def remove_outliers(sr: pd.Series, lower: float = 0.25, upper: float = 0.75) -> pd.Series:
    iqr = sr.quantile(upper) - sr.quantile(lower)
    lim = np.abs((sr - sr.median()) / iqr) < 2.22
    return sr[lim]


def EWMA(df: pd.DataFrame, span: int) -> pd.DataFrame:
    # Forwards EWMA.
    fwd = df.ewm(span=span).mean()
    # Backwards EWMA.
    bwd = df.iloc[::-1].ewm(span=span).mean()
    # Add and take the mean of the forwards and backwards EWMA.
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    np_fbewma = np.mean(stacked_ewma, axis=0)
    res = pd.DataFrame(data=np_fbewma, index=df.index)
    return res


def remove_spikes(spikey: pd.DataFrame, averaged_signal: pd.DataFrame, delta: float) -> pd.DataFrame:
    """
    method from https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
    """
    np_spikey = np.array(spikey)
    np_fbewma = np.array(averaged_signal)
    cond_delta = np.abs(np_spikey - np_fbewma) > delta
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    res = pd.DataFrame(np_remove_outliers, index=spikey.index)
    return res


def buffer_nans(df: pd.DataFrame, buffer: int) -> pd.DataFrame:
    # Forwards MA.
    fwd = df.rolling(buffer).mean()
    # Backwards MA.
    bwd = df.iloc[::-1].rolling(buffer).mean()
    bwd = bwd[::-1]
    cond_nan = np.logical_or(pd.isna(bwd), pd.isna(fwd))
    np_buffer = np.where(cond_nan, np.nan, df)
    res = pd.DataFrame(np_buffer, index=df.index)
    return res
