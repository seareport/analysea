from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

from analysea.utils import detect_time_step


def interp(df: pd.Series[Any], new_index: pd.Index[Any]) -> pd.DataFrame:
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(data=np.interp(new_index, df.index, df), index=new_index)
    return df_out


def filter_fft(df: pd.DataFrame, fcut: float = 20) -> pd.DataFrame:
    # df is a single channel dataframe with :
    # index as pandas.DatetimeIndex
    data = df.dropna().values

    temp_fft = sp.fftpack.fft(data)
    temp_psd = np.abs(temp_fft) ** 2
    # calculate the time step for the signal
    fs = 1 / detect_time_step(df).total_seconds()
    fA = fs * 3600 * 24  # seconds in a day
    fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1 / fA)
    temp_fft_bis = temp_fft.copy()
    temp_fft_bis[np.abs(fftfreq) > fcut] = 0
    # suppressing everything passed the 10th harmonic
    # (first one being the semi-diurnal constituent of the tide)
    temp_slow = np.real(sp.fftpack.ifft(temp_fft_bis))
    res = pd.Series(temp_slow, index=df[~df.isna()].index)
    return interp(res, df.index)


def signaltonoise(
    df: pd.DataFrame,
    axis: Literal["index", 0] | Literal["columns", 1] = 0,
    ddof: int = 0,
) -> Any:
    m = df.mean()
    sd = df.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(m / sd))


def remove_numerical(df: pd.DataFrame) -> pd.DataFrame:
    # remove numerical errors :
    df = df[df.isin(df.value_counts()[df.value_counts() < int(len(df) / 100)].index)]
    return df


# BUTTERWORTH FILTERS
# https://en.wikipedia.org/wiki/Butterworth_filter
def butter_filter(
    cutoff: float,
    fs: float,
    btype: str,
    order: int = 5,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_highpass_filter(data: pd.DataFrame, cutoff: float, fs: float, order: int = 5) -> pd.DataFrame:
    b, a = butter_filter(cutoff, fs, "high", order=order)
    y = sp.signal.filtfilt(b, a, data)
    return pd.DataFrame(data=y, index=data.index)


def butter_lowpass_filter(data: pd.DataFrame, cutoff: float, fs: float, order: int = 5) -> pd.DataFrame:
    b, a = butter_filter(cutoff, fs, "low", order=order)
    y = sp.signal.filtfilt(b, a, data)
    return pd.DataFrame(data=y, index=data.index)


# FIR FILTERS
def FIR_highpass(taps: int, cutoff: float, fs: float) -> Any:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b = sp.signal.firwin(taps, normal_cutoff, pass_zero=False)
    return b


def FIR_highpass_filter(data: pd.DataFrame, taps: int, cutoff: float, fs: float) -> pd.DataFrame:
    # fs : sampling frequency
    # cutoff : cutoff frequency
    # width of the anylisis window
    b = FIR_highpass(taps, cutoff, fs)
    y = sp.signal.lfilter(b, 1.0, data)
    return pd.DataFrame(data=y, index=data.index)


def clip_data(unclipped: pd.DataFrame, high_clip: float, low_clip: float) -> pd.DataFrame:
    """Clip unclipped between high_clip and low_clip.
    unclipped contains a single column of unclipped data."""
    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped)
    # clip data above HIGH_CLIP or below LOW_CLIP
    cond_high_clip = (np_unclipped > high_clip) | (np_unclipped < low_clip)
    np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
    return pd.DataFrame(data=np_clipped, index=unclipped.index)
