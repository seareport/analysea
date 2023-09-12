from __future__ import annotations

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER
from matplotlib.gridspec import GridSpec

from analysea.tide import demean_amps_phases
from analysea.tide import get_const_amps_labels
from analysea.utils import completeness

# ===================
# global variables
# ===================
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


# ===================
# PLOT FUNCTIONS
# ===================
def plot_gaps(
    df: pd.DataFrame, gaps: pd.Series[Any], fileout: str
) -> Tuple[None, matplotlib.pyplot.figure]:
    fig, ax = plt.subplots(figsize=(19, 10))
    df.interpolate().plot(ax=ax)
    for i, (ig, gap) in enumerate(gaps.items()):
        ax.hlines(y=0, xmin=ig - gap, xmax=ig, color="r", linewidth=30)
        if i == 0:
            ax.hlines(y=0, xmin=gaps.index[0], xmax=ig - gap, color="g", linewidth=10)
        else:
            ax.hlines(y=0, xmin=gaps.index[i - 1], xmax=ig - gap, color="g", linewidth=10)
    #
    textStr = "\n".join(
        (
            f"{len(gaps)} bigs gaps with average gap duration: {gaps.mean()}",
            f"with the biggest gap being : {gaps.max()}",
            f"completeness: {np.round(completeness(df),2)}%",
        )
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(0.7, 0.1, textStr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)
    plt.tight_layout()
    fig.savefig(fileout)
    plt.close()
    return fig, ax


def plot_multiyear_tide_analysis(
    keep_const: List[str],
    tides: List[Dict[Any, Any]],
    range_years: List[int],
    lat: float,
    lon: float,
    df: pd.DataFrame,
    title: str,
    fileout: str,
    zoom: bool = False,
) -> None:
    fig = plt.figure(layout="constrained")
    fig.set_size_inches(30, 30 / 1.61803398875)
    gs = GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    # time serie plot
    # min_time = pd.Timestamp(df.index.min())
    # max_time = pd.Timestamp(df.index.max())
    if zoom:
        df = df.loc[: df.index[0] + pd.Timedelta(days=60)]
    ax0.plot(df.index, df["anomaly"], label="Raw Signal", color="k", linestyle="-.")
    ax0.plot(df.index, df["tide"], label="Yearly Tide Fit", color="green")
    ax0.plot(df.index, df["surge"], label="Residual Yearly", color="cyan")
    ax0.set_xlabel("Time")
    ax0.grid(which="major", axis="y", linestyle="--", zorder=-1)
    ax0.set_ylabel("Elevation (m)")
    ax0.legend()
    # creating the bar plot
    width = 0.05  # the width of the bars
    multiplier = 0
    #
    x = np.arange(0, len(keep_const))
    mean_amps = np.zeros(len(keep_const))
    mean_phases = np.zeros(len(keep_const))

    for iyear, tide in enumerate(tides):
        year = range_years[iyear]
        amps, const, phases = get_const_amps_labels(keep_const, tide)
        mean_amps += amps
        mean_phases += phases
        offset = width * multiplier - width * len(keep_const) / 4
        ax1.bar(x + offset, amps, width, label=year)
        multiplier += 1

    _, mean_amps, mean_phases = demean_amps_phases(tides, keep_const)

    ax1.hlines(
        mean_amps,
        x - width * len(tides) * 0.6,
        x + width * len(tides) * 0.6,
        color="k",
        linestyles="--",
        alpha=0.6,
        label="mean",
    )
    rect = ax1.bar(x, mean_amps, alpha=0)  # only for the labels
    ax1.bar_label(
        rect,
        padding=1,
        fmt="%.3f",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xticks(x, const)
    ax1.grid(which="major", axis="y", color="k", linestyle="--", zorder=-1)
    ax1.grid(which="minor", axis="y", linestyle="--", zorder=-1)
    # ax1.set_yscale('log')
    ax1.set_xlabel("Tidals contituents")
    ax1.set_ylabel("Amplitude (m)")
    ax1.legend()
    ax0.set_title(title)
    fig.tight_layout()
    # add location of the gauge
    if zoom:
        xticks_extent = list(np.arange(np.floor(lon) - 1, np.floor(lon) + 2, 0.2))
        yticks_extent = list(np.arange(np.floor(lat) - 1, np.floor(lat) + 2, 0.1))
        root, ext = os.path.splitext(fileout)
        fileout = root + "_zoom." + ext
        ax2.set_extent([lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5], ccrs.PlateCarree())
    else:
        xticks_extent = list(np.arange(np.floor(lon) - 2, np.floor(lon) + 3, 0.5))
        yticks_extent = list(np.arange(np.floor(lat) - 2, np.floor(lat) + 3, 0.25))
        ax2.set_extent([lon - 2, lon + 2, lat - 2, lat + 2], ccrs.PlateCarree())
    ax2.add_feature(cf.COASTLINE)
    ax2.add_feature(cf.LAND)
    ax2.add_wms(
        wms="http://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv?",
        layers=["GEBCO_LATEST"],
    )
    ax2.scatter(lon, lat, color="r", marker="*", s=100, zorder=2)
    gl = ax2.gridlines(linewidths=0.1, linestyle="--")
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlocator = mticker.FixedLocator(xticks_extent)
    gl.ylocator = mticker.FixedLocator(yticks_extent)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    fig.savefig(fileout)
    plt.close()
