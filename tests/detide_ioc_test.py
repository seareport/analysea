import os
import sys

import numpy as np
import pandas as pd
from searvey import ioc

from analysea.filters import filter_fft
from analysea.filters import remove_numerical
from analysea.filters import signaltonoise
from analysea.plot import plot_gaps
from analysea.plot import plot_multiyear_tide_analysis
from analysea.spikes import despike_prominence
from analysea.steps import step_function_ruptures
from analysea.tide import yearly_tide_analysis
from analysea.utils import correct_unit
from analysea.utils import detect_gaps

# ===================
# global variables
# ===================
# load in data information
IOC_STATIONS = ioc.get_ioc_stations()
IOC_CODE = "wlgt"

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


# main call
def main():
    # here is the IOC code
    # fmt: off
    _ioc_index = np.where(IOC_STATIONS.ioc_code == IOC_CODE)[0][0]
    df0 = pd.read_parquet(os.path.join( "tests/data", IOC_CODE + ".gzip"))
    #
    lat = IOC_STATIONS.iloc[_ioc_index].lat
    lon = IOC_STATIONS.iloc[_ioc_index].lon
    title = IOC_STATIONS.iloc[_ioc_index].location
    filenameOutGaps = os.path.join('tests/data/graphs', IOC_CODE + "_gaps.png")
    filenameOut = os.path.join('tests/data/graphs', IOC_CODE + ".png")
    #
    df = pd.DataFrame()
    df["slevel"] = remove_numerical(df0.slevel)
    df["correct"], _ = correct_unit(df.slevel)
    df["step"], stepsx, steps = step_function_ruptures(df.correct)
    if (len(stepsx) > 2) & (np.max(steps) > 1.0):
        # if step are not bigger than 1 meter
        df.correct = df.slevel - df.step
    threshold = np.max([1,df.correct.std() * 3])
    _, _, df["anomaly"] = despike_prominence(df.correct, threshold)
    _, _, big_gaps = detect_gaps(df)
    plot_gaps(df.anomaly.interpolate(), big_gaps, filenameOutGaps)
    # assign parameters for tide analysis
    if signaltonoise(df.slevel) < 0:
        df["filtered"] = filter_fft(df.anomaly)
        df.anomaly = df.filtered
    #
    OPTS["lat"] = lat
    df["tide"], df["surge"], coefs, years = yearly_tide_analysis(df.anomaly, 365, OPTS)
    plot_multiyear_tide_analysis(ASTRO_PLOT, coefs, years, lat, lon, df, title, filenameOut)
    plot_multiyear_tide_analysis(ASTRO_PLOT, coefs, years, lat, lon, df, title, filenameOut, zoom=True)
    # fmt: on

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ~~~~ Jenkins' success message ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n\nMy work is done\n\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
