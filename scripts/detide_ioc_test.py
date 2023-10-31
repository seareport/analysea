#!/usr/bin/env python
import json
import os
import sys

import numpy as np
import pandas as pd
from searvey import ioc

from analysea.filters import remove_numerical
from analysea.plot import plot_gaps
from analysea.plot import plot_multiyear_tide_analysis
from analysea.spikes import despike_prominence
from analysea.steps import step_function_ruptures
from analysea.tide import demean_amps_phases
from analysea.tide import yearly_tide_analysis
from analysea.utils import correct_unit
from analysea.utils import detect_gaps
from analysea.utils import json_format

# ===================
# global variables
# ===================
# load in data information
IOC_STATIONS = ioc.get_ioc_stations()
IOC_CODE = "acap2"

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
    # fmt: off
    _ioc_index = np.where(IOC_STATIONS.ioc_code == IOC_CODE)[0][0]
    df0 = pd.read_parquet(os.path.join( "tests/data", IOC_CODE + ".gzip"))
    #
    lat = IOC_STATIONS.iloc[_ioc_index].lat
    lon = IOC_STATIONS.iloc[_ioc_index].lon
    title = IOC_STATIONS.iloc[_ioc_index].location
    OPTS["lat"] = lat

    # start completing the json export file
    js_1 =  dict()
    js_1['lat'] = lat
    js_1['lon'] = lon
    min_time = pd.Timestamp(df0.index.min())
    max_time = pd.Timestamp(df0.index.max())
    js_1['first_obs'] = min_time
    js_1['last_obs'] =  max_time
    js_1 = json_format(js_1)

    filenameOutGaps = os.path.join('tests/data/graphs', IOC_CODE + "_gaps.png")
    filenameOut = os.path.join('tests/data/graphs', IOC_CODE + ".png")
    #
    df = pd.DataFrame()
    df["slevel"] = remove_numerical(df0.slevel)
    df["correct"], _ = correct_unit(df.slevel)
    df["step"], stepsx, steps = step_function_ruptures(df.correct)
    threshold = np.max([1,df.correct.std() * 3])
    if (len(stepsx) > 2) and (np.max(steps) > 1.0):
        ipeaks, peaks, df['correct'] = despike_prominence(df.correct - df.step, threshold) # despike once already
    _, _, df["anomaly"] = despike_prominence(df.correct, threshold)
    # detect big gaps
    _, _, big_gaps = detect_gaps(df)
    plot_gaps(df.anomaly, big_gaps, filenameOutGaps)
    #
    df1 = df.reset_index().drop_duplicates(subset="time", keep="last").set_index("time")
    df1["tide"], df1["surge"], coefs, years = yearly_tide_analysis(df1.anomaly, 365, OPTS)
    plot_multiyear_tide_analysis(ASTRO_PLOT, coefs, years, lat, lon, df1, title, filenameOut)
    plot_multiyear_tide_analysis(ASTRO_PLOT, coefs, years, lat, lon, df1, title, filenameOut, zoom=True)
    # save the coefs calculated and average them
    const, mean_amps,mean_phases = demean_amps_phases(coefs, coefs[0]['name'])
    js_out = json_format(coefs[-1])
    js_out['weights'] = 0 # weights list is too long and unused in the reconstruction
    js_out['A'] = mean_amps.tolist()
    js_out['g'] = mean_phases.tolist()
    # js_out = js_out.pop('weights')
    for key in js_1.keys():
        js_out[key] = js_1[key]
    with open(f"tests/data/processed/{IOC_CODE}.json", "w") as fp:
        json.dump(js_out, fp)
    # fmt: on

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ~~~~ Jenkins' success message ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n\nMy work is done\n\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
