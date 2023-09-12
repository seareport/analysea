#!/usr/bin/env python
import json
import os
import sys

import numpy as np
import pandas as pd
from searvey import ioc

from analysea.filters import remove_numerical
from analysea.filters import signaltonoise
from analysea.plot import plot_gaps
from analysea.plot import plot_multiyear_tide_analysis
from analysea.spikes import despike_prominence
from analysea.steps import step_function_ruptures
from analysea.tide import demean_amps_phases
from analysea.tide import yearly_tide_analysis
from analysea.utils import completeness
from analysea.utils import correct_unit
from analysea.utils import detect_gaps
from analysea.utils import json_format

# ===================
# global variables
# ===================
# load in data information
IOC_STATIONS = ioc.get_ioc_stations()
IOC_STATIONS["lon"] = IOC_STATIONS["geometry"].x
IOC_STATIONS["lat"] = IOC_STATIONS["geometry"].y
IOC_STATIONS = IOC_STATIONS.drop("geometry", axis=1)
DATA_FOLDER = os.getcwd()  # put all the stations data in the same folder as this script

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

IOC_OUT = pd.DataFrame(IOC_STATIONS)


# main call
def main():
    for _ioc_file in sorted(os.listdir(os.path.join(os.getcwd(), "data"))):
        if _ioc_file.endswith(".gzip"):
            print("   > _ioc_file = ", _ioc_file)
            # here is the IOC case
            _ioc_code = os.path.splitext(_ioc_file)[0]
            _ioc_index = np.where(IOC_OUT.ioc_code == _ioc_code)[0][0]
            df0 = pd.read_parquet(os.path.join(DATA_FOLDER, "data", _ioc_code + ".gzip"))
            # fill already some info
            lat = IOC_OUT.iloc[_ioc_index].lat
            lon = IOC_OUT.iloc[_ioc_index].lon
            title = IOC_OUT.iloc[_ioc_index].location
            filenameOut = os.path.join("./tests/data/graphs", _ioc_code + ".png")
            filenameOutGaps = os.path.join("./tests/data/graphs", _ioc_code + "_gaps.png")
            OPTS["lat"] = lat
            # start completing the json export file
            js_1 = dict()
            js_1["lat"] = lat
            js_1["lon"] = lon
            min_time = pd.Timestamp(df0.index.min())
            max_time = pd.Timestamp(df0.index.max())
            js_1["first_obs"] = min_time
            js_1["last_obs"] = max_time

            # check if the file exists
            if os.path.exists(filenameOut):
                print("   > file ", filenameOut, " already exists")
                continue
            #
            try:
                df = pd.DataFrame()
                df["slevel"] = remove_numerical(df0.slevel)
                df["correct"], units_flag = correct_unit(df.slevel)
                df["step"], stepsx, steps = step_function_ruptures(df.correct)
                threshold = np.max([1, df.correct.std() * 3])
                if (len(stepsx) > 2) and (np.max(steps) > 1.0):  # if step are not bigger than 1 meter
                    ipeaks, peaks, df["correct"] = despike_prominence(
                        df.correct - df.step, threshold
                    )  # despike once already
                _, _, df["anomaly"] = despike_prominence(df.correct, threshold)
                # calculate completeness
                cpltness = completeness(df.anomaly)
                missing = 100 - np.round(cpltness, 2)
                if cpltness < 70:
                    IOC_OUT.loc[_ioc_index, ["analysis"]] = "Failed: < 70% completeness"
                    IOC_OUT.loc[_ioc_index, ["missing"]] = missing
                    IOC_OUT.loc[_ioc_index, ["analysed"]] = False
                    #
                    js_1["analysis"] = "Failed: missing data"
                    js_1["missing"] = missing
                    js_1["analysed"] = False
                    #
                    print("completeness < 70% :", cpltness)
                    with open(f"./tests/data/processed/{_ioc_code}.json", "w") as f:
                        json.dump(json_format(js_1), f, indent=2)
                    continue
                #
                if (max_time - min_time).days < 365:
                    IOC_OUT.loc[_ioc_index, ["analysis"]] = "Failed: not enough data"
                    IOC_OUT.loc[_ioc_index, ["missing"]] = missing
                    IOC_OUT.loc[_ioc_index, ["analysed"]] = False
                    #
                    js_1["analysis"] = "Failed: missing data"
                    js_1["missing"] = missing
                    js_1["analysed"] = False
                    #
                    print("ignore : period is less than a year:", (max_time - min_time).days, "days")
                    with open(f"./tests/data/processed/{_ioc_code}.json", "w") as f:
                        json.dump(json_format(js_1), f, indent=2)
                    continue
                else:
                    # detect big gaps
                    gaps, small_gaps, big_gaps = detect_gaps(df)
                    plot_gaps(df.anomaly, big_gaps, filenameOutGaps)
                    # assign parameters before running tides (eventual crash)
                    IOC_OUT.loc[_ioc_index, ["missing"]] = 100 - np.round(cpltness, 2)
                    IOC_OUT.loc[_ioc_index, ["biggest_gap"]] = str(big_gaps.max())
                    IOC_OUT.loc[_ioc_index, ["total_gaps"]] = len(gaps)
                    IOC_OUT.loc[_ioc_index, ["steps"]] = len(stepsx)
                    IOC_OUT.loc[_ioc_index, ["snr"]] = (signaltonoise(df.slevel),)
                    IOC_OUT.loc[_ioc_index, ["unit_flag"]] = units_flag
                    #
                    js_1["missing"] = 100 - np.round(cpltness, 2)
                    js_1["biggest_gap"] = str(big_gaps.max())
                    js_1["total_gaps"] = len(gaps)
                    js_1["steps"] = len(stepsx)
                    js_1["snr"] = (signaltonoise(df.slevel),)
                    js_1["unit_flag"] = units_flag
                    #
                    df1 = df.reset_index().drop_duplicates(subset="time", keep="last").set_index("time")
                    df1["tide"], df1["surge"], coefs, years = yearly_tide_analysis(df1.anomaly, 365, OPTS)
                    # save figures
                    if len(coefs) == 0:
                        IOC_OUT.iloc[_ioc_index]["analysis"] = js_1["analysis"] = "Failed: during tides"
                        with open(f"./tests/data/processed/{_ioc_code}.json", "w") as f:
                            json.dump(js_1, f, indent=2)
                        continue
                    # printing out results
                    IOC_OUT.loc[_ioc_index, ["perc_analysed"]] = js_1["perc_analysed"] = (
                        len(df1.surge) / len(df.slevel) * 100
                    )
                    IOC_OUT.loc[_ioc_index, ["analysis"]] = js_1["analysis"] = "Success"
                    IOC_OUT.loc[_ioc_index, ["analysed"]] = js_1["analysed"] = True
                    # save the coefs calculated and average them
                    const, mean_amps, mean_phases = demean_amps_phases(coefs, coefs[0]["name"])
                    js_out = json_format(coefs[-1])
                    js_1 = json_format(js_1)
                    js_out["weights"] = 0  # weights list is too long and unused in the reconstruction
                    js_out["A"] = mean_amps.tolist()
                    js_out["g"] = mean_phases.tolist()
                    for key in js_1.keys():
                        js_out[key] = js_1[key]
                    with open(f"tests/data/processed/{_ioc_code}.json", "w") as fp:
                        json.dump(js_out, fp)

                    out = pd.DataFrame(data=df1.surge, index=df1.index)
                    out.to_parquet(f"./data/processed/{_ioc_code}_surge.gzip", compression="gzip")
                    # plots
                    plot_multiyear_tide_analysis(
                        ASTRO_PLOT, coefs, years, lat, lon, df1, title, filenameOut
                    )
                    plot_multiyear_tide_analysis(
                        ASTRO_PLOT, coefs, years, lat, lon, df1, title, filenameOut, zoom=True
                    )

            except Exception as e:
                print("Error: ", e)
                IOC_OUT.loc[_ioc_index, ["first_obs"]] = min_time.strftime("%Y-%m-%dT%H:%M:%S")
                IOC_OUT.loc[_ioc_index, ["last_obs"]] = (max_time.strftime("%Y-%m-%dT%H:%M:%S"),)
                IOC_OUT.loc[_ioc_index, ["analysed"]] = False
            IOC_OUT.to_parquet("IOC_api_analysed.gzip")
            with open(f"./data/processed/{_ioc_code}.json", "w") as fp:
                json.dump(json_format(js_1), f, indent=2)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ~~~~ Jenkins' success message ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n\nMy work is done\n\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
