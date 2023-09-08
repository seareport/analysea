import json
import os
import sys

import numpy as np
import pandas as pd

from analysea.filters import filter_fft
from analysea.filters import remove_numerical
from analysea.filters import signaltonoise
from analysea.gesla import GeslaDataset
from analysea.plot import plot_gaps
from analysea.plot import plot_multiyear_tide_analysis
from analysea.spikes import despike_prominence
from analysea.steps import step_function_ruptures
from analysea.tide import demean_amps_phases
from analysea.tide import yearly_tide_analysis
from analysea.utils import calculate_completeness
from analysea.utils import correct_unit
from analysea.utils import detect_gaps

# ===================
# global variables
# ===================
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

meta_file = "/home/tomsail/apps/obs/GESLA/GESLA3_ALL.csv"
data_path = "/home/tomsail/apps/obs/GESLA/GESLA3.0_ALL/"

g3 = GeslaDataset(meta_file=meta_file, data_path=data_path)
GESLA_STATIONS = pd.DataFrame(g3.meta)
GESLA_STATIONS["first_obs"] = np.empty(len(GESLA_STATIONS), dtype=str)
GESLA_STATIONS["last_obs"] = np.empty(len(GESLA_STATIONS), dtype=str)
GESLA_STATIONS["missing"] = np.empty(len(GESLA_STATIONS))
GESLA_STATIONS["biggest_gap"] = np.empty(len(GESLA_STATIONS), dtype=str)
GESLA_STATIONS["total_gaps"] = np.empty(len(GESLA_STATIONS))
GESLA_STATIONS["steps"] = np.empty(len(GESLA_STATIONS))
GESLA_STATIONS["snr"] = np.empty(len(GESLA_STATIONS))
GESLA_STATIONS["unit_flag"] = np.empty(len(GESLA_STATIONS), dtype=bool)
GESLA_STATIONS["perc_analysed"] = np.empty(len(GESLA_STATIONS))
GESLA_STATIONS["analysis"] = np.empty(len(GESLA_STATIONS), dtype=str)
GESLA_STATIONS["analysed"] = np.empty(len(GESLA_STATIONS), dtype=bool)


# main call
def main():
    for _gesla_file in sorted(g3.meta.filename):
        # here is the IOC case
        _gesla_index = np.where(g3.meta.filename == _gesla_file)[0][0]
        df0, meta = g3.file_to_pandas(_gesla_file)
        df0 = df0.where(df0.use_flag == 1)
        print("   > _gesla_file = ", _gesla_file, "-", len(df0), "records")
        # fill already some info
        lat = GESLA_STATIONS.iloc[_gesla_index].latitude
        lon = GESLA_STATIONS.iloc[_gesla_index].longitude
        title = GESLA_STATIONS.iloc[_gesla_index].site_name
        filenameOut = os.path.join("./tests/data/graphs", _gesla_file + ".png")
        filenameOutGaps = os.path.join("./tests/data/graphs", _gesla_file + "_gaps.png")
        OPTS["lat"] = lat
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
            if calculate_completeness(df.anomaly) < 70:
                GESLA_STATIONS.loc[_gesla_index, ["analysis"]] = "Failed: missing data"
                GESLA_STATIONS.loc[_gesla_index, ["missing"]] = 100 - np.round(
                    calculate_completeness(df.anomaly), 2
                )
                GESLA_STATIONS.loc[_gesla_index, ["analysed"]] = False
                print("completeness < 70% :", calculate_completeness(df.anomaly))
                continue
            #
            min_time = pd.Timestamp(df.index.min())
            max_time = pd.Timestamp(df.index.max())
            if (max_time - min_time).days < 365:
                GESLA_STATIONS.loc[_gesla_index, ["analysis"]] = "Failed: time span less than a year"
                GESLA_STATIONS.loc[_gesla_index, ["missing"]] = 100 - np.round(
                    calculate_completeness(df.anomaly), 2
                )
                GESLA_STATIONS.loc[_gesla_index, ["analysed"]] = False
                print("ignore : period is less than a year:", (max_time - min_time).days, "days")
                continue
            else:
                # assign parameters for tide analysis
                if signaltonoise(df.slevel) < 0:
                    df["filtered"] = filter_fft(df.anomaly)
                    df.anomaly = df.filtered
                # detect big gaps
                gaps, small_gaps, big_gaps = detect_gaps(df)
                plot_gaps(df.anomaly, big_gaps, filenameOutGaps)
                # assign parameters before running tides (eventual crash)
                GESLA_STATIONS.loc[_gesla_index, ["missing"]] = 100 - np.round(
                    calculate_completeness(df.anomaly), 2
                )
                GESLA_STATIONS.loc[_gesla_index, ["biggest_gap"]] = str(big_gaps.max())
                GESLA_STATIONS.loc[_gesla_index, ["total_gaps"]] = len(gaps)
                GESLA_STATIONS.loc[_gesla_index, ["steps"]] = len(stepsx)
                GESLA_STATIONS.loc[_gesla_index, ["snr"]] = (signaltonoise(df.slevel),)
                GESLA_STATIONS.loc[_gesla_index, ["unit_flag"]] = units_flag
                #
                df["tide"], df["surge"], coefs, years = yearly_tide_analysis(df.anomaly, 365, OPTS)
                # save figures
                if len(coefs) == 0:
                    GESLA_STATIONS.iloc[_gesla_index]["analysis"] = "Failed: missing data"
                    continue
                # printing out results
                GESLA_STATIONS.loc[_gesla_index, ["perc_analysed"]] = len(df.surge) / len(df.slevel) * 100
                GESLA_STATIONS.loc[_gesla_index, ["analysis"]] = "Success"
                GESLA_STATIONS.loc[_gesla_index, ["analysed"]] = True
                with open(f"./data/processed/{_gesla_file}.json", "w") as fp:
                    json.dump(
                        {
                            "const": ASTRO_WRITE,
                            "amps": demean_amps_phases(coefs, ASTRO_WRITE)[0].tolist(),
                            "phases": demean_amps_phases(coefs, ASTRO_WRITE)[1].tolist(),
                        },
                        fp,
                    )
                out = pd.DataFrame(data=df.surge, index=df.index)
                out.to_parquet(f"./data/processed/{_gesla_file}_surge.gzip", compression="gzip")
                # plots
                plot_multiyear_tide_analysis(ASTRO_PLOT, coefs, years, lat, lon, df, title, filenameOut)
                plot_multiyear_tide_analysis(
                    ASTRO_PLOT, coefs, years, lat, lon, df, title, filenameOut, zoom=True
                )

        except Exception as e:
            print("Error: ", e)
            GESLA_STATIONS.loc[_gesla_index, ["first_obs"]] = (
                pd.Timestamp(df0.index.min()).strftime("%Y-%m-%dT%H:%M:%S"),
            )
            GESLA_STATIONS.loc[_gesla_index, ["last_obs"]] = (
                pd.Timestamp(df0.index.max()).strftime("%Y-%m-%dT%H:%M:%S"),
            )
            GESLA_STATIONS.loc[_gesla_index, ["analysed"]] = False
        GESLA_STATIONS.to_parquet("GESLA_STATIONS_api_analysed.gzip")

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ~~~~ Jenkins' success message ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n\nMy work is done\n\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
