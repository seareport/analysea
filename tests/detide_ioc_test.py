import json
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
from analysea.tide import demean_amps_phases
from analysea.tide import yearly_tide_analysis
from analysea.utils import calculate_completeness
from analysea.utils import correct_unit
from analysea.utils import detect_gaps

# ===================
# global variables
# ===================
# load in data information
IOC_STATIONS = ioc.get_ioc_stations()

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
    for _ioc_file in sorted(os.listdir(os.path.join(os.getcwd(), "data"))):
        print("   > _ioc_file = ", _ioc_file)
        # here is the IOC case
        _ioc_code = os.path.splitext(_ioc_file)[0]
        _ioc_index = np.where(IOC_STATIONS.ioc_code == _ioc_code)[0][0]
        df0 = pd.read_parquet(os.path.join(os.getcwd(), "data", _ioc_code + ".gzip"))
        # fill already some info
        lat = IOC_STATIONS.iloc[_ioc_index].lat
        lon = IOC_STATIONS.iloc[_ioc_index].lon
        #
        resout = {
            "lat": lat,
            "lon": lon,
            "const_name": None,
            "means_amps": None,
            "mean_phases": None,
            "missing": None,
            "perc_analysed": None,
            "biggest_gap": None,
            "total_gap": None,
            "steps": None,
            "unit_flag": None,
            "first_obs": pd.Timestamp(df0.index.min()).strftime("%m-%d-%YT%H:%M:%S"),
            "last_obs": pd.Timestamp(df0.index.max()).strftime("%m-%d-%YT%H:%M:%S"),
            "code": _ioc_code,
            "analysis": "",
        }
        df = pd.DataFrame()
        df["slevel"] = remove_numerical(df0.slevel)
        df["correct"], units_flag = correct_unit(df.slevel)
        df["step"], stepsx, steps = step_function_ruptures(df.correct)
        if (len(stepsx) > 2) & (
            np.max(steps) > 1.0
        ):  # if step are not bigger than 1 meter
            df.correct = df.slevel - df.step
        ipeaks, peaks, df["anomaly"] = despike_prominence(
            df.correct, df.correct.std() * 3
        )
        # calculate completeness
        if calculate_completeness(df.anomaly) < 70:
            resout["analysis"] = "Failed: missing data"
            resout["missing"] = 100 - np.round(calculate_completeness(df.anomaly), 2)
            continue
        else:
            try:
                gaps, small_gaps, big_gaps = detect_gaps(df)
                filenameOutGaps = os.path.join(
                    os.getcwd(), "graphs", _ioc_code + "_gaps.png"
                )
                plot_gaps(df.anomaly.interpolate(), big_gaps, filenameOutGaps)
                # assign parameters for tide analysis
                if signaltonoise(df.slevel) < 0:
                    df["filtered"] = filter_fft(df.anomaly)
                    df.anomaly = df.filtered
                # assign parameters before running tides (eventual crash)
                resout["missing"] = 100 - np.round(
                    calculate_completeness(df.anomaly), 2
                )
                resout["biggest_gap"] = str(big_gaps.max())
                resout["total_gap"] = len(gaps)
                resout["steps"] = len(stepsx)
                resout["snr"] = (signaltonoise(df.slevel),)
                resout["unit_flag"] = units_flag
                #
                OPTS["lat"] = lat
                filenameOut = os.path.join(os.getcwd(), "graphs", _ioc_code + ".png")
                df["tide"], df["surge"], coefs, years = yearly_tide_analysis(
                    df.anomaly, 365, OPTS
                )
                # save figures
                if len(coefs) == 0:
                    resout["analysis"] = "Failed: missing data"
                    resout["missing"] = 100 - np.round(
                        calculate_completeness(df.anomaly), 2
                    )
                    continue
                title = IOC_STATIONS.iloc[_ioc_index].Location
                plot_multiyear_tide_analysis(
                    ASTRO_PLOT, coefs, years, lat, lon, df, title, filenameOut
                )
                plot_multiyear_tide_analysis(
                    ASTRO_PLOT,
                    coefs,
                    years,
                    lat,
                    lon,
                    df,
                    title,
                    filenameOut,
                    zoom=True,
                )

                # printing out results
                resout["biggest_gap"] = str(big_gaps.max())
                resout["perc_analysed"] = len(df.surge) / len(df.slevel) * 100
                resout["means_amps"], resout["mean_phases"] = demean_amps_phases(
                    coefs, ASTRO_WRITE
                )
                resout["means_amps"] = resout["means_amps"].tolist()
                resout["mean_phases"] = resout["mean_phases"].tolist()
                resout["const_name"] = ASTRO_WRITE
                resout["analysis"] = "Success"
                with open(f"./data/processed/{_ioc_code}.json", "w") as fp:
                    json.dump(resout, fp)
                out = pd.DataFrame(data=df.surge, index=df.index)
                out.to_parquet(
                    f"./data/processed/{_ioc_code}_surge.gzip", compression="gzip"
                )

                IOC_STATIONS.iloc[_ioc_index]["missing"] = resout["missing"]
                IOC_STATIONS.iloc[_ioc_index]["first_obs"] = resout["first_obs"]
                IOC_STATIONS.iloc[_ioc_index]["last_obs"] = resout["last_obs"]
                IOC_STATIONS.iloc[_ioc_index]["biggest_gap"] = resout["biggest_gap"]
                IOC_STATIONS.iloc[_ioc_index]["analysed"] = True
            except Exception as e:
                print(e)
                IOC_STATIONS.iloc[_ioc_index]["missing"] = resout["missing"]
                IOC_STATIONS.iloc[_ioc_index]["first_obs"] = resout["first_obs"]
                IOC_STATIONS.iloc[_ioc_index]["last_obs"] = resout["last_obs"]
                IOC_STATIONS.iloc[_ioc_index]["biggest_gap"] = resout["biggest_gap"]
                IOC_STATIONS.iloc[_ioc_index]["analysed"] = False

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ~~~~ Jenkins' success message ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\n\nMy work is done\n\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
