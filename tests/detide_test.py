import pandas as pd
import pytest

from analysea.tide import detide
from analysea.tide import tide_analysis
from analysea.utils import cleanup
from analysea.utils import interpolate
from analysea.utils import resample

OPTS = {  # tidal analysis options
    "constit": "auto",
    "method": "ols",  # ols is faster and good for missing data (Ponchaut et al., 2001)
    "order_constit": "frequency",
    "Rayleigh_min": 0.97,
    "lat": 16,
    "verbose": True,
}  # careful if there is only one Nan parameter, the analysis crashes


STATIONS = pytest.mark.parametrize(
    "station",
    [
        pytest.param("tests/data/abas.csv", id="abas"),
        pytest.param("tests/data/abed.csv", id="abed"),
        pytest.param("tests/data/abur.csv", id="abur"),
        pytest.param("tests/data/acap2.csv", id="acap2"),
        pytest.param("tests/data/acnj.csv", id="acnj"),
    ],
)


@STATIONS
def test_detide(station):
    raw = pd.read_csv(station, index_col=0, parse_dates=True)
    clean = cleanup(raw)
    respl = resample(raw)
    interp = interpolate(respl)
    for sensor in clean.columns:
        signal = interp[sensor]
        detided = detide(signal, **OPTS)  # acapulco latitude
        detided.describe()
        assert len(detided) > 0


@STATIONS
def test_tide_coef(station):
    df = pd.read_csv(station, index_col=0, parse_dates=True)
    clean = cleanup(df)
    for sensor in clean.columns:
        ts = clean[sensor]
        ta = tide_analysis(ts, **OPTS)
        tide = ta.tide
        surge = ta.surge

    assert isinstance(tide, pd.DataFrame)
    assert isinstance(surge, pd.DataFrame)
    assert not df.empty
