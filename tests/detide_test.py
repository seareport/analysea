import pandas as pd
import pytest

from analysea.tide import detide
from analysea.tide import tide_analysis
from analysea.tide import yearly_tide_analysis
from analysea.utils import cleanup

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
        pytest.param("tests/data/abed.parquet", id="abed"),
    ],
)


@STATIONS
def test_detide(station):
    raw = pd.read_parquet(station)
    for sensor in raw.columns:
        detided = detide(raw[sensor], resample_detide=True, **OPTS)
        detided.describe()
        assert len(detided) > 0


@STATIONS
def test_tide_coef(station):
    df = pd.read_parquet(station)
    clean = cleanup(df)
    for sensor in clean.columns:
        ts = clean[sensor]
        ta = tide_analysis(ts, resample_detide=True, **OPTS)
        tide = ta.tide
        surge = ta.surge

        assert isinstance(tide, pd.DataFrame)
        assert isinstance(surge, pd.DataFrame)
        assert not df.empty


@STATIONS
def test_tide_multiyear(station):
    df = pd.read_parquet(station)
    clean = cleanup(df)
    for sensor in clean.columns:
        ts = clean[sensor]
        ta = yearly_tide_analysis(ts, **OPTS)
        tide = ta.tide
        surge = ta.surge

    assert isinstance(tide, pd.DataFrame)
    assert isinstance(surge, pd.DataFrame)
    assert not df.empty


@STATIONS
def test_tide_multiyear_on_1year(station):
    df = pd.read_parquet(station)
    clean = cleanup(df)
    for sensor in clean.columns:
        ts = clean[sensor]["2024-06-01":]
        ta = yearly_tide_analysis(ts, **OPTS)
        tide = ta.tide
        surge = ta.surge

    assert isinstance(tide, pd.DataFrame)
    assert isinstance(surge, pd.DataFrame)
    assert not df.empty
