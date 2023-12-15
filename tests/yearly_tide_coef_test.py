import observer
import pandas as pd

from analysea.tide import yearly_tide_analysis
from analysea.utils import cleanup

# UTIDE analysis options
OPTS = {
    "conf_int": "linear",
    "constit": "auto",
    "method": "ols",  # ols is faster and good for missing data (Ponchaut et al., 2001)
    "order_constit": "frequency",
    "Rayleigh_min": 0.97,
    "lat": None,
    "verbose": False,
}  # careful if there is only one Nan parameter, the analysis crashes


def yearly_tide_test():
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2023-12-31")
    #
    raw = observer.scrape_ioc_station(ioc_code="acap2", start_date=start, end_date=end)

    clean = cleanup(raw)

    tide, surge, coefs, years = yearly_tide_analysis(clean, lat=16, verbose=False)  # acapulco latitude
    print(tide.head(), surge.head(), coefs, years)


yearly_tide_test()
