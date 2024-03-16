from __future__ import annotations

import datetime
from typing import Any
from typing import Dict
from typing import TypedDict
from typing import TypeVar
from typing import Union

import numpy.typing as npt
import pandas as pd
from typing_extensions import NotRequired
from typing_extensions import TypeAlias  # "from typing" in Python 3.9+


StrDict: TypeAlias = Dict[str, Any]
DateTimeLike: TypeAlias = Union[str, datetime.date, datetime.datetime, pd.Timestamp]

ScalarOrArray = TypeVar("ScalarOrArray", int, float, npt.NDArray[Any])


class UTideArgs(TypedDict):
    constit: NotRequired[str]
    method: NotRequired[str]
    order_constit: NotRequired[str]
    Rayleigh_min: NotRequired[float]
    lat: str
    verbose: NotRequired[str]
