import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TABLE_KEYS = ("table", "event_prices", "summary")


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _map(v: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(v, dict):
        return {k: None if x is None else fn(x) for k, x in v.items()}
    if isinstance(v, list):
        return [None if x is None else fn(x) for x in v]
    return None if v is None else fn(v)


def decode_value(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v) == {"date"}:
            return _map(v["date"], _parse_date)
        if "instant" in v and set(v) <= {"instant", "tz"}:
            tz = v.get("tz") or "UTC"
            return _map(v["instant"], lambda s: pd.Timestamp(s).tz_convert(tz))
        return {k: decode_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode_value(x) for x in v]
    return v


def decode_column(values: list, kind: str, tz: str | None = None) -> pd.Series:
    has_null = any(v is None for v in values)
    match kind:
        case "date":
            return pd.Series([None if v is None else _parse_date(v) for v in values], dtype=object)
        case "instant":
            s = pd.to_datetime(pd.Series(values, dtype=object), utc=True, format="ISO8601")
            return s.dt.as_unit("ns").dt.tz_convert(tz or "UTC")
        case "double":
            return pd.Series([np.nan if v is None else v for v in values], dtype="float64")
        case "integer":
            if has_null:
                return pd.Series([np.nan if v is None else v for v in values], dtype="float64")
            return pd.Series(values, dtype="int64")
        case "logical":
            return pd.Series(values, dtype=object if has_null else "bool")
        case "character":
            return pd.Series(values, dtype=object)
    raise ValueError(f"Unknown column type {kind!r}.")


def decode_table(enc: dict) -> pd.DataFrame:
    cols = {
        name: decode_column(values, enc["types"][name], enc["tz"].get(name))
        for name, values in enc["columns"].items()
    }
    df = pd.DataFrame(cols, index=pd.RangeIndex(enc["nrow"]))
    if len(df) != enc["nrow"]:
        raise ValueError(f"Decoded {len(df)} rows, fixture says {enc['nrow']}.")
    df.attrs = {k: decode_value(v) for k, v in enc["attrs"].items()}
    df.attrs["r_class"] = enc["class"]
    return df


def read_fixture(name: str) -> dict:
    raw = json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    fx = dict(raw)
    fx["args"] = decode_value(raw["args"])
    inp = dict(raw["input"])
    if inp["kind"] == "rows":
        inp["data"] = decode_table(inp["data"])
        inp["as_event_prices_args"] = decode_value(inp["as_event_prices_args"])
    fx["input"] = inp
    fx["output"] = {
        k: decode_table(v) if k in TABLE_KEYS else v for k, v in raw["output"].items()
    }
    return fx



@pytest.fixture
def load_fixture() -> Callable[[str], dict]:
    return read_fixture


@pytest.fixture
def brexit_horizons() -> dict[str, dt.date]:
    return {"1W": dt.date(2016, 5, 31), "2W": dt.date(2016, 6, 7), "1M": dt.date(2016, 6, 23)}


@pytest.fixture
def us_horizons() -> dict[str, dt.date]:
    return {"1W": dt.date(2016, 10, 17), "2W": dt.date(2016, 10, 24), "1M": dt.date(2016, 11, 8)}


@pytest.fixture
def clock_value() -> Callable[[pd.DataFrame, str, str], float]:
    def _value(res: pd.DataFrame, horizon: str, method: str) -> float:
        sel = res.loc[(res["horizon"] == horizon) & (res["method"] == method), "A"]
        return float(sel.to_numpy().item())

    return _value