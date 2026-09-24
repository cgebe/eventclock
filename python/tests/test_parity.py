import pandas as pd
import pytest

from eventclock import as_event_prices, event_clock, load_dataset
from helpers import (
    assert_frame_parity,
    build_input,
    fixture_names,
    py_args,
    r_messages,
    record_all,
    record_warnings,
)

SOURCE_SHA = "30781dd48a691dbdb253b6c10463e3b2d1731aed"
FIXTURES = fixture_names()
ROWS_FIXTURES = ["clock_gaps", "clock_ny_datebounds", "path_ny_datebounds", "validate_constructed"]


def test_fixture_set_is_complete():
    assert len(FIXTURES) == 30


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_decodes(load_fixture, name):
    fx = load_fixture(name)
    assert fx["name"] == name
    assert fx["source_sha"] == SOURCE_SHA
    assert any(k in fx["output"] for k in ("table", "event_prices", "summary"))


@pytest.mark.parametrize("name", ["ep_brexit", "ep_us", "ep_pm2024"])
def test_as_event_prices(load_fixture, name):
    fx = load_fixture(name)
    ep, warns = record_warnings(as_event_prices, load_dataset(fx["input"]["name"]), **fx["args"])
    assert warns == fx["warnings"]
    out = fx["output"]
    exp = out["event_prices"]
    assert_frame_parity(ep.data, exp)
    assert ep.time_kind == out["time_kind"]
    assert ep.clip == tuple(exp.attrs["clip"])
    assert ep.market_id == exp.attrs.get("market_id")
    assert ep.event_date == exp.attrs.get("event_date")
    assert_frame_parity(ep.summary(), out["summary"])


@pytest.mark.parametrize("name", ROWS_FIXTURES)
def test_rows_inputs_build(load_fixture, name):
    fx = load_fixture(name)
    rows = fx["input"]["data"]
    ep, warns = record_warnings(build_input, fx)
    assert warns == []
    assert len(ep) == len(rows)
    is_instant = pd.api.types.is_datetime64_any_dtype(rows["time"].dtype)
    assert ep.time_kind == ("instant" if is_instant else "date")
    assert_frame_parity(ep.data[["time", "q_raw"]].rename(columns={"q_raw": "q"}), rows[["time", "q"]])


CLOCK_FIXTURES = [
    "clock_brexit",
    "clock_us",
    "clock_brexit_se",
    "clock_brexit_k2",
    "clock_brexit_k3",
    "clock_brexit_k5",
    "clock_pm2024",
    "clock_pm2024_datebounds",
    "clock_ny_datebounds",
    "clock_gaps",
]


@pytest.mark.parametrize("name", CLOCK_FIXTURES)
def test_event_clock(load_fixture, name):
    fx = load_fixture(name)
    res, warns, msgs = record_all(event_clock, build_input(fx), **py_args(fx["args"]))
    assert warns == fx["warnings"]
    assert msgs == r_messages(fx)
    assert_frame_parity(res, fx["output"]["table"])
