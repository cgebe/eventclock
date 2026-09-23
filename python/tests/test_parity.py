from pathlib import Path

import pytest

SOURCE_SHA = "30781dd48a691dbdb253b6c10463e3b2d1731aed"
FIXTURES = sorted(p.stem for p in (Path(__file__).parent / "fixtures").glob("*.json"))


def test_fixture_set_is_complete():
    assert len(FIXTURES) == 30


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_decodes(load_fixture, name):
    fx = load_fixture(name)
    assert fx["name"] == name
    assert fx["source_sha"] == SOURCE_SHA
    assert any(k in fx["output"] for k in ("table", "event_prices", "summary"))