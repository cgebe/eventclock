import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._utils import (
    TimeKind,
    align_bound,
    as_instant,
    as_rng,
    clip_q,
    ec_logit,
    ec_warn,
    is_date,
    is_instant,
    r_format_time,
    r_mad,
)
from .event_prices import EventPrices, as_event_prices
from .kernels import (
    kernel_bipower,
    kernel_drop_largest,
    kernel_rv,
    kernel_rv_boot,
    kernel_rv_se,
    kernel_truncated,
)
from .params import ec_default_params

DEFAULTS = ec_default_params()
METHODS: tuple[str, ...] = DEFAULTS["methods"]
SE_METHODS = ("quarticity", "bootstrap")


@dataclass(frozen=True)
class Increments:
    dL: np.ndarray
    n_obs: int
    n_gaps: int
    max_gap_days: float


def _bound(x: EventPrices, bound: Any, side: str) -> Any:
    aligned = align_bound(bound, x.time_kind, x.tz, side)
    return pd.Timestamp(aligned) if is_date(aligned) else aligned


def window_rows(x: EventPrices, from_: Any, to: Any, warn_na: bool = True) -> pd.DataFrame:
    lo = _bound(x, from_, "start")
    hi = _bound(x, to, "end")
    in_win = (x.time >= lo) & (x.time <= hi)
    na_q = x.data["q"].isna()
    n_na = int((in_win & na_q).sum())
    if warn_na and n_na > 0:
        ec_warn(
            f"{n_na} missing observation{'s' if n_na != 1 else ''} inside the window skipped; "
            "adjacent increments bridge these gaps "
            "(see the Missing observations section of `?event_clock`)."
        )
    return x.data[in_win & ~na_q].reset_index(drop=True)


def gap_stats(time: pd.Series) -> tuple[int, float]:
    if len(time) < 2:
        return 0, np.nan
    days = (time.diff().iloc[1:] / pd.Timedelta(days=1)).to_numpy(dtype="float64")
    return int(np.sum(days > 1.5 * np.median(days))), float(np.max(days))


def window_increments(
    x: EventPrices, from_: Any, to: Any, sample_every: int, clip: tuple[float, float]
) -> Increments:
    d = window_rows(x, from_, to)
    n_obs = len(d)
    empty = Increments(np.empty(0), n_obs, 0, np.nan)
    if n_obs < 2:
        return empty
    d = d.iloc[::sample_every]
    if len(d) < 2:
        return empty
    L = ec_logit(clip_q(d["q"].to_numpy(dtype="float64"), clip))
    n_gaps, max_gap_days = gap_stats(d["time"])
    return Increments(np.diff(L), n_obs, n_gaps, max_gap_days)


def native_time(ts: pd.Timestamp, time_kind: TimeKind) -> Any:
    return ts.date() if time_kind == "date" else ts


def column_time(v: Any) -> pd.Timestamp:
    return pd.Timestamp(v).as_unit("ns") if is_date(v) else as_instant(v)


def time_kind_of(values: list, name: str) -> TimeKind:
    if values and all(is_date(v) for v in values):
        return "date"
    if values and all(is_instant(v) for v in values):
        if len({str(as_instant(v).tz) for v in values}) > 1:
            raise ValueError(f"`{name}` must use one time zone.")
        return "instant"
    raise ValueError(f"`{name}` must contain dates or datetimes of one kind.")


def split_labeled(v: Any, name: str) -> tuple[list[str], list]:
    if isinstance(v, dict):
        labels, values = [str(k) for k in v], list(v.values())
    elif isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        labels, values = [""] * len(v), list(v)
    else:
        labels, values = [""], [v]
    if not values:
        return [], []
    fmt = r_format_time(values, time_kind_of(values, name))
    return [lab if lab != "" else f for lab, f in zip(labels, fmt)], values


def match_methods(methods: str | tuple[str, ...] | list[str]) -> list[str]:
    ms = [methods] if isinstance(methods, str) else list(methods)
    if not ms:
        raise ValueError("`methods` must contain at least one method.")
    bad = [m for m in ms if m not in METHODS]
    if bad:
        raise ValueError(
            f"`methods` must be among {', '.join(METHODS)}; unknown: {', '.join(map(str, bad))}."
        )
    return ms


def check_sample_every(sample_every: Any) -> int:
    if not sample_every >= 1:
        raise ValueError("`sample_every` must be at least 1.")
    return int(sample_every)


def _estimate(
    method: str, dL: np.ndarray, trunc_sd: float, scale_fn: Callable[[np.ndarray], float]
) -> float:
    match method:
        case "rv":
            return kernel_rv(dL)
        case "truncated":
            return kernel_truncated(dL, trunc_sd=trunc_sd, scale_fn=scale_fn)
        case "bipower":
            return kernel_bipower(dL)
        case "largest1":
            return kernel_drop_largest(dL, 1)
        case "largest2":
            return kernel_drop_largest(dL, 2)
    raise ValueError(f"Unknown method {method!r}.")


def _rv_inference(
    dL: np.ndarray,
    A: float,
    conf: float,
    se_method: str,
    boot_reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    z = norm.ppf(1 - (1 - conf) / 2)
    if se_method == "quarticity":
        s = kernel_rv_se(dL)
        return s, float(np.exp(np.log(A) - z * s / A)), float(np.exp(np.log(A) + z * s / A))
    rv_star = kernel_rv_boot(dL, reps=boot_reps, rng=rng)
    s = float(np.std(rv_star, ddof=1)) if len(rv_star) > 1 else np.nan
    lo, hi = np.quantile(rv_star, [(1 - conf) / 2, 1 - (1 - conf) / 2], method="linear")
    return s, float(lo), float(hi)


CLOCK_COLUMNS = [
    "market_id", "from", "to", "horizon", "n_obs", "n_incr",
    "n_gaps", "max_gap_days", "method", "A",
]


def event_clock(
    x: Any,
    from_: Any = None,
    to: Any = None,
    methods: str | tuple[str, ...] | list[str] = METHODS,
    sample_every: int = DEFAULTS["sample_every"],
    trunc_sd: float = DEFAULTS["trunc_sd"],
    scale_fn: Callable[[np.ndarray], float] = r_mad,
    clip: tuple[float, float] | None = None,
    se: bool = False,
    conf: float = 0.95,
    se_method: str = "quarticity",
    boot_reps: int = 999,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    x = as_event_prices(x)
    clip = clip if clip is not None else x.clip
    methods = match_methods(methods)
    if se_method not in SE_METHODS:
        raise ValueError(
            f'`se_method` must be one of "quarticity" or "bootstrap", not "{se_method}".'
        )
    sample_every = check_sample_every(sample_every)
    if not (conf > 0 and conf < 1):
        raise ValueError("`conf` must lie strictly inside (0, 1).")
    gen = as_rng(rng)

    if from_ is None:
        from_ = native_time(x.time.min(), x.time_kind)
    if to is None:
        to = native_time(x.time.max(), x.time_kind)
    labels, to_values = split_labeled(to, "to")

    cols: dict[str, list] = {c: [] for c in CLOCK_COLUMNS}
    if se:
        cols |= {"se": [], "ci_lo": [], "ci_hi": []}
    irv = [i for i, m in enumerate(methods) if m == "rv"]

    for label, to_i in zip(labels, to_values):
        w = window_increments(x, from_, to_i, sample_every, clip)
        A = [
            np.nan if len(w.dL) == 0 else _estimate(m, w.dL, trunc_sd, scale_fn)
            for m in methods
        ]
        k = len(methods)
        cols["market_id"] += [x.market_id] * k
        cols["from"] += [column_time(from_)] * k
        cols["to"] += [column_time(to_i)] * k
        cols["horizon"] += [label] * k
        cols["n_obs"] += [w.n_obs] * k
        cols["n_incr"] += [len(w.dL)] * k
        cols["n_gaps"] += [w.n_gaps] * k
        cols["max_gap_days"] += [w.max_gap_days] * k
        cols["method"] += methods
        cols["A"] += A
        if se:
            se_v, lo_v, hi_v = [np.nan] * k, [np.nan] * k, [np.nan] * k
            if len(irv) == 1 and len(w.dL) >= 2 and A[irv[0]] > 0:
                j = irv[0]
                se_v[j], lo_v[j], hi_v[j] = _rv_inference(
                    w.dL, A[j], conf, se_method, boot_reps, gen
                )
            cols["se"] += se_v
            cols["ci_lo"] += lo_v
            cols["ci_hi"] += hi_v

    out = pd.DataFrame(cols)
    for c in ("n_obs", "n_incr", "n_gaps"):
        out[c] = out[c].astype("int64")
    for c in ("max_gap_days", "A", "se", "ci_lo", "ci_hi"):
        if c in out:
            out[c] = out[c].astype("float64")
    return out

