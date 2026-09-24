"""Triple-barrier labelling, matching the backtest engine's bracket exactly.

A label answers: *if the model fires on this bar, you enter at the next bar's
open -- what happens first?* One of three things, the same three barriers a
``Target(stop, take, max_bars)`` attaches in :mod:`trader.backtest`:

* the **take** barrier at ``entry * (1 + take)`` -> label ``+1``
* the **stop** barrier at ``entry * (1 - stop)`` -> label ``-1``
* neither, within ``max_bars`` bars -> the **vertical** barrier: label
  ``sign(return)``, or ``0`` when ``|return| <= min_return``

The engine's tie rule is reproduced: when a single bar's range spans both the
stop and the take, the **stop** is taken (real fills are path-dependent and
unknowable from OHLC; assuming the adverse touch came first biases labels the
safe way). A bar that *opens* beyond a barrier -- an overnight gap -- exits at
that open instead of the barrier level, and the open decides which barrier it
was, again matching the engine. ``touch_price`` is that fill price.

The result frame is indexed by the **decision bar** (the bar the model sees),
one row per input bar, so it aligns 1:1 with a feature frame on ``time``. The
trailing ``max_bars + 1`` rows have no room for a full forward window and get a
NaN label, dropped only at the dataset boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "BarrierParams",
    "LabelConfig",
    "barrier_params_from_target",
    "target_from_barrier_params",
    "triple_barrier",
]


@dataclass(frozen=True)
class BarrierParams:
    """The three barriers, in the units :class:`trader.strategies.base.Target` uses."""

    stop: float | None
    take: float | None
    max_bars: int

    def __post_init__(self) -> None:
        if self.max_bars < 1:
            raise ValueError(f"max_bars must be >= 1, got {self.max_bars}")
        for name in ("stop", "take"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive fraction or None, got {value}")


@dataclass(frozen=True)
class LabelConfig(BarrierParams):
    """:class:`BarrierParams` plus the choices only labelling needs."""

    min_return: float = 0.0
    entry: Literal["next_open", "close"] = "next_open"


def barrier_params_from_target(target: object) -> BarrierParams:
    """Pull the barriers off a :class:`~trader.strategies.base.Target`.

    Raises:
        ValueError: the target has no ``max_bars`` -- triple-barrier labelling
            needs a vertical barrier.
    """
    max_bars = getattr(target, "max_bars", None)
    if max_bars is None:
        raise ValueError("Target has no max_bars; triple-barrier labelling needs one")
    return BarrierParams(
        stop=getattr(target, "stop", None),
        take=getattr(target, "take", None),
        max_bars=int(max_bars),
    )


def target_from_barrier_params(params: BarrierParams | dict) -> dict:
    """Barrier kwargs for ``Target(weight, **kwargs)``."""
    if isinstance(params, BarrierParams):
        return {"stop": params.stop, "take": params.take, "max_bars": params.max_bars}
    return {
        "stop": params.get("stop"),
        "take": params.get("take"),
        "max_bars": int(params["max_bars"]),
    }


_COLUMNS = (
    "label",
    "entry_time",
    "entry_price",
    "touch_time",
    "touch_price",
    "barrier",
    "bars_held",
    "ret",
)


def triple_barrier(
    frame: pd.DataFrame,
    *,
    stop: float | None,
    take: float | None,
    max_bars: int,
    min_return: float = 0.0,
    entry: Literal["next_open", "close"] = "next_open",
) -> pd.DataFrame:
    """Label every bar of ``frame`` by which barrier its forward window hits first.

    Args:
        frame: a canonical bar frame, oldest first.
        stop: downside barrier as a positive fraction of the entry price, or None.
        take: upside barrier, same units, or None.
        max_bars: vertical barrier -- give up after this many bars from entry.
        min_return: timeout deadband; ``|ret| <= min_return`` labels ``0``.
        entry: ``"next_open"`` (fill at the next bar's open, matching the engine)
            or ``"close"`` (fill at the decision bar's close).

    Returns:
        A frame indexed by the decision bar's ``time``, one row per input bar,
        with columns ``label`` (in ``{-1.0, 0.0, +1.0}`` or NaN), ``entry_time``,
        ``entry_price``, ``touch_time``, ``touch_price``, ``barrier``
        (``"stop"`` | ``"take"`` | ``"time"`` | None), ``bars_held``, ``ret``.
    """
    # Validate through the dataclass.
    BarrierParams(stop=stop, take=take, max_bars=max_bars)

    n = len(frame)
    times = frame["time"].reset_index(drop=True)
    if n < 2:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in _COLUMNS},
            index=pd.DatetimeIndex(times, name="time"),
        )

    idx = np.arange(n)
    op = frame["open"].to_numpy(dtype=float)
    hi = frame["high"].to_numpy(dtype=float)
    lo = frame["low"].to_numpy(dtype=float)
    cl = frame["close"].to_numpy(dtype=float)

    if entry == "next_open":
        entry_price = np.full(n, np.nan)
        entry_price[:-1] = op[1:]
        entry_offset = 1
        entry_time = times.shift(-1)
    elif entry == "close":
        entry_price = cl.copy()
        entry_offset = 0
        entry_time = times.copy()
    else:  # pragma: no cover - guarded by the type
        raise ValueError("entry must be 'next_open' or 'close'")

    up = entry_price * (1.0 + take) if take is not None else np.full(n, np.inf)
    dn = entry_price * (1.0 - stop) if stop is not None else np.full(n, -np.inf)
    finite_entry = np.isfinite(entry_price)

    sentinel = n + max_bars + 10
    first_k = np.full(n, sentinel)
    kind = np.zeros(n, dtype=int)  # +1 take, -1 stop
    fill = np.full(n, np.nan)  # the price the barrier exit actually fills at

    for k in range(1, max_bars + 1):
        bar = idx + k
        valid = bar < n
        safe = np.where(valid, bar, 0)
        bar_op = np.where(valid, op[safe], np.nan)
        bar_hi = np.where(valid, hi[safe], np.nan)
        bar_lo = np.where(valid, lo[safe], np.nan)
        live = valid & finite_entry
        # a bar that opens beyond a barrier (an overnight gap) exits at that
        # open, and the open -- first in time -- decides which barrier it was
        gap_stop = live & (bar_op <= dn)
        gap_take = live & ~gap_stop & (bar_op >= up)
        hit_stop = live & (bar_lo <= dn)
        hit_take = live & (bar_hi >= up)
        newly = (first_k == sentinel) & (hit_stop | hit_take)
        this_kind = np.where(
            gap_stop,
            -1,
            np.where(gap_take, 1, np.where(hit_stop, -1, np.where(hit_take, 1, 0))),
        )  # without a gap, the stop wins ties
        this_fill = np.where(gap_stop | gap_take, bar_op, np.where(this_kind > 0, up, dn))
        first_k = np.where(newly, k, first_k)
        kind = np.where(newly, this_kind, kind)
        fill = np.where(newly, this_fill, fill)

    touched = first_k <= max_bars
    timeout_exit = idx + entry_offset + max_bars
    has_timeout = finite_entry & (timeout_exit < n)

    label = np.full(n, np.nan)
    barrier = np.full(n, None, dtype=object)
    touch_pos = np.full(n, -1)
    touch_price = np.full(n, np.nan)
    bars_held = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    if touched.any():
        tk = first_k[touched]
        signed = kind[touched]
        label[touched] = signed.astype(float)
        barrier[touched] = np.where(signed > 0, "take", "stop")
        touch_pos[touched] = idx[touched] + tk
        touch_price[touched] = fill[touched]
        bars_held[touched] = tk
        ret[touched] = touch_price[touched] / entry_price[touched] - 1.0

    timed = (~touched) & has_timeout
    if timed.any():
        exit_pos = timeout_exit[timed]
        exit_price = op[exit_pos] if entry == "next_open" else cl[exit_pos]
        r = exit_price / entry_price[timed] - 1.0
        lab = np.sign(r)
        lab[np.abs(r) <= min_return] = 0.0
        label[timed] = lab
        barrier[timed] = "time"
        touch_pos[timed] = exit_pos
        touch_price[timed] = exit_price
        bars_held[timed] = max_bars
        ret[timed] = r

    touch_time = pd.Series(pd.NaT, index=range(n), dtype=times.dtype)
    resolved = touch_pos >= 0
    if resolved.any():
        touch_time.iloc[resolved] = times.to_numpy()[touch_pos[resolved]]

    return pd.DataFrame(
        {
            "label": label,
            "entry_time": entry_time.to_numpy(),
            "entry_price": entry_price,
            "touch_time": touch_time.to_numpy(),
            "touch_price": touch_price,
            "barrier": barrier,
            "bars_held": bars_held,
            "ret": ret,
        },
        index=pd.DatetimeIndex(times, name="time"),
    )
