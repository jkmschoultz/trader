"""Turn one instrument's bars into windowed sequences with triple-barrier labels.

The no-leakage rule, stated for the arrays this produces: ``X[k]`` is the
``window`` feature rows ending at decision bar ``i``; ``y[k]`` is that bar's
triple-barrier label, which describes a trade entered at bar ``i + 1``'s open --
strictly after every row in ``X[k]``. Feature warmup NaNs and the trailing
unlabelled rows are dropped here, nowhere earlier.

``time_split`` then carves train / val / test by wall-clock date, **purging**
training samples whose label window reaches past the boundary and **embargoing**
a band just before each boundary, so no training label overlaps a validation or
test bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from trader.features.base import FeatureSpec
from trader.labels.triple_barrier import LabelConfig

__all__ = [
    "SequenceBundle",
    "SplitSpec",
    "WindowSpec",
    "build_bundle",
    "time_split",
    "walk_forward_splits",
]

_NS_PER_MIN = 60 * 1_000_000_000


def _epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Nanoseconds since the epoch, regardless of the index's own resolution."""
    return index.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]").astype("int64")


@dataclass(frozen=True)
class WindowSpec:
    """Everything that decides the shape of a training example."""

    window: int
    feature_set: str
    base_horizon: int
    context_horizons: tuple[int, ...]
    label: LabelConfig

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")


@dataclass
class SequenceBundle:
    """Model-ready arrays for one instrument (or several, concatenated)."""

    X: np.ndarray  # (n, window, F) float32
    y: np.ndarray  # (n,) int64 in {0, 1, 2}   (from {-1, 0, +1})
    w: np.ndarray  # (n,) float32 sample weights
    t: np.ndarray  # (n,) int64 nanoseconds -- decision-bar timestamps
    feature_names: list[str]
    feature_spec: FeatureSpec

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class SplitSpec:
    """Where to cut train / val / test, and how wide the safety band is.

    ``test_end`` bounds the test window on the right -- required for
    walk-forward, where a later fold reuses this fold's test period for
    training. Left ``None`` the test set runs to the end of the data.
    ``train_start`` bounds it on the left, for a rolling (fixed-width) train
    window; ``None`` means train is everything up to ``train_end`` (anchored).
    """

    train_end: datetime
    val_end: datetime
    embargo_bars: int | None = None
    test_end: datetime | None = None
    train_start: datetime | None = None

    def __post_init__(self) -> None:
        if self.val_end <= self.train_end:
            raise ValueError("val_end must be after train_end")
        if self.test_end is not None and self.test_end <= self.val_end:
            raise ValueError("test_end must be after val_end")
        if self.train_start is not None and self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")


def build_bundle(
    bars: pd.DataFrame,
    *,
    spec: WindowSpec,
    session=None,
    scaler=None,
    weights: pd.Series | None = None,
) -> SequenceBundle:
    """Build ``(X, y, w, t)`` for one canonical bar frame.

    Args:
        bars: canonical base-horizon frame, oldest first.
        spec: the window / feature-set / label configuration.
        session: inferred trading hours for the session-relative features.
        scaler: if given, ``transform`` is applied to ``X`` (fit elsewhere, on
            the training split only).
        weights: optional per-decision-bar sample weights, indexed by bar
            ``time``; missing bars and the default are weight 1.

    Raises:
        ValueError: fewer bars than ``window``, or a bad context horizon.
    """
    from trader.features.pipeline import compute_feature_frame

    features, feature_spec = compute_feature_frame(
        bars,
        feature_set=spec.feature_set,
        base_horizon=spec.base_horizon,
        context_horizons=spec.context_horizons,
        session=session,
    )
    labels = _triple_barrier_from(bars, spec.label)

    feat = features.to_numpy(dtype=np.float32)
    n = feat.shape[0]
    window = spec.window
    if n < window + 1:
        raise ValueError(f"need at least window+1 ({window + 1}) bars, got {n}")

    # windows[j] covers feature rows j .. j+window-1 -> it is the window whose
    # decision bar is i = j + window - 1.
    strided = np.lib.stride_tricks.sliding_window_view(feat, window, axis=0)
    strided = np.ascontiguousarray(strided.transpose(0, 2, 1))  # (n-window+1, window, F)

    decision_idx = np.arange(window - 1, n)
    label_raw = labels["label"].to_numpy(dtype=float)[decision_idx]
    times_ns = _epoch_ns(features.index)[decision_idx]

    valid = ~np.isnan(strided).any(axis=(1, 2)) & ~np.isnan(label_raw)
    strided = strided[valid]
    label_raw = label_raw[valid]
    times_ns = times_ns[valid]

    y = (label_raw + 1.0).astype(np.int64)  # {-1,0,+1} -> {0,1,2}

    if weights is not None:
        aligned = weights.reindex(features.index[decision_idx][valid]).to_numpy(dtype=np.float32)
        w = np.where(np.isnan(aligned), 1.0, aligned).astype(np.float32)
    else:
        w = np.ones(len(y), dtype=np.float32)

    if scaler is not None:
        strided = scaler.transform(strided)

    return SequenceBundle(
        X=strided.astype(np.float32),
        y=y,
        w=w,
        t=times_ns.astype(np.int64),
        feature_names=list(features.columns),
        feature_spec=feature_spec,
    )


def _triple_barrier_from(bars: pd.DataFrame, label: LabelConfig) -> pd.DataFrame:
    from trader.labels.triple_barrier import triple_barrier

    return triple_barrier(
        bars,
        stop=label.stop,
        take=label.take,
        max_bars=label.max_bars,
        min_return=label.min_return,
        entry=label.entry,
    )


def time_split(
    bundle: SequenceBundle,
    split: SplitSpec,
    *,
    base_horizon: int,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean ``(train, val, test)`` masks over ``bundle.t``, purged and embargoed.

    * **purge** -- a training (or validation) sample whose label window
      ``[t, t + (1 + max_bars) bars]`` reaches past its boundary is dropped.
    * **embargo** -- samples in the ``embargo_bars`` band immediately before a
      boundary are dropped (default ``window`` is folded in by the caller via
      ``embargo_bars``; here it defaults to ``max_bars``).
    """
    t = bundle.t.astype(np.int64)
    train_end = np.int64(pd.Timestamp(split.train_end).value)
    val_end = np.int64(pd.Timestamp(split.val_end).value)

    embargo_bars = split.embargo_bars if split.embargo_bars is not None else max_bars
    purge_ns = np.int64((1 + max_bars) * base_horizon * _NS_PER_MIN)
    embargo_ns = np.int64(embargo_bars * base_horizon * _NS_PER_MIN)

    in_train = t < train_end
    if split.train_start is not None:
        in_train = in_train & (t >= np.int64(pd.Timestamp(split.train_start).value))
    in_val = (t >= train_end) & (t < val_end)
    in_test = t >= val_end

    train_mask = in_train & (t + purge_ns < train_end) & (t < train_end - embargo_ns)
    val_mask = in_val & (t + purge_ns < val_end) & (t < val_end - embargo_ns)
    test_mask = in_test
    if split.test_end is not None:
        test_end = np.int64(pd.Timestamp(split.test_end).value)
        # right-purge the test set too: a label window that reaches past
        # test_end depends on bars a later fold trains on.
        test_mask = test_mask & (t < test_end) & (t + purge_ns < test_end)

    return train_mask, val_mask, test_mask


def walk_forward_splits(
    t: np.ndarray,
    *,
    n_folds: int,
    train_days: float,
    val_days: float,
    test_days: float,
    embargo_bars: int | None = None,
    mode: Literal["rolling", "anchored"] = "rolling",
    step_days: float | None = None,
) -> list[SplitSpec]:
    """``n_folds`` chronological ``SplitSpec``s over the span of ``t``.

    ``t`` is the sorted int64-ns decision-bar array (``bundle.t``). Each fold is
    ``train_days`` of training, then ``val_days``, then ``test_days``; the origin
    advances by ``step_days`` (default: ``test_days``, so test windows tile the
    tail without overlap). ``mode="rolling"`` keeps the train window a fixed
    width; ``"anchored"`` starts every train window at the first sample
    (expanding). Raises ``ValueError`` if the data span cannot fit ``n_folds``.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    start = pd.Timestamp(int(np.min(t)))
    end = pd.Timestamp(int(np.max(t)))
    step = pd.Timedelta(days=step_days if step_days is not None else test_days)
    train_span = pd.Timedelta(days=train_days)
    val_span = pd.Timedelta(days=val_days)
    test_span = pd.Timedelta(days=test_days)

    splits: list[SplitSpec] = []
    for i in range(n_folds):
        train_end = start + train_span + i * step
        val_end = train_end + val_span
        test_end = val_end + test_span
        if test_end > end:
            raise ValueError(
                f"data spans {start:%Y-%m-%d}..{end:%Y-%m-%d}; not enough for "
                f"{n_folds} folds of {train_days}+{val_days}+{test_days}d "
                f"stepped {step.days}d (fold {i + 1} needs data to {test_end:%Y-%m-%d})"
            )
        train_start = None if mode == "anchored" else train_end - train_span
        splits.append(
            SplitSpec(
                train_end=train_end.to_pydatetime(),
                val_end=val_end.to_pydatetime(),
                embargo_bars=embargo_bars,
                test_end=test_end.to_pydatetime(),
                train_start=train_start.to_pydatetime() if train_start is not None else None,
            )
        )
    return splits
