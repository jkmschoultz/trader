"""Turn a :class:`BacktestSpec` into a :class:`~trader.backtest.result.BacktestResult`.

This is the body of the old ``trader backtest`` command with the argparse and
the ``print`` calls taken out, so the API can run exactly the same thing. It
reads bars from the lake, resolves any symbols that were not given as a Uic
(the only step that needs the network), infers a session calendar per
instrument, and calls :func:`trader.backtest.run`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.inputs import parse_since

__all__ = ["BacktestSpec", "run_backtest"]


class BacktestSpec(BaseModel):
    """Everything needed to run one backtest.

    ``symbols`` and ``uics`` pair up positionally: give a Uic to skip the
    symbol-resolution network call for that instrument. ``horizon`` accepts a
    label (``"5m"``) or minutes; ``since`` accepts ``"90d"``, ``"2y"``, a date,
    an ISO timestamp, or ``"all"``.
    """

    symbols: list[str] = Field(min_length=1)
    strategy: str
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    # Preferred exchange suffix (e.g. "xnas") for resolving a bare ticker;
    # ignored when the symbol already names a venue or a uic is given.
    exchange: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    horizon: int = 5
    since: datetime | None = None
    allocator: str = "equal-weight"
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    starting_cash: float = 100_000.0
    leverage: float = 1.0

    @field_validator("horizon", mode="before")
    @classmethod
    def _horizon(cls, value: object) -> int:
        from trader.saxo.charts import ChartError, parse_horizon

        try:
            return parse_horizon(value)  # type: ignore[arg-type]
        except ChartError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("since", mode="before")
    @classmethod
    def _since(cls, value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value  # type: ignore[return-value]
        return parse_since(str(value))

    @field_validator("uics")
    @classmethod
    def _uics_fit(cls, value: list[int], info: Any) -> list[int]:
        symbols = info.data.get("symbols") or []
        if len(value) > len(symbols):
            raise ValueError("more uics than symbols")
        return value


async def run_backtest(
    settings: Settings,
    spec: BacktestSpec,
    *,
    progress: Callable[[float], None] | None = None,
):
    """Run ``spec`` and return a :class:`~trader.backtest.result.BacktestResult`.

    Raises:
        InvalidRequest: unknown strategy or allocator, or bad strategy params.
        SeriesNotStored: the lake has no bars for one of the symbols.
        trader.saxo.instruments.InstrumentNotFound: a symbol matched nothing.
    """
    import inspect

    from trader import backtest as bt
    from trader import strategies
    from trader.service._panel import load_panel

    try:
        strategy_cls = strategies.get_strategy(spec.strategy)
    except strategies.UnknownStrategy as exc:
        raise InvalidRequest(str(exc)) from exc

    params = dict(spec.params)
    # A strategy that loads from the model registry (``lstm``) needs to know
    # where it is; the request rarely says, so hand it the configured path.
    accepts = inspect.signature(strategy_cls.__init__).parameters
    if "models_dir" in accepts and "models_dir" not in params:
        params["models_dir"] = str(settings.models_dir)

    try:
        strategy = strategy_cls(**params)
    except strategies.UnknownStrategy as exc:
        raise InvalidRequest(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidRequest(f"bad params for strategy {spec.strategy!r}: {exc}") from exc

    try:
        allocator = bt.get_allocator(spec.allocator)
    except KeyError as exc:
        raise InvalidRequest(exc.args[0]) from exc

    series = await load_panel(
        settings,
        symbols=spec.symbols,
        uics=spec.uics,
        asset_type=spec.asset_type,
        exchange=spec.exchange,
        horizon=spec.horizon,
        since=spec.since,
    )
    panel: dict[str, Any] = {s.label: s.frame for s in series}
    instruments: dict[str, Any] = {
        s.label: bt.Instrument(key=s.key, session=s.session) for s in series
    }

    return bt.run(
        panel,
        strategy,
        horizon=spec.horizon,
        allocator=allocator,
        cost_model=bt.CostModel(
            commission_bps=spec.fee_bps,
            half_spread_bps=spec.spread_bps,
            slippage_bps=spec.slippage_bps,
        ),
        instruments=instruments,
        starting_cash=spec.starting_cash,
        leverage_cap=spec.leverage,
        progress=progress,
    )
