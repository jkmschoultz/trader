"""Command-line entry point.

    trader auth login|status|logout   # Saxo authentication
    trader account                    # user, accounts, and balances

    trader instruments search KEYWORDS      # find a Uic
    trader instruments show SYMBOL          # full detail for one instrument
    trader exchange [EXCHANGE_ID]           # exchanges and their sessions

    trader data depth SYMBOL          # how much history Saxo serves
    trader data backfill SYMBOL       # fetch bars into the Parquet lake
    trader data coverage              # what the lake holds
    trader data symbols               # fetch Saxo symbols for every stored uic
    trader data sessions SYMBOL       # inferred trading hours and real gaps

    trader backtest --symbol SYMBOL --strategy NAME   # run a strategy over stored bars

    trader features list                    # registered feature sets
    trader features build --symbol SYMBOL   # compute model features from the lake
    trader labels --symbol SYMBOL           # triple-barrier label balance
    trader train --symbol SYMBOL ...        # train an LSTM and register it
    trader tune --symbol SYMBOL --grid K=v1,v2   # walk-forward grid sweep
    trader models list|show                 # browse the model registry

    trader serve                      # run the API + UI (needs the [api] extra)

The ``data``, ``backtest``, ``features``, ``labels``, and ``models`` subcommands
need the optional data dependencies (``pip install -e ".[data]"``); ``train``
also needs ``[model]`` and ``serve`` also needs ``[api]``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from trader.config import Settings, get_settings
from trader.saxo.accounts import get_balance, get_user, list_accounts
from trader.saxo.auth import AuthError, ReauthRequired, SaxoAuth
from trader.saxo.charts import ChartError, parse_horizon
from trader.saxo.client import SaxoAPIError, SaxoClient
from trader.saxo.instruments import (
    AmbiguousInstrument,
    Instrument,
    InstrumentNotFound,
    get_details,
    resolve,
    search,
)
from trader.saxo.tokens import TokenStore


class UsageError(Exception):
    """A bad invocation, reported without a traceback."""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_data_extra():
    """Import the data layer, or explain how to install it.

    ``trader.data`` needs pandas and pyarrow, which are an optional extra so the
    API client stays installable without them. Importing lazily keeps
    ``trader auth`` working on a base install.
    """
    try:
        from trader.data import bars, calendars, depth, ingest, lake  # noqa: PLC0415

        return bars, calendars, depth, ingest, lake
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise UsageError(
            f"the data commands need the optional data dependencies ({exc}). "
            'Install them with:  pip install -e ".[data]"'
        ) from exc


def _require_backtest_extra():
    """Import the backtest engine and strategy registry, or explain the extra.

    Both live on top of :mod:`trader.data` and share its ``[data]`` extra, so a
    base install that cannot import one cannot import the other either.
    """
    try:
        from trader import backtest, strategies  # noqa: PLC0415

        return backtest, strategies
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise UsageError(
            f"the backtest command needs the optional data dependencies ({exc}). "
            'Install them with:  pip install -e ".[data]"'
        ) from exc


def _require_model_extra(model_type: str = "lstm"):
    """Import the model layer, or explain the ``[model]`` extra.

    ``model_type="gbm"`` needs lightgbm; anything else needs torch. sklearn and
    the ``trader.models`` package are needed either way.
    """
    try:
        import sklearn  # noqa: F401, PLC0415

        if model_type == "gbm":
            import lightgbm  # noqa: F401, PLC0415
        else:
            import torch  # noqa: F401, PLC0415

        from trader import models  # noqa: PLC0415

        return models
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise UsageError(
            f"this command needs the optional model dependencies ({exc}). "
            'Install them with:  pip install -e ".[model]"'
        ) from exc


# --------------------------------------------------------------------------- auth


async def _cmd_auth_login(settings: Settings) -> int:
    auth = SaxoAuth(settings)
    try:
        tokens = await auth.login_interactive()
    finally:
        await auth.aclose()
    print(
        f"\nSigned in to {settings.saxo.environment.value}. "
        f"Access token valid until {tokens.access_expires_at:%H:%M:%S UTC}."
    )
    if tokens.refresh_expires_at:
        print(f"Refresh token valid until {tokens.refresh_expires_at:%H:%M:%S UTC}.")
    return 0


def _cmd_auth_status(settings: Settings) -> int:
    if settings.saxo.token_24h.get_secret_value():
        print("Using a developer-portal 24-hour token from TRADER_SAXO__TOKEN_24H (SIM only).")
        return 0

    tokens = TokenStore(settings.token_file, settings.token_key_file).load()
    if tokens is None:
        print(f"No stored tokens for {settings.saxo.environment.value}. Run: trader auth login")
        return 1

    print(f"Environment:   {settings.saxo.environment.value}")
    print(
        f"Access token:  {'EXPIRED' if tokens.access_expired() else 'valid'} "
        f"(until {tokens.access_expires_at:%Y-%m-%d %H:%M:%S UTC})"
    )
    if tokens.is_static:
        print("Refresh token: n/a (static token)")
    else:
        state = "EXPIRED" if tokens.refresh_expired() else "valid"
        until = (
            f" (until {tokens.refresh_expires_at:%Y-%m-%d %H:%M:%S UTC})"
            if tokens.refresh_expires_at
            else ""
        )
        print(f"Refresh token: {state}{until}")
    return 0


def _cmd_auth_logout(settings: Settings) -> int:
    TokenStore(settings.token_file, settings.token_key_file).clear()
    print(f"Cleared stored tokens for {settings.saxo.environment.value}.")
    return 0


async def _cmd_account(settings: Settings) -> int:
    async with SaxoClient(settings) as client:
        user = await get_user(client)
        accounts = await list_accounts(client)

        print(f"User:        {user.name or user.user_id} (UserId {user.user_id})")
        print(f"ClientKey:   {user.client_key}")
        print(f"Environment: {settings.saxo.environment.value}")
        print()

        if not accounts:
            print("No accounts returned.")
            return 1

        for account in accounts:
            balance = await get_balance(
                client, client_key=user.client_key, account_key=account.account_key
            )
            flag = "" if account.active else "  [inactive]"
            print(f"Account {account.account_id} ({account.account_type}){flag}")
            print(f"  AccountKey:       {account.account_key}")
            print(f"  Cash balance:     {balance.cash_balance:,.2f} {balance.currency}")
            print(f"  Total value:      {balance.total_value:,.2f} {balance.currency}")
            print(f"  Margin available: {balance.margin_available:,.2f} {balance.currency}")
    return 0


# -------------------------------------------------------------------- instruments


async def _cmd_instruments_search(settings: Settings, args) -> int:
    async with SaxoClient(settings) as client:
        results = await search(
            client,
            args.keywords,
            asset_types=(args.asset_type,) if args.asset_type else None,
            limit=args.limit,
        )
    if not results:
        print(f"No instruments match {args.keywords!r}.")
        return 1
    width = max(len(i.symbol) for i in results)
    for item in results:
        print(
            f"{item.symbol:<{width}}  uic {item.uic:<9} {item.asset_type:<14} "
            f"{item.exchange_id:<10} {item.description}"
        )
    return 0


async def _cmd_instruments_show(settings: Settings, args) -> int:
    async with SaxoClient(settings) as client:
        instrument = await _resolve(client, args.symbol, args.asset_type)
        details = await get_details(client, instrument.uic, instrument.asset_type)

    print(f"Symbol:      {details.symbol}")
    print(f"Description: {details.description}")
    print(f"Uic:         {details.uic}")
    print(f"AssetType:   {details.asset_type}")
    print(f"Exchange:    {details.exchange_id} ({details.exchange.get('Name', '')})")
    print(f"Currency:    {details.currency}")
    print(f"Tradable:    {details.is_tradable} ({details.trading_status or 'unknown'})")
    print(f"Tick size:   {details.tick_size if details.tick_size is not None else 'scheme-based'}")
    print(f"Lot size:    {details.lot_size}")
    print(f"Min trade:   {details.minimum_trade_size}")
    return 0


async def _cmd_exchange(settings: Settings, args) -> int:
    from trader.data.calendars import SessionCalendar  # noqa: PLC0415
    from trader.saxo.exchanges import get_exchange, list_exchanges  # noqa: PLC0415

    async with SaxoClient(settings) as client:
        if not args.exchange_id:
            exchanges = await list_exchanges(client)
            for item in sorted(exchanges, key=lambda e: e.exchange_id):
                print(f"{item.exchange_id:<16} {item.country_code:<4} {item.name}")
            print(f"\n{len(exchanges)} exchanges.")
            return 0
        exchange = await get_exchange(client, args.exchange_id)

    calendar = SessionCalendar.from_exchange(exchange)
    now = datetime.now(UTC)
    print(f"Exchange:  {exchange.exchange_id} -- {exchange.name}")
    print(f"Country:   {exchange.country_code}   Currency: {exchange.currency}")
    tz_source = f"code {exchange.time_zone_code} ({exchange.time_zone_abbreviation})"
    print(f"Time zone: {tz_source} -> {calendar.timezone}")
    print(f"All day:   {exchange.all_day}")

    state = calendar.state_at(now)
    open_now = calendar.is_open(now)
    print(
        f"Now:       {'OPEN' if open_now else 'closed' if open_now is False else 'unknown'}"
        f" ({state or 'not covered by published sessions'})"
    )
    upcoming = calendar.next_open(now)
    if upcoming:
        print(f"Next open: {upcoming:%Y-%m-%d %H:%M} UTC")

    if calendar.sessions:
        print("\nPublished sessions (UTC):")
        for session in calendar.sessions:
            marker = " <-- now" if session.contains(now) else ""
            print(
                f"  {session.start:%Y-%m-%d %H:%M} -> {session.end:%Y-%m-%d %H:%M}  "
                f"{session.state}{marker}"
            )
        first, last = calendar.covers
        print(
            f"\nThese cover {first:%Y-%m-%d} to {last:%Y-%m-%d} only -- Saxo publishes a "
            "rolling window, not a historical calendar."
        )
    return 0


# --------------------------------------------------------------------------- data


async def _cmd_data_depth(settings: Settings, args) -> int:
    _, _, depth, _, _ = _require_data_extra()

    horizons = (
        tuple(parse_horizon(h) for h in args.horizons.split(","))
        if args.horizons
        else depth.DEFAULT_HORIZONS
    )

    async with SaxoClient(settings) as client:
        instrument = await _resolve(client, args.symbol, args.asset_type)
        report = await depth.probe(
            client,
            uic=instrument.uic,
            asset_type=instrument.asset_type,
            symbol=instrument.symbol,
            horizons=horizons,
            max_pages=args.max_pages,
        )

    print()
    print(depth.render(report))

    out = args.out or settings.state_dir / f"history-depth-{instrument.uic}.json"
    depth.save_report(report, out)
    print(f"\nReport written to {out}")
    return 0


async def _cmd_data_backfill(settings: Settings, args) -> int:
    _, _, _, ingest, lake_mod = _require_data_extra()
    from trader.data.instruments import InstrumentRegistry  # noqa: PLC0415

    horizons = [parse_horizon(h) for h in args.horizon.split(",")]
    since = _parse_since(args.since)

    store = lake_mod.BarLake(settings.data_dir)
    registry = InstrumentRegistry(settings.data_dir)
    async with SaxoClient(settings) as client:
        instrument = await _resolve(client, args.symbol, args.asset_type)
        registry.put(
            instrument.asset_type,
            instrument.uic,
            symbol=instrument.symbol,
            description=instrument.description,
            currency=instrument.currency,
            exchange_id=instrument.exchange_id,
        )
        print(f"{instrument.symbol} -> uic {instrument.uic} ({instrument.asset_type})")
        if since:
            print(f"Fetching back to {since:%Y-%m-%d %H:%M} UTC into {store.root}\n")
        else:
            print(f"Fetching all available history into {store.root}\n")

        for horizon in horizons:
            key = lake_mod.SeriesKey(instrument.asset_type, instrument.uic, horizon)
            result = await ingest.backfill(
                client,
                store,
                key,
                since=since,
                max_bars=args.max_bars,
                max_pages=args.max_pages,
                resume=not args.no_resume,
            )
            print(result)

        print(f"\n{client.request_counts} requests issued per service group.")

    print()
    for horizon in horizons:
        key = lake_mod.SeriesKey(instrument.asset_type, instrument.uic, horizon)
        coverage = store.coverage(key)
        if coverage:
            print(lake_mod.coverage_line(coverage, instrument.symbol))
    return 0


def _cmd_data_coverage(settings: Settings, args) -> int:
    _, _, _, _, lake_mod = _require_data_extra()
    from trader.data.instruments import InstrumentRegistry  # noqa: PLC0415

    store = lake_mod.BarLake(settings.data_dir)
    registry = InstrumentRegistry(settings.data_dir)
    keys = store.series()
    if not keys:
        print(f"No series stored under {store.root}. Run: trader data backfill SYMBOL")
        return 1

    unnamed = {(k.asset_type, k.uic) for k in keys if not registry.symbol_for(k.asset_type, k.uic)}

    print(f"Lake: {store.root}\n")
    for key in keys:
        coverage = store.coverage(key)
        if coverage:
            print(lake_mod.coverage_line(coverage, registry.symbol_for(key.asset_type, key.uic)))
    print(f"\n{len(keys)} series.")
    if unnamed:
        print(
            f"{len(unnamed)} uic(s) have no symbol yet. Run: trader data symbols"
        )
    return 0


async def _cmd_data_symbols(settings: Settings, args) -> int:
    _require_data_extra()
    from trader.data.instruments import InstrumentRegistry  # noqa: PLC0415
    from trader.service.data import refresh_symbols  # noqa: PLC0415

    records = await refresh_symbols(
        settings, refresh=args.refresh, progress=lambda msg: print(f"  fetched {msg}")
    )
    if not records:
        print("The lake is empty. Run: trader data backfill SYMBOL")
        return 1

    width = max((len(r.symbol) for r in records if r.symbol), default=1)
    for record in records:
        symbol = record.symbol or "?"
        updated = f"{record.updated:%Y-%m-%d}" if record.updated else "-"
        print(
            f"{symbol:<{width}}  {record.asset_type:<14} uic {record.uic:<9} "
            f"{record.currency:<4} {updated}  {record.description}"
        )
    print(f"\n{len(records)} instruments in {InstrumentRegistry(settings.data_dir).path}")
    return 0


async def _cmd_data_sessions(settings: Settings, args) -> int:
    bars_mod, calendars, _, _, lake_mod = _require_data_extra()
    from trader.saxo.exchanges import get_exchange  # noqa: PLC0415

    horizon = parse_horizon(args.horizon)
    store = lake_mod.BarLake(settings.data_dir)

    async with SaxoClient(settings) as client:
        instrument = await _resolve(client, args.symbol, args.asset_type)
        key = lake_mod.SeriesKey(instrument.asset_type, instrument.uic, horizon)
        frame = store.read(key)
        if frame.empty:
            print(f"Nothing stored for {key}. Run: trader data backfill {args.symbol}")
            return 1

        zone = UTC
        if instrument.exchange_id:
            try:
                exchange = await get_exchange(client, instrument.exchange_id)
                zone = calendars.resolve_timezone(exchange)
            except SaxoAPIError as exc:
                print(f"warning: could not fetch exchange {instrument.exchange_id}: {exc}")

    print(f"{key}   timezone {zone}\n")
    summary = bars_mod.describe(frame, horizon)
    for name in ("rows", "first", "last", "density", "gaps", "largest_gap_bars", "off_grid"):
        print(f"  {name:<17} {summary[name]}")
    print(f"  {'has_volume':<17} {summary['has_volume']}")
    print(f"  {'has_spread':<17} {summary['has_spread']}")
    print(f"  {'has_trading_state':<17} {summary['has_trading_state']}")

    if summary["has_trading_state"]:
        # The venue's own label, when it reports one, beats guessing hours from
        # timestamps -- print it directly rather than only the inferred version.
        counts = frame["trading_state"].value_counts(dropna=False)
        print("\nMarketTradingState counts (from Saxo, not inferred):")
        for state, count in counts.items():
            print(f"  {state!s:<20} {count:,}")

    hours = calendars.infer_regular_hours(frame, zone)
    if hours is None:
        print("\nToo few trading days stored to infer session hours.")
        return 0

    print(f"\nInferred regular hours: {hours}")
    inside = calendars.filter_regular_hours(frame, hours)
    share = len(inside) / len(frame) if len(frame) else 0.0
    print(f"Bars inside those hours: {len(inside):,} of {len(frame):,} ({share:.1%})")

    real_gaps = calendars.drop_session_breaks(bars_mod.gaps(frame, horizon), hours)
    if not real_gaps:
        print("\nNo gaps inside trading hours -- the stored series is continuous.")
    else:
        print(f"\n{len(real_gaps)} gap(s) inside trading hours:")
        for gap in sorted(real_gaps, key=lambda g: g.missing, reverse=True)[: args.limit]:
            print(f"  {gap}")

    table = calendars.session_table(frame, zone)
    print(f"\n{len(table)} local trading days; median {int(table['bars'].median()):,} bars/day.")
    thin = table[table["bars"] < table["bars"].median() * 0.5]
    if len(thin):
        print(
            f"{len(thin)} day(s) with under half the median bar count, e.g. "
            f"{', '.join(str(d) for d in thin['date'].head(5))}"
        )
    return 0


# ----------------------------------------------------------------------- backtest


async def _cmd_backtest(settings: Settings, args) -> int:
    _require_backtest_extra()
    from pydantic import ValidationError  # noqa: PLC0415

    from trader.service.backtest import BacktestSpec, run_backtest  # noqa: PLC0415
    from trader.service.errors import ServiceError  # noqa: PLC0415

    try:
        spec = BacktestSpec(
            symbols=args.symbol,
            uics=args.uic,
            asset_type=args.asset_type,
            strategy=args.strategy,
            params=_parse_params(args.param),
            horizon=args.horizon,
            since=args.since,
            allocator=args.allocator,
            fee_bps=args.fee_bps,
            spread_bps=args.spread_bps,
            slippage_bps=args.slippage_bps,
            starting_cash=args.starting_cash,
            leverage=args.leverage,
        )
    except ValidationError as exc:
        raise UsageError(_first_error(exc)) from exc

    try:
        result = await run_backtest(settings, spec)
    except ServiceError as exc:
        raise UsageError(str(exc)) from exc

    print()
    print(result)

    if not result.trades.empty:
        worst = result.trades.nsmallest(min(3, len(result.trades)), "pnl")
        print("\n  Worst trades:")
        for _, trade in worst.iterrows():
            print(
                f"    {trade['label']:<12} "
                f"{trade['entry_time']:%Y-%m-%d %H:%M} -> {trade['exit_time']:%Y-%m-%d %H:%M}  "
                f"{trade['pnl']:>+12,.0f}  [{trade['exit_reason']}]"
            )

    if args.out:
        import json  # noqa: PLC0415

        args.out.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\nReport written to {args.out}")
    return 0


# ----------------------------------------------------------------------- features


def _cmd_features_list(settings: Settings) -> int:
    _require_data_extra()
    from trader.service.catalog import list_feature_sets  # noqa: PLC0415

    for info in list_feature_sets():
        tag = "  (multi-timeframe)" if info.context_capable else ""
        print(f"{info.name}{tag}")
        if info.summary:
            print(f"  {info.summary}")
        print(f"  {len(info.columns)} columns: {', '.join(info.columns)}\n")
    return 0


async def _cmd_features_build(settings: Settings, args) -> int:
    _require_data_extra()
    from pydantic import ValidationError  # noqa: PLC0415

    from trader.service.errors import ServiceError  # noqa: PLC0415
    from trader.service.features import FeatureBuildSpec, run_feature_build  # noqa: PLC0415

    context = args.context.split(",") if args.context else []
    try:
        spec = FeatureBuildSpec(
            symbols=[args.symbol],
            uics=[args.uic] if args.uic is not None else [],
            asset_type=args.asset_type,
            exchange=args.exchange,
            horizon=args.horizon,
            context_horizons=context,
            since=args.since,
            feature_set=args.feature_set,
            persist=args.persist,
        )
    except ValidationError as exc:
        raise UsageError(_first_error(exc)) from exc

    try:
        summaries = await run_feature_build(settings, spec)
    except ServiceError as exc:
        raise UsageError(str(exc)) from exc

    for summary in summaries:
        print(f"\n{summary['key']}  ({summary['label']})  {summary['feature_set']}")
        print(f"  rows        {summary['rows']:,}")
        print(f"  columns     {len(summary['columns'])}")
        print(f"  warmup      {summary['warmup']} bars")
        print(f"  nan rows    {summary['nan_rows']:,}")
        print(f"  span        {summary['first']} -> {summary['last']}")
        print(f"  digest      {summary['digest'][:12]}")
        if spec.persist:
            print(f"  written     {summary['persisted_files']} file(s)")
    return 0


# ------------------------------------------------------------------------- labels


async def _cmd_labels(settings: Settings, args) -> int:
    _require_data_extra()
    from pydantic import ValidationError  # noqa: PLC0415

    from trader.service.errors import ServiceError  # noqa: PLC0415
    from trader.service.labels import LabelSpec, run_labelling  # noqa: PLC0415

    try:
        spec = LabelSpec(
            symbols=[args.symbol],
            uics=[args.uic] if args.uic is not None else [],
            asset_type=args.asset_type,
            exchange=args.exchange,
            horizon=args.horizon,
            since=args.since,
            stop=args.stop,
            take=args.take,
            max_bars=args.max_bars,
            min_return=args.min_return,
        )
    except ValidationError as exc:
        raise UsageError(_first_error(exc)) from exc

    try:
        summary = await run_labelling(settings, spec)
    except ServiceError as exc:
        raise UsageError(str(exc)) from exc

    barriers = summary["barriers"]
    print(
        f"\nbarriers: stop={barriers['stop']} take={barriers['take']} "
        f"max_bars={barriers['max_bars']}  min_return={args.min_return}"
    )
    print(f"events:   {summary['n_events']:,}")
    counts, pct = summary["class_counts"], summary["class_pct"]
    for cls, name in ((-1, "down (-1)"), (0, "flat ( 0)"), (1, "up   (+1)")):
        print(f"  {name}   {counts[cls]:>8,}   {pct[cls]:>7.1%}")
    breakdown = summary["barrier_breakdown"]
    print(
        f"barrier:  stop {breakdown['stop']:,}  take {breakdown['take']:,}  "
        f"time {breakdown['time']:,}"
    )
    print(f"mean |ret|:       {summary['mean_abs_ret']}")
    print(f"median bars held: {summary['median_bars_held']}")

    if len(summary["per_symbol"]) > 1:
        print("\nper symbol:")
        for label, item in summary["per_symbol"].items():
            print(f"  {label:<12} {item['n_events']:,} events  {item['class_pct']}")
    return 0


# -------------------------------------------------------------------------- train


async def _cmd_train(settings: Settings, args) -> int:
    _require_model_extra(args.model_type)
    from pydantic import ValidationError  # noqa: PLC0415

    from trader.service.errors import ServiceError  # noqa: PLC0415
    from trader.service.training import TrainingSpec, run_training  # noqa: PLC0415

    context = args.context.split(",") if args.context else []
    try:
        spec = TrainingSpec(
            symbols=args.symbol,
            uics=args.uic,
            asset_type=args.asset_type,
            exchange=args.exchange,
            name=args.name,
            model_type=args.model_type,
            horizon=args.horizon,
            context_horizons=context,
            feature_set=args.feature_set,
            stop=args.stop,
            take=args.take,
            max_bars=args.max_bars,
            min_return=args.min_return,
            window=args.window,
            train_end=args.train_end,
            val_end=args.val_end,
            embargo_bars=args.embargo_bars,
            since=args.since,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_leaves=args.num_leaves,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            lr=args.lr,
            use_sample_weights=args.sample_weights,
            seed=args.seed,
        )
    except ValidationError as exc:
        raise UsageError(_first_error(exc)) from exc

    last = [0.0]

    def _progress(value: float) -> None:
        if value - last[0] >= 0.05 or value >= 1.0:
            last[0] = value
            print(f"  ... {value:5.0%}", end="\r", flush=True)

    try:
        result = await run_training(settings, spec, progress=_progress)
    except ServiceError as exc:
        raise UsageError(str(exc)) from exc

    print(" " * 20, end="\r")
    metrics = result["metrics"]
    print(f"\nmodel_id: {result['model_id']}")
    print("metrics:")
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key in metrics:
            print(f"  {key:<16} {metrics[key]}")
    report = result["report"]
    confusions = (("val", report.get("val_confusion")), ("test", report.get("test_confusion")))
    for name, matrix in confusions:
        if matrix:
            print(f"{name} confusion (rows = true down/flat/up):")
            for row in matrix:
                print("  " + "  ".join(f"{v:>6}" for v in row))
    return 0


# --------------------------------------------------------------------------- tune


def _parse_grid(items: list[str]) -> dict[str, list[object]]:
    """Turn ``["window=16,32", "lr=1e-3,3e-4"]`` into ``{"window": [16, 32], ...}``."""
    from trader.service.inputs import coerce_scalar  # noqa: PLC0415

    grid: dict[str, list[object]] = {}
    for item in items:
        if "=" not in item:
            raise UsageError(f"--grid must be KEY=v1,v2 (got {item!r})")
        key, _, raw = item.partition("=")
        values = [coerce_scalar(v.strip()) for v in raw.split(",") if v.strip()]
        if not values:
            raise UsageError(f"--grid {key}: no values")
        grid[key.strip()] = values
    return grid


async def _cmd_tune(settings: Settings, args) -> int:
    is_model = args.strategy in ("lstm", "gbm")
    if is_model:
        _require_model_extra(args.model_type)
    else:
        _require_backtest_extra()
    from pathlib import Path  # noqa: PLC0415

    from pydantic import ValidationError  # noqa: PLC0415

    from trader.service.errors import ServiceError  # noqa: PLC0415
    from trader.service.tuning import TuningSpec, render, run_tuning, save_report  # noqa: PLC0415

    context = args.context.split(",") if args.context else []
    try:
        spec = TuningSpec(
            strategy=args.strategy,
            params=_parse_params(args.param),
            base=dict(
                symbols=args.symbol,
                uics=args.uic,
                asset_type=args.asset_type,
                exchange=args.exchange,
                name=args.name,
                model_type=args.model_type,
                horizon=args.horizon,
                context_horizons=context,
                feature_set=args.feature_set,
                stop=args.stop,
                take=args.take,
                max_bars=args.max_bars,
                min_return=args.min_return,
                window=args.window,
                embargo_bars=args.embargo_bars,
                since=args.since,
                hidden=args.hidden,
                layers=args.layers,
                dropout=args.dropout,
                bidirectional=args.bidirectional,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_leaves=args.num_leaves,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_child_samples=args.min_child_samples,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                lr=args.lr,
                use_sample_weights=args.sample_weights,
                seed=args.seed,
            ),
            cv=dict(
                folds=args.folds,
                mode=args.cv_mode,
                train_days=args.train_days,
                val_days=args.val_days,
                test_days=args.test_days,
                step_days=args.step_days,
                fee_bps=args.fee_bps,
                spread_bps=args.spread_bps,
                slippage_bps=args.slippage_bps,
                allocator=args.allocator,
                leverage=args.leverage,
                threshold=args.threshold,
                on_no_signal=args.on_no_signal,
            ),
            grid=_parse_grid(args.grid),
            top_k=args.top,
            max_workers=args.workers,
        )
    except ValidationError as exc:
        raise UsageError(_first_error(exc)) from exc

    last = [0.0]

    def _progress(value: float) -> None:
        if value - last[0] >= 0.02 or value >= 1.0:
            last[0] = value
            print(f"  ... {value:5.0%}", end="\r", flush=True)

    try:
        report = await run_tuning(settings, spec, progress=_progress)
    except ServiceError as exc:
        raise UsageError(str(exc)) from exc

    print(" " * 20, end="\r")
    print(render(report))
    if args.out:
        path = save_report(report, Path(args.out))
        print(f"\nreport: {path}")
    return 0


# ------------------------------------------------------------------------- models


def _cmd_models_list(settings: Settings) -> int:
    _require_data_extra()
    from trader.service.catalog import list_models  # noqa: PLC0415

    rows = list_models(settings)
    if not rows:
        print(f"No models under {settings.models_dir}. Train one: trader train ...")
        return 1
    for model in rows:
        ctx = f"+{','.join(map(str, model.context_horizons))}" if model.context_horizons else ""
        f1 = model.metrics.get("test_macro_f1", model.metrics.get("val_macro_f1", "?"))
        print(
            f"{model.id:<40} {model.feature_set:<9} {model.horizon}m{ctx:<8} "
            f"w{model.window:<4} f1={f1}"
        )
    print(f"\n{len(rows)} model(s).")
    return 0


def _cmd_models_show(settings: Settings, args) -> int:
    _require_data_extra()
    import json  # noqa: PLC0415

    from trader.models.registry import ModelNotFound, ModelRegistry  # noqa: PLC0415

    try:
        manifest = ModelRegistry(settings.models_dir).manifest(args.model_id)
    except ModelNotFound as exc:
        raise UsageError(str(exc)) from exc
    print(json.dumps(manifest, indent=2))
    return 0


# -------------------------------------------------------------------------- serve


def _cmd_serve(settings: Settings, args) -> int:
    try:
        from trader.api.serve import run  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise UsageError(
            f"the serve command needs the API dependencies ({exc}). "
            'Install them with:  pip install -e ".[data,api]"'
        ) from exc

    print(
        f"trader API on http://{args.host}:{args.port}  (environment: "
        f"{settings.saxo.environment.value})"
    )
    run(host=args.host, port=args.port, reload=args.reload)
    return 0


# ---------------------------------------------------------------------- helpers


async def _resolve(client: SaxoClient, symbol: str, asset_type: str | None) -> Instrument:
    """Resolve a symbol, turning ambiguity into an actionable message."""
    try:
        return await resolve(client, symbol, asset_type=asset_type)
    except AmbiguousInstrument as exc:
        lines = "\n".join(
            f"  --asset-type {c.asset_type:<14} uic {c.uic:<9} {c.symbol}  {c.description}"
            for c in exc.candidates
        )
        raise UsageError(
            f"{symbol!r} matches several instruments. Re-run with one of:\n{lines}"
        ) from exc


def _parse_since(value: str | None) -> datetime | None:
    """Parse ``--since`` as a date, a timestamp, or a lookback like ``90d``."""
    from trader.service.inputs import parse_since  # noqa: PLC0415

    try:
        return parse_since(value)
    except ValueError as exc:
        raise UsageError(f"--since: {exc}") from exc


def _parse_params(items: list[str]) -> dict[str, object]:
    """Turn ``["fast=10", "long_only=true"]`` into a coerced kwargs dict."""
    from trader.service.inputs import parse_params  # noqa: PLC0415

    try:
        return parse_params(items)
    except ValueError as exc:
        raise UsageError(f"--param must be KEY=VALUE ({exc})") from exc


def _first_error(exc: Exception) -> str:
    """The first line of a pydantic ``ValidationError``, for a one-line usage message."""
    try:
        first = exc.errors()[0]  # type: ignore[attr-defined]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "input"
        return f"{loc}: {first.get('msg', exc)}"
    except (AttributeError, IndexError, KeyError):
        return str(exc)


def _add_gbm_args(parser: argparse.ArgumentParser) -> None:
    """LightGBM hyperparameters shared by ``train`` and ``tune`` (used when --model-type gbm)."""
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=-1, help="-1 = no limit")
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    auth = sub.add_parser("auth", help="manage Saxo authentication")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("login", help="browser login and store tokens")
    auth_sub.add_parser("status", help="show stored token state")
    auth_sub.add_parser("logout", help="discard stored tokens")

    sub.add_parser("account", help="show user, accounts, and balances")

    instruments = sub.add_parser("instruments", help="look up instruments and their Uics")
    instruments_sub = instruments.add_subparsers(dest="instruments_command", required=True)
    search_p = instruments_sub.add_parser("search", help="search by ticker, ISIN, or name")
    search_p.add_argument("keywords")
    search_p.add_argument("--asset-type", help="restrict to one asset type, e.g. Stock")
    search_p.add_argument("--limit", type=int, default=20)
    show_p = instruments_sub.add_parser("show", help="full detail for one symbol")
    show_p.add_argument("symbol")
    show_p.add_argument("--asset-type")

    exchange_p = sub.add_parser("exchange", help="exchanges and their trading sessions")
    exchange_p.add_argument("exchange_id", nargs="?", help="omit to list every exchange")

    data = sub.add_parser("data", help="market data lake")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    depth_p = data_sub.add_parser("depth", help="measure how much history Saxo serves")
    depth_p.add_argument("symbol")
    depth_p.add_argument("--asset-type")
    depth_p.add_argument("--horizons", help="comma-separated, e.g. 1m,5m,1h,1d")
    depth_p.add_argument(
        "--max-pages", type=int, default=40, help="probe's own cap per horizon (default 40)"
    )
    depth_p.add_argument("--out", type=_path, help="where to write the JSON report")

    backfill_p = data_sub.add_parser("backfill", help="fetch bars into the Parquet lake")
    backfill_p.add_argument("symbol")
    backfill_p.add_argument("--asset-type")
    backfill_p.add_argument(
        "--horizon", default="1m", help="comma-separated, e.g. 1m,5m (default 1m)"
    )
    backfill_p.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")
    backfill_p.add_argument("--max-bars", type=int, help="stop after roughly this many bars")
    backfill_p.add_argument("--max-pages", type=int, default=1000)
    backfill_p.add_argument(
        "--no-resume", action="store_true", help="refetch instead of extending stored coverage"
    )

    data_sub.add_parser("coverage", help="what the lake currently holds")

    symbols_p = data_sub.add_parser(
        "symbols", help="fetch Saxo symbols for every uic in the lake"
    )
    symbols_p.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch every uic, not only those with no recorded symbol",
    )

    sessions_p = data_sub.add_parser("sessions", help="inferred trading hours and real gaps")
    sessions_p.add_argument("symbol")
    sessions_p.add_argument("--asset-type")
    sessions_p.add_argument("--horizon", default="1m")
    sessions_p.add_argument("--limit", type=int, default=10, help="gaps to list")

    backtest_p = sub.add_parser("backtest", help="run a strategy over stored bars")
    backtest_p.add_argument(
        "--symbol",
        action="append",
        required=True,
        metavar="SYMBOL",
        help="instrument to trade; repeat for a basket",
    )
    backtest_p.add_argument(
        "--uic",
        action="append",
        type=int,
        default=[],
        help="skip symbol resolution; pairs positionally with --symbol",
    )
    backtest_p.add_argument("--asset-type", help="asset type for every --symbol (default Stock)")
    backtest_p.add_argument("--strategy", required=True, help="registered strategy name")
    backtest_p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="strategy parameter, e.g. --param fast=10; repeatable",
    )
    backtest_p.add_argument("--horizon", default="5m", help="bar size (default 5m)")
    backtest_p.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")
    backtest_p.add_argument(
        "--allocator",
        default="equal-weight",
        help="equal-weight | passthrough | fixed-fraction | vol-target",
    )
    backtest_p.add_argument("--starting-cash", type=float, default=100_000.0)
    backtest_p.add_argument(
        "--fee-bps", type=float, default=0.0, help="commission, basis points of notional"
    )
    backtest_p.add_argument(
        "--spread-bps", type=float, default=0.0, help="half-spread when no ask is stored"
    )
    backtest_p.add_argument("--slippage-bps", type=float, default=0.0)
    backtest_p.add_argument("--leverage", type=float, default=1.0, help="gross exposure cap")
    backtest_p.add_argument("--out", type=_path, help="write a JSON report here")

    features_p = sub.add_parser("features", help="build causal model features from the lake")
    features_sub = features_p.add_subparsers(dest="features_command", required=True)
    features_sub.add_parser("list", help="registered feature sets and their columns")
    fb = features_sub.add_parser("build", help="compute features for a symbol and summarise")
    fb.add_argument("--symbol", required=True)
    fb.add_argument("--uic", type=int, help="skip symbol resolution")
    fb.add_argument("--asset-type", help="asset type (default Stock)")
    fb.add_argument("--exchange", help="preferred exchange suffix for a bare ticker, e.g. xnas")
    fb.add_argument("--horizon", default="5m", help="base bar size (default 5m)")
    fb.add_argument("--context", help="comma-separated context horizons, e.g. 15m,1h")
    fb.add_argument("--feature-set", default="price_v1", help="registered feature set")
    fb.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")
    fb.add_argument("--persist", action="store_true", help="also write to the feature cache")

    labels_p = sub.add_parser("labels", help="triple-barrier label balance for a symbol")
    labels_p.add_argument("--symbol", required=True)
    labels_p.add_argument("--uic", type=int, help="skip symbol resolution")
    labels_p.add_argument("--asset-type", help="asset type (default Stock)")
    labels_p.add_argument("--exchange", help="preferred exchange suffix for a bare ticker")
    labels_p.add_argument("--horizon", default="5m", help="bar size (default 5m)")
    labels_p.add_argument("--stop", type=float, default=0.005, help="stop fraction (default 0.005)")
    labels_p.add_argument("--take", type=float, default=0.01, help="take fraction (default 0.01)")
    labels_p.add_argument("--max-bars", type=int, default=24, help="vertical barrier (default 24)")
    labels_p.add_argument("--min-return", type=float, default=0.0, help="timeout deadband")
    labels_p.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")

    train_p = sub.add_parser("train", help="train a model and register it (needs [model])")
    train_p.add_argument("--symbol", action="append", required=True, metavar="SYMBOL")
    train_p.add_argument("--uic", action="append", type=int, default=[])
    train_p.add_argument("--asset-type")
    train_p.add_argument("--exchange")
    train_p.add_argument("--name", default="lstm", help="registry name prefix")
    train_p.add_argument(
        "--model-type", choices=("lstm", "gbm"), default="lstm", help="model family (default lstm)"
    )
    train_p.add_argument("--horizon", default="5m")
    train_p.add_argument("--context", help="comma-separated context horizons, e.g. 15m,1h")
    train_p.add_argument("--feature-set", default="price_v1")
    train_p.add_argument("--stop", type=float, default=0.005)
    train_p.add_argument("--take", type=float, default=0.01)
    train_p.add_argument("--max-bars", type=int, default=24)
    train_p.add_argument("--min-return", type=float, default=0.0)
    train_p.add_argument("--window", type=int, default=32, help="sequence length in bars")
    train_p.add_argument("--train-end", required=True, help="train/val boundary date")
    train_p.add_argument("--val-end", required=True, help="val/test boundary date")
    train_p.add_argument("--embargo-bars", type=int, help="default window + max_bars")
    train_p.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")
    train_p.add_argument("--hidden", type=int, default=64)
    train_p.add_argument("--layers", type=int, default=2)
    train_p.add_argument("--dropout", type=float, default=0.2)
    train_p.add_argument("--bidirectional", action="store_true")
    train_p.add_argument("--epochs", type=int, default=40)
    train_p.add_argument("--batch-size", type=int, default=128)
    _add_gbm_args(train_p)
    train_p.add_argument("--lr", type=float, default=1e-3, help="LSTM Adam lr / GBM learning rate")
    train_p.add_argument(
        "--sample-weights", action="store_true", help="weight by return / uniqueness"
    )
    train_p.add_argument("--seed", type=int, default=0)

    tune_p = sub.add_parser(
        "tune", help="walk-forward grid sweep over a strategy's knobs"
    )
    tune_p.add_argument("--symbol", action="append", required=True, metavar="SYMBOL")
    tune_p.add_argument("--uic", action="append", type=int, default=[])
    tune_p.add_argument("--asset-type")
    tune_p.add_argument("--exchange")
    tune_p.add_argument("--name", default="lstm", help="registry name prefix")
    tune_p.add_argument(
        "--strategy",
        default="lstm",
        help="registered strategy to sweep: lstm, gbm (model), or ma_cross / orb (classical)",
    )
    tune_p.add_argument(
        "--model-type",
        choices=("lstm", "gbm"),
        default="lstm",
        help="model family for a model sweep",
    )
    tune_p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="K=V",
        help="fixed strategy param for a classical sweep, e.g. --param flat_eod=true; repeatable",
    )
    tune_p.add_argument("--horizon", default="15m")
    tune_p.add_argument("--context", help="comma-separated context horizons, e.g. 1h,4h")
    tune_p.add_argument("--feature-set", default="price_v1")
    tune_p.add_argument("--stop", type=float, default=0.005)
    tune_p.add_argument("--take", type=float, default=0.01)
    tune_p.add_argument("--max-bars", type=int, default=24)
    tune_p.add_argument("--min-return", type=float, default=0.0)
    tune_p.add_argument("--window", type=int, default=32)
    tune_p.add_argument("--embargo-bars", type=int, help="default window + max_bars")
    tune_p.add_argument("--since", help="YYYY-MM-DD, an ISO timestamp, 90d, 2y, or 'all'")
    tune_p.add_argument("--hidden", type=int, default=64)
    tune_p.add_argument("--layers", type=int, default=2)
    tune_p.add_argument("--dropout", type=float, default=0.2)
    tune_p.add_argument("--bidirectional", action="store_true")
    tune_p.add_argument("--epochs", type=int, default=40)
    tune_p.add_argument("--batch-size", type=int, default=128)
    _add_gbm_args(tune_p)
    tune_p.add_argument("--lr", type=float, default=1e-3, help="LSTM Adam lr / GBM learning rate")
    tune_p.add_argument("--sample-weights", action="store_true")
    tune_p.add_argument("--seed", type=int, default=0)
    tune_p.add_argument("--folds", type=int, default=5)
    tune_p.add_argument("--cv-mode", choices=("rolling", "anchored"), default="rolling")
    tune_p.add_argument("--train-days", type=float, default=365.0)
    tune_p.add_argument("--val-days", type=float, default=45.0)
    tune_p.add_argument("--test-days", type=float, default=45.0)
    tune_p.add_argument("--step-days", type=float, help="origin step (default: test-days)")
    tune_p.add_argument("--fee-bps", type=float, default=0.0)
    tune_p.add_argument("--spread-bps", type=float, default=0.0)
    tune_p.add_argument("--slippage-bps", type=float, default=0.0)
    tune_p.add_argument("--allocator", default="equal-weight")
    tune_p.add_argument("--leverage", type=float, default=1.0)
    tune_p.add_argument("--threshold", type=float, default=0.15)
    tune_p.add_argument("--on-no-signal", choices=("hold", "flat"), default="hold")
    tune_p.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="KEY=v1,v2",
        help="field to sweep, e.g. --grid window=16,32 --grid stop=0.004,0.008",
    )
    tune_p.add_argument("--top", type=int, default=5, help="configs to keep in report['top']")
    tune_p.add_argument(
        "--workers", type=int, default=0, help="parallel configs (0 = auto, 1 = in-process)"
    )
    tune_p.add_argument("--out", help="also write the full JSON report here")

    models_p = sub.add_parser("models", help="browse the trained-model registry")
    models_sub = models_p.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="every registered model")
    models_show = models_sub.add_parser("show", help="the full manifest of one model")
    models_show.add_argument("model_id")

    serve_p = sub.add_parser("serve", help="run the API and, if built, the UI")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    return parser


def _path(value: str):
    from pathlib import Path  # noqa: PLC0415

    return Path(value)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    settings = get_settings()

    try:
        if args.command == "auth":
            if args.auth_command == "login":
                return asyncio.run(_cmd_auth_login(settings))
            if args.auth_command == "status":
                return _cmd_auth_status(settings)
            if args.auth_command == "logout":
                return _cmd_auth_logout(settings)
        elif args.command == "account":
            return asyncio.run(_cmd_account(settings))
        elif args.command == "instruments":
            if args.instruments_command == "search":
                return asyncio.run(_cmd_instruments_search(settings, args))
            if args.instruments_command == "show":
                return asyncio.run(_cmd_instruments_show(settings, args))
        elif args.command == "exchange":
            return asyncio.run(_cmd_exchange(settings, args))
        elif args.command == "data":
            if args.data_command == "depth":
                return asyncio.run(_cmd_data_depth(settings, args))
            if args.data_command == "backfill":
                return asyncio.run(_cmd_data_backfill(settings, args))
            if args.data_command == "coverage":
                return _cmd_data_coverage(settings, args)
            if args.data_command == "symbols":
                return asyncio.run(_cmd_data_symbols(settings, args))
            if args.data_command == "sessions":
                return asyncio.run(_cmd_data_sessions(settings, args))
        elif args.command == "backtest":
            return asyncio.run(_cmd_backtest(settings, args))
        elif args.command == "features":
            if args.features_command == "list":
                return _cmd_features_list(settings)
            if args.features_command == "build":
                return asyncio.run(_cmd_features_build(settings, args))
        elif args.command == "labels":
            return asyncio.run(_cmd_labels(settings, args))
        elif args.command == "train":
            return asyncio.run(_cmd_train(settings, args))
        elif args.command == "tune":
            return asyncio.run(_cmd_tune(settings, args))
        elif args.command == "models":
            if args.models_command == "list":
                return _cmd_models_list(settings)
            if args.models_command == "show":
                return _cmd_models_show(settings, args)
        elif args.command == "serve":
            return _cmd_serve(settings, args)
    except ReauthRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (UsageError, InstrumentNotFound, ChartError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AuthError, SaxoAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # A backfill commits each page as it lands, so an interrupt here loses
        # at most the page in flight; re-running resumes.
        print("\ninterrupted", file=sys.stderr)
        return 130

    parser.error(f"unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
