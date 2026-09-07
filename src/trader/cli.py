"""Command-line entry point.

    trader auth login|status|logout   # Saxo authentication
    trader account                    # user, accounts, and balances

    trader instruments search KEYWORDS      # find a Uic
    trader instruments show SYMBOL          # full detail for one instrument
    trader exchange [EXCHANGE_ID]           # exchanges and their sessions

    trader data depth SYMBOL          # how much history Saxo serves
    trader data backfill SYMBOL       # fetch bars into the Parquet lake
    trader data coverage              # what the lake holds
    trader data sessions SYMBOL       # inferred trading hours and real gaps

The ``data`` subcommands need the optional data dependencies::

    pip install -e ".[data]"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta

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

    horizons = [parse_horizon(h) for h in args.horizon.split(",")]
    since = _parse_since(args.since)

    store = lake_mod.BarLake(settings.data_dir)
    async with SaxoClient(settings) as client:
        instrument = await _resolve(client, args.symbol, args.asset_type)
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
            print(coverage)
    return 0


def _cmd_data_coverage(settings: Settings, args) -> int:
    _, _, _, _, lake_mod = _require_data_extra()

    store = lake_mod.BarLake(settings.data_dir)
    keys = store.series()
    if not keys:
        print(f"No series stored under {store.root}. Run: trader data backfill SYMBOL")
        return 1

    print(f"Lake: {store.root}\n")
    for key in keys:
        coverage = store.coverage(key)
        if coverage:
            print(coverage)
    print(f"\n{len(keys)} series.")
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
    if not value:
        return None
    text = value.strip()
    if text.casefold() in {"all", "max"}:
        return None
    if text[-1:].casefold() in {"d", "y"} and text[:-1].replace(".", "", 1).isdigit():
        days = float(text[:-1]) * (365.25 if text[-1:].casefold() == "y" else 1)
        return datetime.now(UTC) - timedelta(days=days)
    try:
        from dateutil.parser import isoparse  # noqa: PLC0415

        parsed = isoparse(text)
    except (ImportError, ValueError) as exc:
        raise UsageError(
            f"cannot read --since {value!r}; use YYYY-MM-DD, an ISO timestamp, "
            "a lookback like 90d or 2y, or 'all'"
        ) from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


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

    sessions_p = data_sub.add_parser("sessions", help="inferred trading hours and real gaps")
    sessions_p.add_argument("symbol")
    sessions_p.add_argument("--asset-type")
    sessions_p.add_argument("--horizon", default="1m")
    sessions_p.add_argument("--limit", type=int, default=10, help="gaps to list")

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
            if args.data_command == "sessions":
                return asyncio.run(_cmd_data_sessions(settings, args))
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
