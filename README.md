# trader

An intraday trading system on the Saxo Bank OpenAPI: research and backtesting, paper trading,
live trading, and a UI to drive all of it. The modelling centrepiece is a multi-timeframe LSTM
classifier trained on triple-barrier labels.

**Status: Phase 4 (features, triple-barrier labels, LSTM) in progress.** Phases 0-3 are
complete and verified against sim. See [the plan](#roadmap) for what comes next.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env      # then fill in your Saxo app credentials
```

Configuration resolves from three layers, later winning: `config/settings.toml` (committed,
non-secret) → `.env` (gitignored, secrets) → environment variables. Nested keys use a `__`
delimiter, so `TRADER_SAXO__APP_KEY` sets `settings.saxo.app_key`.

### Registering the redirect URL

The OAuth login runs a one-shot callback server on your machine. The redirect URL in `.env`
must **exactly** match one registered on your app at
[developer.saxo](https://www.developer.saxo/) → Apps:

```
http://localhost:8765/callback
```

If you use a different port, register that one too.

### Confidential vs public apps

Saxo registers apps in one of two shapes, and they need different token requests:

| App type | Has AppSecret | Token exchange |
|---|---|---|
| Confidential | yes | HTTP Basic `client_id:client_secret`, no `code_verifier` |
| Public | no | PKCE `code_verifier`, no Basic auth |

Setting `TRADER_SAXO__APP_SECRET` selects the confidential flow automatically. Using the wrong
one fails with a bare `400 Bad Request` from the token endpoint and nothing else — if that
happens, `TRADER_SAXO__OAUTH_FLOW=pkce|secret` forces the choice.

## Usage

```bash
.venv/bin/trader auth login     # browser login; stores refreshable tokens
.venv/bin/trader auth status    # what is stored and how long it has left
.venv/bin/trader auth logout    # discard stored tokens
.venv/bin/trader account        # user, accounts, and balances -- proves the connection works

.venv/bin/trader instruments search AAPL        # find a Uic by ticker
.venv/bin/trader exchange NASDAQ                # is it open, and its published sessions
.venv/bin/trader data depth AAPL:xnas --asset-type Stock    # how far back Saxo actually serves bars
.venv/bin/trader data backfill AAPL:xnas --asset-type Stock --horizon 1m,5m --since 90d
.venv/bin/trader data coverage                  # what the local Parquet lake holds
.venv/bin/trader data symbols                    # fetch Saxo symbols for every stored uic
.venv/bin/trader data sessions AAPL:xnas --asset-type Stock # inferred trading hours, real gaps

.venv/bin/trader backtest --symbol AAPL:xnas --symbol MSFT:xnas --asset-type Stock \
    --strategy ma_cross --horizon 5m --since 90d --param fast=10 --param slow=30 \
    --allocator equal-weight --fee-bps 0.5 --out bt.json
.venv/bin/trader backtest --symbol AAPL:xnas --asset-type Stock --strategy orb \
    --horizon 5m --since 90d --param open_minutes=15 --param stop=0.005 --param take=0.01
```

The `data` and `backtest` commands need the optional data dependencies:
`pip install -e ".[data]"`. See [docs/history-depth.md](docs/history-depth.md) for what the depth
spike found against SIM, and [docs/backtest.md](docs/backtest.md) for the engine's timing model
and cost assumptions. Pass `--uic` alongside `--symbol` to skip the network entirely and run
straight from the lake.

### The UI

```bash
pip install -e ".[ui]"        # data layer + FastAPI + uvicorn
.venv/bin/trader serve        # API on http://127.0.0.1:8000

cd frontend && npm install && npm run dev   # Vite dev server on :5173, proxies /api
```

Open the Vite URL in development. The **Data** tab browses the bar lake and charts a series
(and can trigger a backfill); the **Backtest** tab runs a strategy over stored bars and shows
the equity curve, metrics, and blotter. Long actions run as background jobs with streamed
progress.

For a single-process deployment, `npm run build` writes `frontend/dist/` and `trader serve`
then serves the UI at `/` on the API port. Instrument search and backfills need a live Saxo
session; when the token has lapsed the UI shows a **Sign in to Saxo** button that runs the
same browser flow as `trader auth login` (local machine only). Everything else works off the
lake alone. See [docs/ui.md](docs/ui.md) for the architecture.

### A note on token lifetimes

Measured against SIM: access tokens last **20 minutes** and refresh tokens **60 minutes**, and
both rotate on every refresh. The client refreshes automatically while it is running, but **if
the process is down for more than an hour the session cannot be revived without another browser
login**. That is a property of Saxo's OAuth implementation, not a limitation of this client, and
it shapes how a long-running trading session has to be operated — an unattended overnight
restart cannot re-authenticate itself.

For quick experiments you can skip OAuth entirely by pasting a 24-hour token from the developer
portal into `TRADER_SAXO__TOKEN_24H`. It works against SIM only.

## Development

```bash
.venv/bin/pytest              # tests
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
```

Tests are isolated from your real `.env` by an autouse fixture in `tests/conftest.py`. Without
it, a test constructing `Settings()` would pick up live credentials.

## Layout

| Path | Purpose |
|---|---|
| `src/trader/saxo/` | API client: auth, rate limiting, charts, streaming, trading |
| `src/trader/data/` | Parquet lake, ingestion, session calendars, bar alignment |
| `src/trader/features/` | Causal feature pipeline ([docs/features.md](docs/features.md)) |
| `src/trader/labels/` | Triple-barrier labelling ([docs/labels.md](docs/labels.md)) |
| `src/trader/models/` | Dataset windowing, LSTM, training, model registry ([docs/model.md](docs/model.md)) |
| `src/trader/strategies/` | `Strategy` ABC, registry, `ma_cross`, `orb`, `lstm` |
| `src/trader/backtest/` | Event-driven engine, allocators, cost model, metrics |
| `src/trader/service/` | Framework-agnostic orchestration shared by the CLI and the API |
| `src/trader/api/` | FastAPI service: catalogue, lake, backtests, background jobs |
| `src/trader/execution/` | Broker abstraction: paper and live |
| `frontend/` | React/Vite/Tailwind UI |

`saxo/`, `data/`, `features/`, `labels/`, `models/`, `strategies/`, `backtest/`, `service/`,
`api/`, `frontend/`, and the config layer exist so far. `execution/` arrives in Phase 5.

## Design notes

**Sim is the default everywhere.** No code path selects the live environment implicitly, and
placing a real order requires both `saxo.environment = "live"` and `allow_live_trading = true`.
See [docs/live-trading.md](docs/live-trading.md).

**Reads retry; writes do not.** `SaxoClient` retries GETs on timeouts and 5xx, but never retries
a POST — an order that times out in flight may already have reached the exchange, so replaying
it could double the position. Writes reconcile instead.

**Rate limits are enforced client-side.** Saxo allows 120 requests/minute per service group and
1 order/second. Staying just under those caps is faster in practice than being throttled,
particularly during history backfill. The limiter's memory is per process, not global -- see
[docs/history-depth.md](docs/history-depth.md) for what that means running several short scripts
back to back, and how a 429 is now recovered from correctly regardless of the cause.

**`FirstSampleTime` is not trustworthy.** Saxo's chart response claims to say how far back an
instrument's history goes; measured against SIM, it reported the same fixed value across every
intraday horizon regardless of where the data actually ran out (six and a half years off, for
1-hour bars on AAPL). `trader.data.depth` measures depth empirically instead of trusting the
field. See [docs/history-depth.md](docs/history-depth.md).

**The bar lake is idempotent and resumable by design.** `BarLake.write` merges on bar timestamp,
so replaying a backfill costs time and nothing else. `trader data backfill` commits every page as
it lands, so an interrupted run loses at most the page in flight and a re-run picks up from what
is already stored, in both directions (topping up recent bars, and extending further into the
past).

**Series are keyed by Uic, labelled by symbol.** Partition paths stay numeric because a Saxo
symbol can be reassigned while a Uic cannot. `data/instruments.json` is the side table that maps
each `(asset_type, uic)` back to its symbol; a backfill records it automatically, and
`trader data symbols` fills it in for a lake seeded before this existed. Coverage then reads
`US500.I:CfdOnIndex:4913@1m: …` rather than the bare key.

**The backtest engine cannot look ahead.** A strategy decides on the bar that just closed and
the fill lands at the next bar's open — a price knowable at that instant. Bracket exits are
checked intrabar against `high`/`low`; a bar that opens beyond a barrier (an overnight gap)
exits at that open rather than at the barrier, and when one bar spans both the stop and the
target the stop wins, which biases bracketed strategies down (the safe direction). Strategies emit a
conviction in `[-1, 1]`; an `Allocator` turns the book's convictions into equity-fraction
weights under a leverage cap. See [docs/backtest.md](docs/backtest.md).

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, OAuth2, rate-limited client | done, verified against sim |
| 1 | Data layer, history-depth spike, session calendars | done, verified against sim |
| 2 | Strategy interface + backtest engine | done, verified against sim |
| 3 | FastAPI + React UI | done, verified against sim |
| 4 | Features, triple-barrier labels, LSTM | in progress |
| 5 | Paper trading | |
| 6 | Live trading (gated) | |

The UI comes before the model on purpose: everything worth visualising early — the bar
lake, session calendars, strategy signals, equity curves, trades, and backtest metrics —
already exists after Phase 2, and having that in front of you makes the model work in
Phase 4 easier to judge. The LSTM lands as another `Strategy`, so the UI's backtest and
research views cover it for free; Phase 4 only adds API surface for training runs and the
model registry.

Classical algorithms (ORB, MA cross, RSI mean reversion, VWAP, Donchian) plug into the same
`Strategy` interface. `ma_cross` and `orb` ship now; the rest can be added any time.
