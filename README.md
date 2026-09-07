# trader

An intraday trading system on the Saxo Bank OpenAPI: research and backtesting, paper trading,
live trading, and a UI to drive all of it. The modelling centrepiece is a multi-timeframe LSTM
classifier trained on triple-barrier labels.

**Status: Phase 0 (authentication) complete.** See [the plan](#roadmap) for what comes next.

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
```

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
| `src/trader/features/` | Causal feature pipeline |
| `src/trader/labels/` | Triple-barrier labelling |
| `src/trader/models/` | Dataset windowing, LSTM, training, model registry |
| `src/trader/strategies/` | `Strategy` ABC, registry, and implementations |
| `src/trader/backtest/` | Event-driven engine, cost model, metrics |
| `src/trader/execution/` | Broker abstraction: paper and live |
| `src/trader/api/` | FastAPI service |
| `frontend/` | React/Vite UI |

Only `saxo/` and the config layer exist so far.

## Design notes

**Sim is the default everywhere.** No code path selects the live environment implicitly, and
placing a real order requires both `saxo.environment = "live"` and `allow_live_trading = true`.
See [docs/live-trading.md](docs/live-trading.md).

**Reads retry; writes do not.** `SaxoClient` retries GETs on timeouts and 5xx, but never retries
a POST — an order that times out in flight may already have reached the exchange, so replaying
it could double the position. Writes reconcile instead.

**Rate limits are enforced client-side.** Saxo allows 120 requests/minute per service group and
1 order/second. Staying just under those caps is faster in practice than being throttled,
particularly during history backfill.

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffold, OAuth2, rate-limited client | done, verified against sim |
| 1 | Data layer, history-depth spike, session calendars | next |
| 2 | Strategy interface + backtest engine | |
| 3 | Features, triple-barrier labels, LSTM | |
| 4 | FastAPI + React UI | |
| 5 | Paper trading | |
| 6 | Live trading (gated) | |

Classical algorithms (ORB, MA cross, RSI mean reversion, VWAP, Donchian) plug into the same
`Strategy` interface and can be added any time after Phase 2.
