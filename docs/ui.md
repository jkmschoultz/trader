# The UI: architecture and the decisions in it

Phase 3 is a FastAPI service (`trader.api`) and a React/Vite/Tailwind front-end
(`frontend/`), sitting on a framework-agnostic orchestration layer
(`trader.service`) that the CLI shares. It exists so the lake, strategy signals,
equity curves, trades, and metrics can be seen before the model work in Phase 4.

## Layers

```
frontend/            React + TanStack Query + lightweight-charts
   │  /api (proxied in dev, same-origin in prod)
trader.api           FastAPI routes, background-job store, SSE
trader.service       run_backtest / run_backfill / lake reads / catalogue
trader.backtest      the engine (unchanged; gained an optional progress callback)
trader.data          the bar lake
```

`trader.service` has no HTTP and no argparse in it. `trader backtest` on the CLI
and `POST /api/backtests` build the same `BacktestSpec` and call the same
`run_backtest`, so the two front-ends cannot drift.

## Background jobs

A backtest takes seconds; a backfill can take minutes. Both are submitted as
jobs:

- `POST /api/backtests` / `POST /api/lake/backfill` → `{"job_id": ...}` (202).
- `GET /api/jobs/{id}` → status, progress, and (when done) the result.
- `GET /api/jobs/{id}/events` → Server-Sent Events: a `progress`/`status` ping
  stream, terminated by a `done` event carrying the final snapshot.

The store is in-process and in-memory (`trader.api.jobs.JobStore`): restart the
server and running jobs are gone. That is the right trade for a single-user
local tool and keeps the deployment a single process with no broker. The
front-end's `useJob` hook prefers the SSE stream and falls back to 1 s polling
if it errors.

Errors in the *shape* of a request (bad horizon, no symbols) are rejected
synchronously as 422. Errors that need the lake or the network — unknown
strategy, nothing stored, not signed in — surface as a failed job with the
message in `job.error`, because that is where the work actually happens.

A `ServiceError` may also set `.data`, a JSON-safe dict copied onto the failed
job as `error_data`. `SeriesNotStored` uses it (`{kind: "series_not_stored",
symbol, horizon, asset_type, uic, since}`) so the Backtest view can show a
one-click **Fetch … and re-run** button that fires the backfill and resubmits
the backtest when it lands.

## Symbols

The Backtest and Backfill forms take bare tickers (`AAPL, NVDA`) plus an
**asset type** (default *Stock (CFD)* — the leveraged product) and a
**preferred exchange** (default NASDAQ). Resolution goes through
`trader.service.instruments.resolve_symbol`: when a bare ticker matches the same
company on several venues it picks the first whose `:suffix` matches, trying the
request's `exchange` first, then `settings.saxo.preferred_exchanges` (default
`xnas, xnys, arcx, xlon, xetr`). Pin a listing by giving the venue in the symbol
— `AAPL:xmil` is never second-guessed. `trader instruments` / `trader data` on
the CLI keep the stricter behaviour (exact symbol or `--asset-type`).

## Serving: dev vs prod

- **Development** — two processes. `trader serve` runs uvicorn on `:8000`;
  `npm run dev` runs Vite on `:5173` and proxies `/api` to it (`vite.config.ts`).
  CORS for the Vite origin is allowed in `trader.api.app.DEV_ORIGINS`.
- **Production** — one process. `npm run build` writes `frontend/dist/`; if that
  directory exists, `create_app` mounts it at `/` with `StaticFiles(html=True)`,
  so `trader serve` alone serves both API and UI.

## Saxo auth

The service reuses whatever `TokenStore` holds and refreshes it while the process
is up. If the process is down for over an hour the refresh token expires and a
browser sign-in is required (a property of Saxo's OAuth, see the main README).

- `GET /api/auth/status` — read-only view of the stored session, for the UI's
  banner.
- `POST /api/auth/login` — starts the authorization-code flow as an `auth-login`
  job (one at a time). It binds the local callback port and opens the system
  browser, exactly like `trader auth login`; the authorize URL is also pushed to
  `job.message` so the UI can show a clickable link if the browser did not open.
  On success the job saves the tokens and the banner's `useAuthStatus` refetches.

This is fine for the local single-user tool this is. It is **not** meant for a
shared host: the OAuth loopback redirect only reaches the machine the browser
runs on, so a remote `trader serve` cannot complete the flow this way — sign in
with the CLI on that machine instead.

Instrument search and backfills return 409 when there is no live session;
everything else works off the lake alone.

## Charting

`lightweight-charts` for both the price chart (candles + volume + fill markers)
and the equity chart (area + a dotted drawdown line on a secondary scale). Bars
are sent as `{time: <unix seconds>, open, high, low, close, volume}` — the format
the library wants — and decimated server-side (`read_bars`, stride to
`max_points`, last bar always kept) so a 1-minute series over years stays light.
Gaps are measured on the full slice before decimation.

## What Phase 4 adds here

The LSTM lands as another `Strategy`, so the Backtest and Results views cover it
with no change. Phase 4 adds routes for training runs and the model registry on
top of the same job store.
