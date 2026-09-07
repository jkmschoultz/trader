# Live trading safety

Real orders are gated deliberately. This document records why the gates exist, so they are not
casually removed later.

## The gate

Two independent settings must both be set before any order can reach the live gateway:

| Setting | Default | Meaning |
|---|---|---|
| `saxo.environment` | `sim` | Which Saxo environment hosts to use |
| `allow_live_trading` | `false` | Explicit arming of real-money orders |

`Settings.live_trading_armed` is true only when both agree. Setting the environment to `live` is
**not sufficient on its own** — that alone would make a single environment variable, or a typo in
a config file, the only thing standing between a backtest and a real order.

## Why the environments are fully separated

`saxo/environments.py` defines both host sets as complete, disjoint tuples rather than deriving
live URLs from sim ones by string substitution. A test asserts they share no values. The failure
mode this prevents is a live gateway paired with a sim token, or worse, the reverse: sim
credentials silently failing while a live host is addressed.

Tokens are also stored per environment (`state/tokens-sim.json`, `state/tokens-live.json`), so a
sim session can never present its token to the live gateway.

## Why writes are never retried

`SaxoClient` retries GET requests on timeouts and 5xx responses, and retries nothing else.

An order POST that times out has an ambiguous outcome: the request may have reached Saxo and been
accepted, or it may not. Retrying resolves the ambiguity in the worst possible direction — it can
double a position. The client therefore surfaces the failure, and the live broker reconciles
against Saxo's actual position and order state rather than guessing.

Every request also carries a unique `x-request-id`, because Saxo rejects identical operations
repeated within 15 seconds with a 409 unless they are distinguishable.

## Rate limits that matter for orders

Saxo enforces **1 order per second per session**, separately from the 120 requests/minute per
service group. `ServiceGroupLimiter` holds a dedicated one-token-per-second bucket that every
order acquires in addition to its service-group token. Exceeding this does not merely slow
trading down — rejected orders during a live session leave the strategy's intended and actual
positions out of sync.

## Still to come (Phase 6)

The gate above exists now. The rest of the live-trading safety layer does not, and must be built
before `allow_live_trading` is ever set:

- position reconciliation against Saxo on startup and periodically thereafter
- daily loss limit, max open positions, max notional
- a kill switch that flattens and halts
- typed confirmation in the UI before arming

Do not enable live trading until these exist.
