# History depth: how far back does Saxo actually serve bars?

This is the record of the Phase 1 history-depth spike (`trader data depth`), run against SIM
for AAPL:xnas (Stock, uic 211). It exists because the answer is undocumented, varies by
horizon, and determines what the model layer can be built on — a multi-timeframe LSTM planned
around years of 1-minute bars needs to know now, not after the feature pipeline is written.

## Headline finding: `FirstSampleTime` is not trustworthy

Saxo's chart response includes `ChartInfo.FirstSampleTime`, which reads as an authoritative
answer to "how far back does this instrument go." It is not one. Every horizon probed here —
1m, 5m, 15m, 1h — reported the *same* value, `2003-10-31T14:30:00Z`, regardless of the actual
data available at that horizon:

| Horizon | `FirstSampleTime` reported | Where the data actually ran out |
|---|---|---|
| 1h | 2003-10-31 | **2010-06-21** — 6.9 years short of the reported figure |
| 1d | 1992-01-02 | 1992-01-02 — matches, and is independently plausible (Apple listed 1980) |

The 1h probe walked backwards until Saxo returned an empty page at 2010-06-21, six and a half
years before the `FirstSampleTime` it had been reporting the entire time. Only the daily probe's
reported value turned out to be real. This is why `trader.data.depth` measures depth by walking
history to exhaustion rather than reading the field: for at least the intraday horizons, the
field appears to be a fixed floor common to many instruments rather than a per-instrument fact.

Treat `FirstSampleTime` as a hint at most. `HistoryWalk.stopped_because` distinguishes a walk
that reached it (`"first-sample"`) from one that ran dry before reaching it (`"empty-page"` or
`"no-progress"`) — only the latter two, or an unconditional exhaustion, are trustworthy.

## Measured depth, AAPL:xnas (uic 211), Stock

Run 2026-09-07 against SIM, `trader data depth AAPL:xnas --asset-type Stock`:

| Horizon | Bars | Span | Back to | Conclusive? |
|---|---|---|---|---|
| 1m | 72,000 (capped) | 269 days | 2025-12-09 | No — cut short by `--max-pages` (best of two runs; see below) |
| 5m | 30,000 (capped) | 557 days | 2025-02-24 | No — cut short by `--max-pages` |
| 15m | 30,000 (capped) | 1,683 days (4.6y) | 2022-01-25 | No — cut short by `--max-pages` |
| 1h | 28,438 | 5,919 days (16.2y) | 2010-06-21 | **Yes** — Saxo returned an empty page |
| 1d | 8,738 | 12,664 days (34.7y) | 1992-01-02 | **Yes** — reached `FirstSampleTime` |

The first three rows used the spike's default page cap (25–40 pages) and only establish a
*lower bound*: more history exists past what was fetched. `render()` flags this automatically —
"Cut short by the probe's own page cap ... re-run with a higher `--max-pages`."

A follow-up run at `--max-pages 60` for the 1-minute horizon alone narrowed that lower bound to
the 72,000-bar / 269-day figure already folded into the table above (the default probe alone
only reaches 109 days). Getting a fully conclusive 1-minute number means either a much larger
`--max-pages` (costly: Saxo serves at most 1,200
bars/request, so a full year of 1m bars is roughly 210 requests) or accepting the lower bound
above as sufficient and letting the backfill itself discover the true limit — `trader data
backfill` reports `reached_start_of_history` for exactly this reason, so the eventual full
backfill run is itself the more conclusive measurement, at no extra cost over a dedicated probe.

## A rate-limiting pitfall found while running this spike

Two real bugs surfaced while measuring the above, both fixed in this phase, worth recording
here since they would otherwise resurface on the next long backfill:

1. **A 429's `Retry-After` selection bug.** Saxo returns *one `X-RateLimit-<Bucket>-*` header
   set per quota it tracks*, not one. A request that only exhausted the per-minute `chart`
   quota still carried an unrelated `AppDay` quota alongside it — with its own, much larger,
   `-Reset` value (23 hours vs. the 50 seconds that actually applied). The client used to return
   the first `-Reset` header it happened to iterate to, which could pick the day-quota's; it now
   only considers buckets whose own `-Remaining` is `0`, and picks the soonest reset among those.
2. **The client's rate limiter has no memory across processes.** It correctly keeps one
   long-running process under its configured rate, but several short-lived Python invocations
   run back to back against the same Saxo session can still collectively exceed the real
   server-side limit — each process starts with a full local quota, unaware of what a sibling
   process spent moments earlier. This does not affect the normal Phase 1 usage pattern (one
   backfill run at a time), but is worth knowing before scripting many short probes in sequence.

## What this means for the model design

- **Daily and hourly bars go back 15+ years** for a liquid large-cap name — ample for a
  triple-barrier-labelled classifier at those horizons.
- **1-minute depth is still being pinned down** — at least 269 days confirmed, likely
  substantially more — but that is already enough to start feature-pipeline and labelling work;
  a longer backfill can run concurrently once the true limit is known. Budget backfill time
  accordingly: at ~1,200
  bars/page and Saxo's ~100 req/min service-group limit, walking back N years of 1-minute bars
  costs roughly `N * 252 * 390 / 1200` requests — about 8 requests per trading-year of 1m data.
- **Depth is not uniform across instruments.** This spike measured one liquid US equity; a
  thinly-traded name, a newer listing, or a different asset class (FX, CFD) should be probed
  separately with `trader data depth SYMBOL` before assuming the same horizons apply.

## Reproducing this

```bash
trader data depth AAPL:xnas --asset-type Stock --horizons 1m,5m,15m,1h,1d --max-pages 40
```

Raise `--max-pages` for any horizon `render()` reports as cut short. Each run writes a JSON
report to `state/history-depth-<uic>.json` (or `--out PATH`) for later comparison — Saxo's
retention has changed before and this is the way to notice if it changes again.
