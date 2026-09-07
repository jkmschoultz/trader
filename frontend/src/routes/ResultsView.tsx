import { EquityChart } from "../components/EquityChart";
import { MetricsGrid } from "../components/MetricsGrid";
import { PerInstrumentTable } from "../components/PerInstrumentTable";
import { TradesTable } from "../components/TradesTable";
import type { BacktestResult } from "../api/types";

const money = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });

export function ResultsView({ result }: { result: BacktestResult }) {
  const worst = [...result.trades].sort((a, b) => a.pnl - b.pnl).slice(0, 3);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <h2 className="text-lg font-semibold">
          {(result.config.labels as string[] | undefined)?.join(", ") ?? "Backtest"}
        </h2>
        <span className="font-mono text-sm text-slate-500">
          {money(result.starting_equity)} → {money(result.final_equity)}
        </span>
      </div>

      <section>
        <EquityChart equity={result.equity} />
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Metrics</h3>
        <MetricsGrid metrics={result.metrics} />
      </section>

      <PerInstrumentTable byInstrument={result.by_instrument} />

      {worst.length > 0 && (
        <section>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Worst trades</h3>
          <ul className="font-mono text-sm">
            {worst.map((t, i) => (
              <li key={i} className="text-red-600">
                {t.label} {t.entry_time.slice(0, 16)} → {t.exit_time.slice(0, 16)}{" "}
                {t.pnl.toLocaleString(undefined, { maximumFractionDigits: 0, signDisplay: "always" })}{" "}
                [{t.exit_reason}]
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Trades ({result.trades.length})
        </h3>
        <TradesTable trades={result.trades} />
      </section>
    </div>
  );
}
