import type { Metrics } from "../api/types";

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const num = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "—");

const ROWS: [keyof Metrics, string, (v: number) => string][] = [
  ["total_return", "Total return", pct],
  ["cagr", "CAGR", pct],
  ["ann_vol", "Ann. volatility", pct],
  ["sharpe", "Sharpe", num],
  ["sortino", "Sortino", num],
  ["max_drawdown", "Max drawdown", pct],
  ["calmar", "Calmar", num],
  ["hit_rate", "Hit rate", pct],
  ["profit_factor", "Profit factor", num],
  ["exposure", "Exposure", pct],
  ["turnover", "Turnover", (v) => `${num(v)}x`],
  ["n_trades", "Trades", (v) => String(v)],
];

export function MetricsGrid({ metrics }: { metrics: Metrics }) {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3 lg:grid-cols-4">
      {ROWS.map(([key, label, fmt]) => (
        <div key={key} className="flex items-baseline justify-between border-b border-slate-200 py-1 dark:border-slate-800">
          <span className="text-sm text-slate-500 dark:text-slate-400">{label}</span>
          <span className="font-mono text-sm tabular-nums">{fmt(metrics[key] as number)}</span>
        </div>
      ))}
    </div>
  );
}
