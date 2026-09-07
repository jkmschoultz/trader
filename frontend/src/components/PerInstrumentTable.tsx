import type { InstrumentStats } from "../api/types";

const money = (v: number) =>
  v.toLocaleString(undefined, { maximumFractionDigits: 0, signDisplay: "always" });

export function PerInstrumentTable({ byInstrument }: { byInstrument: Record<string, InstrumentStats> }) {
  const rows = Object.entries(byInstrument);
  if (rows.length <= 1) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-300 text-left text-slate-500 dark:border-slate-700">
            <th className="py-1 pr-4">Instrument</th>
            <th className="py-1 pr-4 text-right">Trades</th>
            <th className="py-1 pr-4 text-right">PnL</th>
            <th className="py-1 pr-4 text-right">Return</th>
            <th className="py-1 pr-4 text-right">Hit rate</th>
            <th className="py-1 text-right">Profit factor</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map(([label, s]) => (
            <tr key={label} className="border-b border-slate-100 dark:border-slate-900">
              <td className="py-1 pr-4">{label}</td>
              <td className="py-1 pr-4 text-right">{s.n_trades}</td>
              <td className={`py-1 pr-4 text-right ${s.pnl >= 0 ? "text-green-600" : "text-red-600"}`}>
                {money(s.pnl)}
              </td>
              <td className="py-1 pr-4 text-right">{(s.return_on_start * 100).toFixed(2)}%</td>
              <td className="py-1 pr-4 text-right">{(s.hit_rate * 100).toFixed(0)}%</td>
              <td className="py-1 text-right">
                {Number.isFinite(s.profit_factor) ? s.profit_factor.toFixed(2) : "∞"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
