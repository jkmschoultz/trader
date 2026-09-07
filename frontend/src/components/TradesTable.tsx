import { useState } from "react";
import type { Trade } from "../api/types";

const fmtTime = (s: string) => s.replace("T", " ").slice(0, 16);
const money = (v: number) =>
  v.toLocaleString(undefined, { maximumFractionDigits: 0, signDisplay: "always" });

export function TradesTable({ trades }: { trades: Trade[] }) {
  const [all, setAll] = useState(false);
  if (trades.length === 0) return <p className="text-sm text-slate-500">No trades.</p>;

  const sorted = [...trades].sort((a, b) => a.entry_time.localeCompare(b.entry_time));
  const shown = all ? sorted : sorted.slice(0, 50);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-300 text-left text-slate-500 dark:border-slate-700">
            <th className="py-1 pr-4">Instrument</th>
            <th className="py-1 pr-4">Entry</th>
            <th className="py-1 pr-4">Exit</th>
            <th className="py-1 pr-4 text-right">PnL</th>
            <th className="py-1 pr-4 text-right">Return</th>
            <th className="py-1">Reason</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {shown.map((t, i) => (
            <tr key={i} className="border-b border-slate-100 dark:border-slate-900">
              <td className="py-1 pr-4">{t.label}</td>
              <td className="py-1 pr-4">{fmtTime(t.entry_time)}</td>
              <td className="py-1 pr-4">{fmtTime(t.exit_time)}</td>
              <td className={`py-1 pr-4 text-right ${t.pnl >= 0 ? "text-green-600" : "text-red-600"}`}>
                {money(t.pnl)}
              </td>
              <td className="py-1 pr-4 text-right">{(t.return * 100).toFixed(2)}%</td>
              <td className="py-1">{t.exit_reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {sorted.length > 50 && (
        <button
          className="mt-2 text-sm text-blue-600 hover:underline"
          onClick={() => setAll((v) => !v)}
        >
          {all ? "Show fewer" : `Show all ${sorted.length}`}
        </button>
      )}
    </div>
  );
}
