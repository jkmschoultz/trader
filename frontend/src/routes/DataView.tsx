import { useEffect, useState } from "react";

import { api, ApiError } from "../api/client";
import { ASSET_TYPES, DEFAULT_ASSET_TYPE } from "../assetTypes";
import { DEFAULT_EXCHANGE, EXCHANGES } from "../exchanges";
import { CheckboxDropdown } from "../components/CheckboxDropdown";
import { PriceChart } from "../components/PriceChart";
import { JobProgress } from "../components/JobProgress";
import { useBars, useJob, useSeries } from "../hooks";
import type { SeriesInfo } from "../api/types";

const fmt = (s: string) => s.replace("T", " ").slice(0, 16);

// Bar sizes Saxo serves, labelled as parse_horizon accepts them.
const HORIZON_OPTIONS = [
  { value: "1m", label: "1m" },
  { value: "5m", label: "5m" },
  { value: "10m", label: "10m" },
  { value: "15m", label: "15m" },
  { value: "30m", label: "30m" },
  { value: "1h", label: "1h" },
  { value: "4h", label: "4h" },
  { value: "1d", label: "1d" },
];

function BackfillPanel({ onDone }: { onDone: () => void }) {
  const [symbol, setSymbol] = useState("");
  const [exchange, setExchange] = useState(DEFAULT_EXCHANGE);
  const [assetType, setAssetType] = useState(DEFAULT_ASSET_TYPE);
  const [horizons, setHorizons] = useState<string[]>(["1m", "5m"]);
  const [since, setSince] = useState("90d");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId);

  useEffect(() => {
    if (job?.status === "done") onDone();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status]);

  const run = async () => {
    setError(null);
    try {
      const { job_id } = await api.submitBackfill({
        symbol,
        asset_type: assetType || null,
        exchange: exchange || null,
        horizons,
        since: since || null,
      });
      setJobId(job_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const field = "rounded border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700";
  const inFlight = jobId != null && job?.status !== "done" && job?.status !== "error";

  return (
    <div className="space-y-2 rounded border border-slate-200 p-3 dark:border-slate-800">
      <h3 className="text-sm font-semibold">Backfill a series</h3>
      <div className="flex flex-wrap items-start gap-2">
        <input
          className={field}
          placeholder="AAPL"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        />
        <select
          className={`${field} w-40`}
          value={exchange}
          onChange={(e) => setExchange(e.target.value)}
        >
          {EXCHANGES.map((x) => (
            <option key={x.value} value={x.value}>
              {x.label}
            </option>
          ))}
        </select>
        <select
          className={`${field} w-40`}
          value={assetType}
          onChange={(e) => setAssetType(e.target.value)}
        >
          {ASSET_TYPES.map((t) => (
            <option key={t.value} value={t.value}>
              {t.label}
            </option>
          ))}
        </select>
        <CheckboxDropdown
          className="w-40"
          options={HORIZON_OPTIONS}
          selected={horizons}
          onChange={setHorizons}
          placeholder="Bar sizes…"
        />
        <input
          className={`${field} w-20`}
          value={since}
          onChange={(e) => setSince(e.target.value)}
        />
        <button
          onClick={run}
          disabled={!symbol || horizons.length === 0 || inFlight}
          className="rounded bg-blue-600 px-3 py-1 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
        >
          Fetch
        </button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {jobId && <JobProgress job={job} />}
    </div>
  );
}

export function DataView() {
  const series = useSeries();
  const [selected, setSelected] = useState<SeriesInfo | null>(null);
  const bars = useBars(
    selected ? { asset_type: selected.asset_type, uic: selected.uic, horizon: selected.horizon } : null,
  );

  return (
    <div className="space-y-4">
      <BackfillPanel onDone={() => series.refetch()} />

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-300 text-left text-slate-500 dark:border-slate-700">
              <th className="py-1 pr-4">Series</th>
              <th className="py-1 pr-4 text-right">Bars</th>
              <th className="py-1 pr-4">First</th>
              <th className="py-1 pr-4">Last</th>
              <th className="py-1 text-right">Files</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {series.data?.map((s) => {
              const active = selected?.uic === s.uic && selected?.horizon === s.horizon;
              return (
                <tr
                  key={`${s.asset_type}-${s.uic}-${s.horizon}`}
                  onClick={() => setSelected(s)}
                  className={`cursor-pointer border-b border-slate-100 dark:border-slate-900 ${
                    active ? "bg-blue-50 dark:bg-blue-950" : "hover:bg-slate-50 dark:hover:bg-slate-900"
                  }`}
                >
                  <td className="py-1 pr-4">
                    {s.asset_type}:{s.uic} @ {s.horizon_label}
                  </td>
                  <td className="py-1 pr-4 text-right">{s.rows.toLocaleString()}</td>
                  <td className="py-1 pr-4">{fmt(s.first)}</td>
                  <td className="py-1 pr-4">{fmt(s.last)}</td>
                  <td className="py-1 text-right">{s.files}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {series.data?.length === 0 && (
          <p className="py-4 text-sm text-slate-500">
            The lake is empty. Backfill a series above, or run{" "}
            <code className="font-mono">trader data backfill</code>.
          </p>
        )}
      </div>

      {selected && (
        <section className="space-y-2">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold">
              {selected.asset_type}:{selected.uic} @ {selected.horizon_label}
            </h3>
            {bars.data && (
              <span className="text-xs text-slate-500">
                {bars.data.returned.toLocaleString()} of {bars.data.rows.toLocaleString()} bars
                {bars.data.decimated ? " (decimated)" : ""} · {bars.data.gaps.length} gaps
              </span>
            )}
          </div>
          {bars.isLoading && <p className="text-sm text-slate-500">Loading…</p>}
          {bars.error && <p className="text-sm text-red-600">{String(bars.error)}</p>}
          {bars.data && <PriceChart bars={bars.data.bars} />}
        </section>
      )}
    </div>
  );
}
