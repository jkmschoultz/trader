import { useMemo, useState } from "react";

import { ASSET_TYPES, DEFAULT_ASSET_TYPE } from "../assetTypes";
import { DEFAULT_EXCHANGE, EXCHANGES } from "../exchanges";
import { useAllocators, useModels, useStrategies } from "../hooks";
import type { BacktestSpec, ParamInfo } from "../api/types";

const field = "rounded border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700";
const label = "block text-xs font-medium text-slate-500";

function coerce(raw: string, p: ParamInfo): unknown {
  const t = raw.trim();
  if (t === "") return p.required ? "" : null;
  if (t === "none" || t === "null") return null;
  if (t === "true" || t === "false") return t === "true";
  const n = Number(t);
  return Number.isNaN(n) ? t : n;
}

export function StrategyForm({
  onSubmit,
  busy,
}: {
  onSubmit: (spec: BacktestSpec) => void;
  busy: boolean;
}) {
  const strategies = useStrategies();
  const allocators = useAllocators();
  const models = useModels();

  const [symbols, setSymbols] = useState("AAPL");
  const [exchange, setExchange] = useState(DEFAULT_EXCHANGE);
  const [uics, setUics] = useState("");
  const [assetType, setAssetType] = useState(DEFAULT_ASSET_TYPE);
  const [strategy, setStrategy] = useState("ma_cross");
  const [horizon, setHorizon] = useState("5m");
  const [since, setSince] = useState("90d");
  const [allocator, setAllocator] = useState("equal-weight");
  const [feeBps, setFeeBps] = useState("0.5");
  const [leverage, setLeverage] = useState("1");
  const [startingCash, setStartingCash] = useState("100000");
  const [params, setParams] = useState<Record<string, string>>({});

  const current = useMemo(
    () => strategies.data?.find((s) => s.name === strategy),
    [strategies.data, strategy],
  );

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const p: Record<string, unknown> = {};
    for (const info of current?.params ?? []) {
      const raw = params[info.name];
      if (raw != null && raw.trim() !== "") p[info.name] = coerce(raw, info);
    }
    onSubmit({
      symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean),
      uics: uics.split(",").map((s) => s.trim()).filter(Boolean).map(Number),
      asset_type: assetType || null,
      exchange: exchange || null,
      strategy,
      params: p,
      horizon,
      since: since || null,
      allocator,
      fee_bps: Number(feeBps) || 0,
      leverage: Number(leverage) || 1,
      starting_cash: Number(startingCash) || 100000,
    });
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <div className="col-span-2 md:col-span-1">
          <label className={label}>Symbols (comma-separated)</label>
          <input
            className={`${field} w-full`}
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="AAPL, NVDA"
          />
        </div>
        <div>
          <label className={label}>Preferred exchange</label>
          <select
            className={`${field} w-full`}
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
          >
            {EXCHANGES.map((x) => (
              <option key={x.value} value={x.value}>
                {x.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={label}>Uics (optional, positional)</label>
          <input className={`${field} w-full`} value={uics} onChange={(e) => setUics(e.target.value)} placeholder="211" />
        </div>
        <div className="col-span-2 md:col-span-3">
          <p className="text-xs text-slate-400">
            Bare tickers (<code>AAPL, NVDA</code>) resolve on the preferred exchange for the
            chosen asset type; add a venue (<code>AAPL:xmil</code>) to pin one.
          </p>
        </div>
        <div>
          <label className={label}>Asset type</label>
          <select
            className={`${field} w-full`}
            value={assetType}
            onChange={(e) => setAssetType(e.target.value)}
          >
            {ASSET_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={label}>Strategy</label>
          <select className={`${field} w-full`} value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {strategies.data?.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={label}>Horizon</label>
          <input className={`${field} w-full`} value={horizon} onChange={(e) => setHorizon(e.target.value)} />
        </div>
        <div>
          <label className={label}>Since</label>
          <input className={`${field} w-full`} value={since} onChange={(e) => setSince(e.target.value)} placeholder="90d" />
        </div>
        <div>
          <label className={label}>Allocator</label>
          <select className={`${field} w-full`} value={allocator} onChange={(e) => setAllocator(e.target.value)}>
            {allocators.data?.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={label}>Fee (bps)</label>
          <input className={`${field} w-full`} value={feeBps} onChange={(e) => setFeeBps(e.target.value)} />
        </div>
        <div>
          <label className={label}>Leverage</label>
          <input className={`${field} w-full`} value={leverage} onChange={(e) => setLeverage(e.target.value)} />
        </div>
        <div>
          <label className={label}>Starting cash</label>
          <input className={`${field} w-full`} value={startingCash} onChange={(e) => setStartingCash(e.target.value)} />
        </div>
      </div>

      {current && current.params.length > 0 && (
        <fieldset className="rounded border border-slate-200 p-3 dark:border-slate-800">
          <legend className="px-1 text-xs text-slate-500">{current.name} parameters</legend>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {current.params
              .filter((p) => p.name !== "models_dir")
              .map((p) => (
                <div key={p.name}>
                  <label className={label}>
                    {p.name} <span className="text-slate-400">{p.type}</span>
                  </label>
                  {strategy === "lstm" && p.name === "model" ? (
                    <select
                      className={`${field} w-full`}
                      value={params[p.name] ?? ""}
                      onChange={(e) => setParams((prev) => ({ ...prev, [p.name]: e.target.value }))}
                    >
                      <option value="">
                        {models.data?.length ? "select a model…" : "no models trained yet"}
                      </option>
                      {models.data?.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.id}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      className={`${field} w-full`}
                      placeholder={p.default == null ? "" : String(p.default)}
                      value={params[p.name] ?? ""}
                      onChange={(e) => setParams((prev) => ({ ...prev, [p.name]: e.target.value }))}
                    />
                  )}
                </div>
              ))}
          </div>
        </fieldset>
      )}

      <button
        type="submit"
        disabled={busy}
        className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {busy ? "Running…" : "Run backtest"}
      </button>
    </form>
  );
}
