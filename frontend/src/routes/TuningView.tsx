import { Fragment, useState } from "react";

import { api, ApiError } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { ASSET_TYPES, DEFAULT_ASSET_TYPE } from "../assetTypes";
import { DEFAULT_EXCHANGE, EXCHANGES } from "../exchanges";
import { useFeatureSets, useJob, useJobs, useStickyJobId } from "../hooks";
import type { GridValue, Job, TuningReport, TuningRow, TuningSpec } from "../api/types";

const field =
  "rounded border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700";
const label = "block text-xs font-medium text-slate-500";

const num = (s: string): number => Number(s);
const list = (s: string): string[] => s.split(",").map((x) => x.trim()).filter(Boolean);

const SWEEPABLE = [
  "window",
  "hidden",
  "layers",
  "dropout",
  "lr",
  "stop",
  "take",
  "max_bars",
  "min_return",
  "feature_set",
  "threshold",
  "on_no_signal",
  "allocator",
  "fee_bps",
  "spread_bps",
  "slippage_bps",
];

function coerce(v: string): GridValue {
  const t = v.trim();
  if (t === "true" || t === "false") return t === "true";
  if (t !== "" && !Number.isNaN(Number(t))) return Number(t);
  return t;
}

const fmtNum = (v: number | null | undefined, digits = 2, suffix = ""): string =>
  typeof v === "number" ? v.toFixed(digits) + suffix : "–";
const fmtPct = (v: number | null | undefined): string =>
  typeof v === "number" ? (v * 100).toFixed(1) + "%" : "–";

function FoldTable({ row }: { row: TuningRow }) {
  if (!row.folds?.length) return null;
  return (
    <table className="mt-1 w-full font-mono text-[11px] tabular-nums">
      <thead className="text-left text-slate-400">
        <tr>
          <th className="py-0.5 pr-3">fold</th>
          <th className="py-0.5 pr-3">test ends</th>
          <th className="py-0.5 pr-3">val F1</th>
          <th className="py-0.5 pr-3">test F1</th>
          <th className="py-0.5 pr-3">return</th>
          <th className="py-0.5 pr-3">Sharpe</th>
          <th className="py-0.5 pr-3">hit</th>
          <th className="py-0.5 pr-3">turn</th>
          <th className="py-0.5 pr-3">trades</th>
        </tr>
      </thead>
      <tbody>
        {row.folds.map((f) => (
          <tr key={f.fold} className="border-t border-slate-100 dark:border-slate-800">
            <td className="py-0.5 pr-3">{f.fold}</td>
            <td className="py-0.5 pr-3">{f.test_end?.slice(0, 10) ?? "–"}</td>
            <td className="py-0.5 pr-3">{fmtNum(f.val_macro_f1, 3)}</td>
            <td className="py-0.5 pr-3">{fmtNum(f.test_macro_f1, 3)}</td>
            <td className="py-0.5 pr-3">{fmtPct(f.metrics.total_return)}</td>
            <td className="py-0.5 pr-3">{fmtNum(f.metrics.sharpe)}</td>
            <td className="py-0.5 pr-3">{fmtPct(f.metrics.hit_rate)}</td>
            <td className="py-0.5 pr-3">{fmtNum(f.metrics.turnover, 0, "x")}</td>
            <td className="py-0.5 pr-3">{f.metrics.n_trades}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResultTable({ report }: { report: TuningReport }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-500">
        {report.n_configs} configs
        {report.n_errored > 0 && `, ${report.n_errored} errored`} · ranked by median
        out-of-sample Sharpe
        {report.report_path && (
          <>
            {" · "}
            <code className="font-mono text-xs">{report.report_path}</code>
          </>
        )}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-slate-400">
            <tr>
              <th className="py-1 pr-4">#</th>
              <th className="py-1 pr-4">config</th>
              <th className="py-1 pr-4">folds &gt;0.5</th>
              <th className="py-1 pr-4">med Sharpe</th>
              <th className="py-1 pr-4">med return</th>
              <th className="py-1 pr-4">med turnover</th>
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {report.results.map((r, i) => {
              const cfg = Object.entries(r.config)
                .map(([k, v]) => `${k}=${v}`)
                .join(" ");
              const a = r.aggregate;
              const isOpen = open === i;
              return (
                <Fragment key={i}>
                  <tr
                    onClick={() => a && setOpen(isOpen ? null : i)}
                    className={`border-t border-slate-100 dark:border-slate-800 ${
                      a ? "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-900" : ""
                    }`}
                  >
                    <td className="py-1 pr-4">{r.rank ?? "–"}</td>
                    <td className="py-1 pr-4">{cfg || "base"}</td>
                    {a ? (
                      <>
                        <td className="py-1 pr-4">
                          {a.folds_sharpe_gt_0_5}/{a.n_folds}
                        </td>
                        <td className="py-1 pr-4">{fmtNum(a.sharpe.median)}</td>
                        <td className="py-1 pr-4">{fmtPct(a.total_return.median)}</td>
                        <td className="py-1 pr-4">{fmtNum(a.turnover.median, 0, "x")}</td>
                      </>
                    ) : (
                      <td className="py-1 pr-4 text-red-600" colSpan={4}>
                        {r.error}
                      </td>
                    )}
                  </tr>
                  {isOpen && (
                    <tr>
                      <td />
                      <td colSpan={5} className="pb-3">
                        <FoldTable row={r} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RecentSweeps({
  jobs,
  currentId,
  onPick,
}: {
  jobs: Job[];
  currentId: string | null;
  onPick: (id: string) => void;
}) {
  const sweeps = jobs.filter((j) => j.kind === "tuning").slice(0, 8);
  if (sweeps.length === 0) return null;
  return (
    <section>
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Recent sweeps
      </h3>
      <ul className="space-y-1 text-xs">
        {sweeps.map((j) => (
          <li key={j.id}>
            <button
              onClick={() => onPick(j.id)}
              className={`rounded px-2 py-1 font-mono hover:bg-slate-100 dark:hover:bg-slate-800 ${
                j.id === currentId ? "bg-slate-100 dark:bg-slate-800" : ""
              }`}
            >
              <span
                className={
                  j.status === "done"
                    ? "text-green-600"
                    : j.status === "error"
                      ? "text-red-600"
                      : "text-blue-600"
                }
              >
                {j.status}
              </span>{" "}
              {new Date(j.created_at * 1000).toLocaleString()}{" "}
              <span className="text-slate-400">{j.message}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function TuningView() {
  const featureSets = useFeatureSets();
  const jobs = useJobs();
  const [jobId, setJobId] = useStickyJobId("job:tune");
  const job = useJob(jobId);
  const [error, setError] = useState<string | null>(null);

  const [symbols, setSymbols] = useState("US500, GER40");
  const [uics, setUics] = useState("");
  const [assetType, setAssetType] = useState(DEFAULT_ASSET_TYPE);
  const [exchange, setExchange] = useState(DEFAULT_EXCHANGE);
  const [horizon, setHorizon] = useState("15m");
  const [context, setContext] = useState("1h, 4h");
  const [featureSet, setFeatureSet] = useState("mtf_v1");
  const [stop, setStop] = useState("0.006");
  const [take, setTake] = useState("0.012");
  const [maxBars, setMaxBars] = useState("12");
  const [minReturn, setMinReturn] = useState("0");
  const [windowBars, setWindowBars] = useState("32");
  const [since, setSince] = useState("3y");
  const [hidden, setHidden] = useState("64");
  const [layers, setLayers] = useState("2");
  const [epochs, setEpochs] = useState("40");

  const [folds, setFolds] = useState("5");
  const [mode, setMode] = useState<"rolling" | "anchored">("rolling");
  const [trainDays, setTrainDays] = useState("365");
  const [valDays, setValDays] = useState("45");
  const [testDays, setTestDays] = useState("45");
  const [feeBps, setFeeBps] = useState("0.2");
  const [spreadBps, setSpreadBps] = useState("1.0");
  const [slippageBps, setSlippageBps] = useState("0.5");
  const [threshold, setThreshold] = useState("0.15");
  const [workers, setWorkers] = useState("0");

  const [grid, setGrid] = useState<{ key: string; values: string }[]>([
    { key: "window", values: "16, 32" },
    { key: "stop", values: "0.004, 0.008" },
  ]);

  const running = jobId != null && (!job || job.status === "queued" || job.status === "running");
  const report = job?.status === "done" ? (job.result as TuningReport) : null;

  const setRow = (i: number, patch: Partial<{ key: string; values: string }>) =>
    setGrid((g) => g.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const gridObj: Record<string, GridValue[]> = {};
    for (const { key, values } of grid) {
      const k = key.trim();
      const vs = list(values).map(coerce);
      if (k && vs.length) gridObj[k] = vs;
    }
    if (Object.keys(gridObj).length === 0) {
      setError("Add at least one grid row with a field and values.");
      return;
    }
    const spec: TuningSpec = {
      base: {
        symbols: list(symbols),
        uics: list(uics).map(Number).filter((n) => !Number.isNaN(n)),
        asset_type: assetType || null,
        exchange: exchange || null,
        horizon,
        context_horizons: list(context),
        feature_set: featureSet,
        stop: num(stop),
        take: num(take),
        max_bars: num(maxBars),
        min_return: num(minReturn) || 0,
        window: num(windowBars),
        since: since || null,
        hidden: num(hidden),
        layers: num(layers),
        epochs: num(epochs),
      },
      cv: {
        folds: num(folds),
        mode,
        train_days: num(trainDays),
        val_days: num(valDays),
        test_days: num(testDays),
        fee_bps: num(feeBps),
        spread_bps: num(spreadBps),
        slippage_bps: num(slippageBps),
        threshold: num(threshold),
      },
      grid: gridObj,
      max_workers: num(workers),
    };
    try {
      const { job_id } = await api.submitTuning(spec);
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="space-y-3">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div className="col-span-2">
            <label className={label}>Symbols</label>
            <input
              className={`${field} w-full`}
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Uics (optional)</label>
            <input
              className={`${field} w-full`}
              value={uics}
              onChange={(e) => setUics(e.target.value)}
              placeholder="4910, 4913"
            />
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
            <label className={label}>Base horizon</label>
            <input
              className={`${field} w-full`}
              value={horizon}
              onChange={(e) => setHorizon(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Context horizons</label>
            <input
              className={`${field} w-full`}
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Feature set</label>
            <select
              className={`${field} w-full`}
              value={featureSet}
              onChange={(e) => setFeatureSet(e.target.value)}
            >
              {(featureSets.data ?? []).map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className={label}>Since</label>
            <input
              className={`${field} w-full`}
              value={since}
              onChange={(e) => setSince(e.target.value)}
            />
          </div>

          <div>
            <label className={label}>Stop</label>
            <input className={`${field} w-full`} value={stop} onChange={(e) => setStop(e.target.value)} />
          </div>
          <div>
            <label className={label}>Take</label>
            <input className={`${field} w-full`} value={take} onChange={(e) => setTake(e.target.value)} />
          </div>
          <div>
            <label className={label}>Max bars</label>
            <input
              className={`${field} w-full`}
              value={maxBars}
              onChange={(e) => setMaxBars(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Min return</label>
            <input
              className={`${field} w-full`}
              value={minReturn}
              onChange={(e) => setMinReturn(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Window</label>
            <input
              className={`${field} w-full`}
              value={windowBars}
              onChange={(e) => setWindowBars(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Hidden</label>
            <input
              className={`${field} w-full`}
              value={hidden}
              onChange={(e) => setHidden(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Layers</label>
            <input
              className={`${field} w-full`}
              value={layers}
              onChange={(e) => setLayers(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Epochs</label>
            <input
              className={`${field} w-full`}
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
            />
          </div>
        </div>

        <fieldset className="grid grid-cols-2 gap-3 rounded border border-slate-200 p-3 md:grid-cols-4 dark:border-slate-800">
          <legend className="px-1 text-xs font-medium text-slate-500">Walk-forward CV</legend>
          <div>
            <label className={label}>Folds</label>
            <input
              className={`${field} w-full`}
              value={folds}
              onChange={(e) => setFolds(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Mode</label>
            <select
              className={`${field} w-full`}
              value={mode}
              onChange={(e) => setMode(e.target.value as "rolling" | "anchored")}
            >
              <option value="rolling">rolling</option>
              <option value="anchored">anchored</option>
            </select>
          </div>
          <div>
            <label className={label}>Train days</label>
            <input
              className={`${field} w-full`}
              value={trainDays}
              onChange={(e) => setTrainDays(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Val days</label>
            <input
              className={`${field} w-full`}
              value={valDays}
              onChange={(e) => setValDays(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Test days</label>
            <input
              className={`${field} w-full`}
              value={testDays}
              onChange={(e) => setTestDays(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Fee bps</label>
            <input
              className={`${field} w-full`}
              value={feeBps}
              onChange={(e) => setFeeBps(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Spread bps</label>
            <input
              className={`${field} w-full`}
              value={spreadBps}
              onChange={(e) => setSpreadBps(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Slippage bps</label>
            <input
              className={`${field} w-full`}
              value={slippageBps}
              onChange={(e) => setSlippageBps(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Threshold</label>
            <input
              className={`${field} w-full`}
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Workers (0 = auto)</label>
            <input
              className={`${field} w-full`}
              value={workers}
              onChange={(e) => setWorkers(e.target.value)}
            />
          </div>
        </fieldset>

        <fieldset className="space-y-2 rounded border border-slate-200 p-3 dark:border-slate-800">
          <legend className="px-1 text-xs font-medium text-slate-500">
            Grid (each row: a field and the values to sweep)
          </legend>
          <datalist id="sweepable">
            {SWEEPABLE.map((f) => (
              <option key={f} value={f} />
            ))}
          </datalist>
          {grid.map((r, i) => (
            <div key={i} className="flex gap-2">
              <input
                list="sweepable"
                className={`${field} w-40`}
                placeholder="field"
                value={r.key}
                onChange={(e) => setRow(i, { key: e.target.value })}
              />
              <input
                className={`${field} flex-1`}
                placeholder="16, 32, 48"
                value={r.values}
                onChange={(e) => setRow(i, { values: e.target.value })}
              />
              <button
                type="button"
                onClick={() => setGrid((g) => g.filter((_, j) => j !== i))}
                className="px-2 text-slate-400 hover:text-red-600"
              >
                ×
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => setGrid((g) => [...g, { key: "", values: "" }])}
            className="text-xs text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
          >
            + add field
          </button>
        </fieldset>

        <button
          type="submit"
          disabled={running}
          className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {running ? "Sweeping…" : "Run sweep"}
        </button>
      </form>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {jobId && <JobProgress job={job} />}
      {report && (
        <section className="rounded border border-slate-200 p-4 dark:border-slate-800">
          <ResultTable report={report} />
        </section>
      )}

      <p className="text-xs text-slate-400">
        A sweep trains a fresh model per config per fold in a scratch registry and keeps only
        its scores — the models are not saved. Use <span className="font-medium">Train</span> to
        register a model from a config you want to keep.
      </p>

      <RecentSweeps jobs={jobs.data ?? []} currentId={jobId} onPick={setJobId} />
    </div>
  );
}
