import { Fragment, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api, ApiError } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { ASSET_TYPES, DEFAULT_ASSET_TYPE } from "../assetTypes";
import { DEFAULT_EXCHANGE, EXCHANGES } from "../exchanges";
import { useFeatureSets, useJob, useModel, useModels, useStickyJobId } from "../hooks";
import type { ModelSummary, TrainingResult, TrainingSpec } from "../api/types";

const field =
  "rounded border border-slate-300 bg-transparent px-2 py-1 text-sm dark:border-slate-700";
const label = "block text-xs font-medium text-slate-500";

const num = (s: string): number => Number(s);
const numOrNull = (s: string): number | null => (s.trim() === "" ? null : Number(s));
const list = (s: string): string[] => s.split(",").map((x) => x.trim()).filter(Boolean);

function Confusion({ title, matrix }: { title: string; matrix: number[][] }) {
  const names = ["down", "flat", "up"];
  return (
    <div>
      <p className={label}>{title} confusion</p>
      <table className="mt-1 font-mono text-xs tabular-nums">
        <thead>
          <tr className="text-slate-400">
            <th className="px-2 py-0.5 text-left">true \ pred</th>
            {names.map((n) => (
              <th key={n} className="px-2 py-0.5 text-right">
                {n}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="px-2 py-0.5 text-slate-400">{names[i]}</td>
              {row.map((v, j) => (
                <td
                  key={j}
                  className={`px-2 py-0.5 text-right ${i === j ? "font-semibold text-green-600" : ""}`}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelDetail({ id }: { id: string }) {
  const { data, isLoading, error } = useModel(id);
  if (isLoading) return <p className="text-xs text-slate-400">loading…</p>;
  if (error || !data) return <p className="text-xs text-red-600">could not load {id}</p>;

  const dist = data.class_distribution ?? {};
  return (
    <div className="space-y-3 py-1 text-xs">
      <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono tabular-nums">
        {Object.entries(data.metrics).map(([k, v]) => (
          <span key={k}>
            <span className="text-slate-400">{k} </span>
            {typeof v === "number" ? v.toFixed(4) : String(v)}
          </span>
        ))}
      </div>
      {Object.keys(dist).length > 0 && (
        <table className="font-mono tabular-nums">
          <thead className="text-slate-400">
            <tr>
              <th className="pr-3 text-left">split</th>
              <th className="pr-3 text-right">down</th>
              <th className="pr-3 text-right">flat</th>
              <th className="pr-3 text-right">up</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(dist).map(([split, c]) => (
              <tr key={split}>
                <td className="pr-3 text-slate-400">{split}</td>
                <td className="pr-3 text-right">{c.down ?? "–"}</td>
                <td className="pr-3 text-right">{c.flat ?? "–"}</td>
                <td className="pr-3 text-right">{c.up ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-slate-500">
        <span>symbols: {data.symbols.join(", ") || "–"}</span>
        <span>train→{String(data.split.train_end ?? "?").slice(0, 10)}</span>
        <span>val→{String(data.split.val_end ?? "?").slice(0, 10)}</span>
        <span>
          net: h{String(data.hyperparameters.hidden)}×{String(data.hyperparameters.layers)}
        </span>
        <span>epochs: {String(data.hyperparameters.epochs ?? "–")}</span>
        <span>lr: {String(data.hyperparameters.lr ?? "–")}</span>
      </div>
    </div>
  );
}

function ModelsTable({ models }: { models: ModelSummary[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (models.length === 0) {
    return <p className="text-sm text-slate-400">No models trained yet.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-slate-400">
          <tr>
            <th className="py-1 pr-4">id</th>
            <th className="py-1 pr-4">features</th>
            <th className="py-1 pr-4">horizon</th>
            <th className="py-1 pr-4">window</th>
            <th className="py-1 pr-4">barriers</th>
            <th className="py-1 pr-4">val F1</th>
            <th className="py-1 pr-4">test F1</th>
          </tr>
        </thead>
        <tbody className="font-mono text-xs">
          {models.map((m) => {
            const ctx = m.context_horizons.length ? `+${m.context_horizons.join(",")}` : "";
            const isOpen = open === m.id;
            return (
              <Fragment key={m.id}>
                <tr
                  onClick={() => setOpen(isOpen ? null : m.id)}
                  className="cursor-pointer border-t border-slate-100 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-900"
                >
                  <td className="py-1 pr-4">
                    {isOpen ? "▾" : "▸"} {m.id}
                  </td>
                  <td className="py-1 pr-4">{m.feature_set}</td>
                  <td className="py-1 pr-4">
                    {m.horizon}m{ctx}
                  </td>
                  <td className="py-1 pr-4">{m.window}</td>
                  <td className="py-1 pr-4">
                    {m.barriers.stop ?? "–"}/{m.barriers.take ?? "–"}/{m.barriers.max_bars}
                  </td>
                  <td className="py-1 pr-4">{m.metrics.val_macro_f1 ?? "–"}</td>
                  <td className="py-1 pr-4">{m.metrics.test_macro_f1 ?? "–"}</td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={7} className="pb-3">
                      <ModelDetail id={m.id} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function TrainView() {
  const featureSets = useFeatureSets();
  const models = useModels();
  const queryClient = useQueryClient();

  const [jobId, setJobId] = useStickyJobId("job:train");
  const [error, setError] = useState<string | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const job = useJob(jobId);

  const [symbols, setSymbols] = useState("AAPL");
  const [assetType, setAssetType] = useState(DEFAULT_ASSET_TYPE);
  const [exchange, setExchange] = useState(DEFAULT_EXCHANGE);
  const [horizon, setHorizon] = useState("5m");
  const [context, setContext] = useState("");
  const [featureSet, setFeatureSet] = useState("price_v1");
  const [stop, setStop] = useState("0.005");
  const [take, setTake] = useState("0.01");
  const [maxBars, setMaxBars] = useState("24");
  const [minReturn, setMinReturn] = useState("0");
  const [windowBars, setWindowBars] = useState("32");
  const [trainEnd, setTrainEnd] = useState("");
  const [valEnd, setValEnd] = useState("");
  const [since, setSince] = useState("2y");
  const [epochs, setEpochs] = useState("40");

  const [hidden, setHidden] = useState("64");
  const [layers, setLayers] = useState("2");
  const [dropout, setDropout] = useState("0.2");
  const [lr, setLr] = useState("0.001");
  const [batchSize, setBatchSize] = useState("128");
  const [seed, setSeed] = useState("0");
  const [bidirectional, setBidirectional] = useState(false);
  const [sampleWeights, setSampleWeights] = useState(false);

  const running = jobId != null && (!job || job.status === "queued" || job.status === "running");
  const result = job?.status === "done" ? (job.result as TrainingResult) : null;

  useEffect(() => {
    if (job?.status === "done") queryClient.invalidateQueries({ queryKey: ["models"] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setJobId(null);
    const spec: TrainingSpec = {
      symbols: list(symbols),
      asset_type: assetType || null,
      exchange: exchange || null,
      horizon,
      context_horizons: list(context),
      feature_set: featureSet,
      stop: numOrNull(stop),
      take: numOrNull(take),
      max_bars: num(maxBars),
      min_return: num(minReturn) || 0,
      window: num(windowBars),
      train_end: trainEnd,
      val_end: valEnd,
      since: since || null,
      epochs: num(epochs),
      hidden: num(hidden),
      layers: num(layers),
      dropout: num(dropout),
      bidirectional,
      lr: num(lr),
      batch_size: num(batchSize),
      seed: num(seed),
      use_sample_weights: sampleWeights,
    };
    try {
      const { job_id } = await api.submitTraining(spec);
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
            <label className={label}>Symbols (comma-separated)</label>
            <input
              className={`${field} w-full`}
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, NVDA"
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
              placeholder="15m, 1h"
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
              placeholder="2y"
            />
          </div>

          <div>
            <label className={label}>Stop (fraction)</label>
            <input
              className={`${field} w-full`}
              value={stop}
              onChange={(e) => setStop(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Take (fraction)</label>
            <input
              className={`${field} w-full`}
              value={take}
              onChange={(e) => setTake(e.target.value)}
            />
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
            <label className={label}>Window (bars)</label>
            <input
              className={`${field} w-full`}
              value={windowBars}
              onChange={(e) => setWindowBars(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Train end</label>
            <input
              type="date"
              className={`${field} w-full`}
              value={trainEnd}
              onChange={(e) => setTrainEnd(e.target.value)}
            />
          </div>
          <div>
            <label className={label}>Val end</label>
            <input
              type="date"
              className={`${field} w-full`}
              value={valEnd}
              onChange={(e) => setValEnd(e.target.value)}
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

        <div>
          <button
            type="button"
            onClick={() => setAdvanced((v) => !v)}
            className="text-xs text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
          >
            {advanced ? "▾" : "▸"} Advanced
          </button>
          {advanced && (
            <div className="mt-2 grid grid-cols-2 gap-3 rounded border border-slate-200 p-3 md:grid-cols-4 dark:border-slate-800">
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
                <label className={label}>Dropout</label>
                <input
                  className={`${field} w-full`}
                  value={dropout}
                  onChange={(e) => setDropout(e.target.value)}
                />
              </div>
              <div>
                <label className={label}>Learning rate</label>
                <input
                  className={`${field} w-full`}
                  value={lr}
                  onChange={(e) => setLr(e.target.value)}
                />
              </div>
              <div>
                <label className={label}>Batch size</label>
                <input
                  className={`${field} w-full`}
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                />
              </div>
              <div>
                <label className={label}>Seed</label>
                <input
                  className={`${field} w-full`}
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                />
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={bidirectional}
                  onChange={(e) => setBidirectional(e.target.checked)}
                />
                Bidirectional
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={sampleWeights}
                  onChange={(e) => setSampleWeights(e.target.checked)}
                />
                Sample weights
              </label>
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={running || !trainEnd || !valEnd}
          className="rounded bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {running ? "Training…" : "Train"}
        </button>
      </form>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {jobId && <JobProgress job={job} />}

      {result && (
        <section className="space-y-3 rounded border border-slate-200 p-4 dark:border-slate-800">
          <p className="text-sm">
            Trained <code className="font-mono">{result.model_id}</code>. Pick it in the{" "}
            <span className="font-medium">Backtest</span> tab (strategy&nbsp;<code>lstm</code>).
          </p>
          <div className="flex flex-wrap gap-6 font-mono text-sm tabular-nums">
            {Object.entries(result.metrics).map(([k, v]) => (
              <span key={k}>
                <span className="text-slate-400">{k} </span>
                {typeof v === "number" ? v.toFixed(4) : String(v)}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-8">
            <Confusion title="val" matrix={result.report.val_confusion} />
            {result.report.test_confusion && (
              <Confusion title="test" matrix={result.report.test_confusion} />
            )}
          </div>
        </section>
      )}

      <section>
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Registered models
        </h3>
        <ModelsTable models={models.data ?? []} />
      </section>
    </div>
  );
}
