import { useEffect, useState } from "react";

import { api, ApiError } from "../api/client";
import { StrategyForm } from "../components/StrategyForm";
import { JobProgress } from "../components/JobProgress";
import { useJob, useStickyJobId } from "../hooks";
import type { BacktestResult, BacktestSpec, Job, SeriesNotStoredData } from "../api/types";
import { ResultsView } from "./ResultsView";

function missingSeries(job: Job | null): SeriesNotStoredData | null {
  const d = job?.error_data;
  return d && (d as SeriesNotStoredData).kind === "series_not_stored"
    ? (d as SeriesNotStoredData)
    : null;
}

function FetchMissing({
  data,
  spec,
  onFetched,
}: {
  data: SeriesNotStoredData;
  spec: BacktestSpec;
  onFetched: () => void;
}) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId);

  useEffect(() => {
    if (job?.status === "done") onFetched();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status]);

  const fetchIt = async () => {
    setError(null);
    try {
      const { job_id } = await api.submitBackfill({
        symbol: data.symbol,
        asset_type: data.asset_type ?? spec.asset_type ?? null,
        exchange: spec.exchange ?? null,
        horizons: [data.horizon],
        since: spec.since ?? data.since ?? null,
      });
      setJobId(job_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const running = jobId != null && job?.status !== "done" && job?.status !== "error";

  return (
    <div className="space-y-2 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
      <p>
        No <code className="font-mono">{data.symbol}</code> bars at{" "}
        <code className="font-mono">{data.horizon}</code> in the lake yet.
      </p>
      <button
        onClick={fetchIt}
        disabled={running}
        className="rounded bg-amber-600 px-3 py-1 text-xs font-medium text-white hover:bg-amber-700 disabled:opacity-50"
      >
        {running ? "Fetching…" : `Fetch ${data.symbol} ${data.horizon} and re-run`}
      </button>
      {error && <p className="text-red-700 dark:text-red-400">{error}</p>}
      {jobId && <JobProgress job={job} />}
    </div>
  );
}

export function BacktestView() {
  const [jobId, setJobId] = useStickyJobId("job:backtest");
  const [lastSpec, setLastSpec] = useState<BacktestSpec | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const job = useJob(jobId);

  const running = jobId != null && (!job || job.status === "queued" || job.status === "running");

  const submit = async (spec: BacktestSpec) => {
    setSubmitError(null);
    setJobId(null);
    setLastSpec(spec);
    try {
      const { job_id } = await api.submitBacktest(spec);
      setJobId(job_id);
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const result = job?.status === "done" ? (job.result as BacktestResult) : null;
  const missing = missingSeries(job);

  return (
    <div className="space-y-6">
      <StrategyForm onSubmit={submit} busy={running} />
      {submitError && <p className="text-sm text-red-600">{submitError}</p>}
      {jobId && <JobProgress job={job} />}
      {missing && lastSpec && (
        <FetchMissing data={missing} spec={lastSpec} onFetched={() => submit(lastSpec)} />
      )}
      {result && <ResultsView result={result} />}
    </div>
  );
}
