import { useJobs } from "../hooks";
import type { Job } from "../api/types";

const KIND_LABEL: Record<string, string> = {
  backfill: "Backfill",
  backtest: "Backtest",
  training: "Training",
  tuning: "Sweep",
  "auth-login": "Login",
};

function Row({ job }: { job: Job }) {
  const pct = job.progress != null ? Math.round(job.progress * 100) : null;
  return (
    <div className="flex items-center gap-3 py-1 text-xs">
      <span className="inline-flex h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-blue-500" />
      <span className="w-16 shrink-0 font-medium">{KIND_LABEL[job.kind] ?? job.kind}</span>
      <div className="h-1 w-24 shrink-0 overflow-hidden rounded bg-slate-200 dark:bg-slate-700">
        <div
          className="h-full bg-blue-500 transition-all"
          style={{ width: `${pct ?? 15}%` }}
        />
      </div>
      {pct != null && <span className="w-8 shrink-0 tabular-nums text-slate-500">{pct}%</span>}
      <span className="truncate font-mono text-slate-500">{job.message || job.status}</span>
    </div>
  );
}

/** A sticky strip of every queued/running job, visible on every tab. */
export function ActiveJobs() {
  const { data } = useJobs();
  const active = (data ?? []).filter((j) => j.status === "queued" || j.status === "running");
  if (active.length === 0) return null;

  return (
    <div className="sticky top-0 z-10 border-b border-slate-200 bg-amber-50/80 px-4 py-1.5 backdrop-blur dark:border-slate-800 dark:bg-amber-950/40">
      {active.map((j) => (
        <Row key={j.id} job={j} />
      ))}
    </div>
  );
}
