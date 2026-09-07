import type { Job } from "../api/types";

export function JobProgress({ job }: { job: Job | null }) {
  if (!job) return null;

  const pct = job.progress != null ? Math.round(job.progress * 100) : null;
  const tone =
    job.status === "error"
      ? "bg-red-500"
      : job.status === "done"
        ? "bg-green-500"
        : "bg-blue-500";

  return (
    <div className="rounded border border-slate-200 p-3 text-sm dark:border-slate-800">
      <div className="mb-1 flex items-center justify-between">
        <span className="font-medium capitalize">{job.status}</span>
        {pct != null && job.status === "running" && <span className="tabular-nums">{pct}%</span>}
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-800">
        <div
          className={`h-full ${tone} transition-all`}
          style={{ width: `${job.status === "done" ? 100 : (pct ?? (job.status === "running" ? 30 : 0))}%` }}
        />
      </div>
      {job.message && <p className="mt-2 font-mono text-xs text-slate-500">{job.message}</p>}
      {job.error && !job.error_data && (
        <p className="mt-2 whitespace-pre-wrap font-mono text-xs text-red-600">{job.error}</p>
      )}
    </div>
  );
}
