import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { api, ApiError } from "../api/client";
import { useAuthStatus, useJob } from "../hooks";

/** Split a message into text + a trailing URL so the link is clickable. */
function withLink(message: string) {
  const m = message.match(/(https?:\/\/\S+)\s*$/);
  if (!m) return <span>{message}</span>;
  const url = m[1];
  return (
    <span>
      {message.slice(0, m.index).trim()}{" "}
      <a href={url} target="_blank" rel="noreferrer" className="underline">
        open sign-in page
      </a>
    </span>
  );
}

export function AuthBanner() {
  const { data } = useAuthStatus();
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId);

  useEffect(() => {
    if (job?.status === "done") {
      qc.invalidateQueries({ queryKey: ["auth"] });
      setJobId(null);
    }
  }, [job?.status, qc]);

  if (!data || data.authenticated) return null;

  const running = jobId != null && (!job || job.status === "queued" || job.status === "running");

  const signIn = async () => {
    setError(null);
    try {
      const { job_id } = await api.startLogin();
      setJobId(job_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  };

  return (
    <div className="border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span>
          Not signed in to Saxo ({data.environment}). Instrument search and backfills are
          unavailable.
        </span>
        <button
          onClick={signIn}
          disabled={running}
          className="rounded bg-amber-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-amber-700 disabled:opacity-50"
        >
          {running ? "Waiting for sign-in…" : "Sign in to Saxo"}
        </button>
      </div>
      {job?.message && running && (
        <p className="mt-1 whitespace-pre-wrap text-xs">{withLink(job.message)}</p>
      )}
      {job?.status === "error" && (
        <p className="mt-1 text-xs text-red-700 dark:text-red-400">{job.error}</p>
      )}
      {error && <p className="mt-1 text-xs text-red-700 dark:text-red-400">{error}</p>}
    </div>
  );
}
