import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "./api/client";
import type { Job } from "./api/types";

export function useAuthStatus() {
  return useQuery({ queryKey: ["auth"], queryFn: api.authStatus, refetchInterval: 60_000 });
}

export function useStrategies() {
  return useQuery({ queryKey: ["strategies"], queryFn: api.strategies, staleTime: Infinity });
}

export function useAllocators() {
  return useQuery({ queryKey: ["allocators"], queryFn: api.allocators, staleTime: Infinity });
}

export function useSeries() {
  return useQuery({ queryKey: ["series"], queryFn: api.series });
}

export function useBars(
  key: { asset_type: string; uic: number; horizon: number } | null,
  maxPoints = 4000,
) {
  return useQuery({
    queryKey: ["bars", key, maxPoints],
    queryFn: () => api.bars({ ...key!, max_points: maxPoints }),
    enabled: key != null,
  });
}

/**
 * Follow a job to completion. Subscribes to the SSE stream and falls back to
 * polling if the stream errors. Returns the latest job snapshot, or null until
 * the first update arrives.
 */
export function useJob(jobId: string | null): Job | null {
  const [job, setJob] = useState<Job | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    setJob(null);
    if (!jobId) return;

    let cancelled = false;
    let source: EventSource | null = null;

    const stopPolling = () => {
      if (pollRef.current != null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    const startPolling = () => {
      if (pollRef.current != null) return;
      pollRef.current = window.setInterval(async () => {
        try {
          const snap = await api.job(jobId);
          if (cancelled) return;
          setJob(snap);
          if (snap.status === "done" || snap.status === "error") stopPolling();
        } catch {
          /* keep trying */
        }
      }, 1000);
    };

    try {
      source = new EventSource(api.jobEventsUrl(jobId));
      source.addEventListener("done", (ev) => {
        try {
          setJob(JSON.parse((ev as MessageEvent).data));
        } catch {
          /* ignore */
        }
        source?.close();
      });
      source.onmessage = () => {
        // progress ping: refetch the authoritative snapshot (cheap, local)
        api.job(jobId).then((snap) => !cancelled && setJob(snap)).catch(() => {});
      };
      source.onerror = () => {
        source?.close();
        startPolling();
      };
    } catch {
      startPolling();
    }

    // one immediate fetch so the UI is not blank while the stream connects
    api.job(jobId).then((snap) => !cancelled && setJob(snap)).catch(() => {});

    return () => {
      cancelled = true;
      source?.close();
      stopPolling();
    };
  }, [jobId]);

  return job;
}
