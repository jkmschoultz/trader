import { useCallback, useEffect, useRef, useState } from "react";
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

export function useFeatureSets() {
  return useQuery({ queryKey: ["feature-sets"], queryFn: api.featureSets, staleTime: Infinity });
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: api.models });
}

export function useModel(id: string | null) {
  return useQuery({
    queryKey: ["model", id],
    queryFn: () => api.model(id!),
    enabled: id != null,
    staleTime: Infinity,
  });
}

/**
 * Every job the server still remembers, polled so it stays live across tab
 * switches. React Query's cache outlives the components that read it, so an
 * in-flight backfill / training / sweep keeps showing after you navigate away
 * and back.
 */
export function useJobs() {
  return useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 2500 });
}

/**
 * A job id that survives navigation and reload within the browser session, so a
 * view re-attaches to the run it kicked off after you come back to it.
 */
export function useStickyJobId(key: string) {
  const [jobId, setJobIdState] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  });
  const setJobId = useCallback(
    (id: string | null) => {
      setJobIdState(id);
      try {
        if (id) sessionStorage.setItem(key, id);
        else sessionStorage.removeItem(key);
      } catch {
        /* private mode / disabled storage: fall back to in-memory only */
      }
    },
    [key],
  );
  return [jobId, setJobId] as const;
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
