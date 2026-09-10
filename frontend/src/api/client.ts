import type {
  AllocatorInfo,
  AuthStatus,
  BacktestSpec,
  BarsResponse,
  FeatureSetInfo,
  InstrumentHit,
  Job,
  ModelInfo,
  ModelSummary,
  SeriesInfo,
  StrategyInfo,
  TrainingSpec,
  TuningSpec,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status line */
    }
    throw new ApiError(resp.status, detail);
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  authStatus: () => request<AuthStatus>("/auth/status"),
  startLogin: () => request<{ job_id: string }>("/auth/login", { method: "POST" }),

  strategies: () => request<StrategyInfo[]>("/catalog/strategies"),
  allocators: () => request<AllocatorInfo[]>("/catalog/allocators"),
  featureSets: () => request<FeatureSetInfo[]>("/catalog/feature-sets"),

  series: () => request<SeriesInfo[]>("/lake/series"),
  bars: (params: {
    asset_type: string;
    uic: number;
    horizon: number;
    start?: string;
    end?: string;
    max_points?: number;
  }) => {
    const q = new URLSearchParams();
    q.set("asset_type", params.asset_type);
    q.set("uic", String(params.uic));
    q.set("horizon", String(params.horizon));
    if (params.start) q.set("start", params.start);
    if (params.end) q.set("end", params.end);
    if (params.max_points) q.set("max_points", String(params.max_points));
    return request<BarsResponse>(`/lake/bars?${q.toString()}`);
  },

  searchInstruments: (q: string, assetType?: string) => {
    const p = new URLSearchParams({ q });
    if (assetType) p.set("asset_type", assetType);
    return request<InstrumentHit[]>(`/instruments/search?${p.toString()}`);
  },

  submitBacktest: (spec: BacktestSpec) =>
    request<{ job_id: string }>("/backtests", {
      method: "POST",
      body: JSON.stringify(spec),
    }),

  submitBackfill: (spec: Record<string, unknown>) =>
    request<{ job_id: string }>("/lake/backfill", {
      method: "POST",
      body: JSON.stringify(spec),
    }),

  refreshSymbols: (refresh = false) =>
    request<{ job_id: string }>(`/lake/symbols?refresh=${refresh ? "true" : "false"}`, {
      method: "POST",
    }),

  submitTraining: (spec: TrainingSpec) =>
    request<{ job_id: string }>("/training", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  models: () => request<ModelSummary[]>("/models"),
  model: (id: string) => request<ModelInfo>(`/models/${id}`),

  submitTuning: (spec: TuningSpec) =>
    request<{ job_id: string }>("/tuning", {
      method: "POST",
      body: JSON.stringify(spec),
    }),

  job: (id: string) => request<Job>(`/jobs/${id}`),
  jobs: () => request<Job[]>("/jobs"),
  jobEventsUrl: (id: string) => `/api/jobs/${id}/events`,
};
