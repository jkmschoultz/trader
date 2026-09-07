// Mirrors the pydantic / dataclass shapes in src/trader/service and src/trader/api.

export interface ParamInfo {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
}

export interface StrategyInfo {
  name: string;
  summary: string;
  params: ParamInfo[];
}

export interface AllocatorInfo {
  name: string;
  summary: string;
  params: ParamInfo[];
}

export interface SeriesInfo {
  asset_type: string;
  uic: number;
  horizon: number;
  horizon_label: string;
  rows: number;
  first: string;
  last: string;
  files: number;
}

export interface Bar {
  time: number; // unix seconds
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
}

export interface Gap {
  after: string;
  before: string;
  missing: number;
}

export interface BarsResponse {
  bars: Bar[];
  gaps: Gap[];
  rows: number;
  returned: number;
  decimated: boolean;
}

export interface InstrumentHit {
  symbol: string;
  uic: number;
  asset_type: string;
  exchange_id: string;
  description: string;
}

export interface AuthStatus {
  authenticated: boolean;
  environment: string;
  kind?: string;
  hint?: string | null;
  access_expired?: boolean;
  access_expires_at?: string;
  refresh_expired?: boolean;
  refresh_expires_at?: string | null;
}

export interface BacktestSpec {
  symbols: string[];
  strategy: string;
  uics?: number[];
  asset_type?: string | null;
  exchange?: string | null;
  params?: Record<string, unknown>;
  horizon?: number | string;
  since?: string | null;
  allocator?: string;
  fee_bps?: number;
  spread_bps?: number;
  slippage_bps?: number;
  starting_cash?: number;
  leverage?: number;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface SeriesNotStoredData {
  kind: "series_not_stored";
  symbol: string;
  horizon: string;
  asset_type: string | null;
  uic: number | null;
  since: string | null;
}

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number | null;
  message: string;
  error: string | null;
  error_data: SeriesNotStoredData | Record<string, unknown> | null;
  created_at: number;
  finished_at: number | null;
  result?: unknown;
}

export interface Metrics {
  total_return: number;
  cagr: number;
  ann_vol: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  calmar: number;
  hit_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  exposure: number;
  turnover: number;
  n_trades: number;
}

export interface EquityPoint {
  time: string;
  equity: number;
}

export interface Trade {
  label: string;
  entry_time: string;
  exit_time: string;
  pnl: number;
  return: number;
  exit_reason: string;
  [k: string]: unknown;
}

export interface Fill {
  time: string;
  label: string;
  side: string;
  units: number;
  price: number;
  [k: string]: unknown;
}

export interface InstrumentStats {
  n_trades: number;
  pnl: number;
  return_on_start: number;
  hit_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
}

export interface BacktestResult {
  config: Record<string, unknown>;
  starting_equity: number;
  final_equity: number;
  metrics: Metrics;
  by_instrument: Record<string, InstrumentStats>;
  equity: EquityPoint[];
  trades: Trade[];
  fills: Fill[];
  n_fills: number;
}
