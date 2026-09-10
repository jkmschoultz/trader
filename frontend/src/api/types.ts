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

// ---- model layer (src/trader/features, src/trader/models, src/trader/service/training) ----

export interface FeatureSetInfo {
  name: string;
  summary: string;
  columns: string[];
  context_capable: boolean;
}

export type ModelType = "lstm" | "gbm";

export interface TrainingSpec {
  symbols: string[];
  uics?: number[];
  asset_type?: string | null;
  exchange?: string | null;
  name?: string;
  model_type?: ModelType;
  horizon?: number | string;
  context_horizons?: (number | string)[];
  feature_set?: string;
  stop?: number | null;
  take?: number | null;
  max_bars?: number;
  min_return?: number;
  window?: number;
  train_end: string;
  val_end: string;
  embargo_bars?: number | null;
  since?: string | null;
  // LSTM
  hidden?: number;
  layers?: number;
  dropout?: number;
  bidirectional?: boolean;
  epochs?: number;
  batch_size?: number;
  // GBM (LightGBM)
  num_leaves?: number;
  n_estimators?: number;
  max_depth?: number;
  min_child_samples?: number;
  subsample?: number;
  colsample_bytree?: number;
  // shared
  lr?: number;
  use_sample_weights?: boolean;
  seed?: number;
}

export interface ModelSummary {
  id: string;
  name: string;
  created_at: string;
  model_type: ModelType;
  horizon: number;
  context_horizons: number[];
  feature_set: string;
  window: number;
  barriers: Record<string, number | null>;
  metrics: Record<string, number>;
  symbols: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  created_at: string;
  model_type: ModelType;
  layout: "sequence" | "tabular";
  base_horizon: number;
  context_horizons: number[];
  feature_set: string;
  feature_digest: string;
  window: number;
  barriers: Record<string, number | null>;
  split: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
  metrics: Record<string, number>;
  class_distribution: Record<string, Record<string, number>>;
  symbols: string[];
  asset_type: string | null;
}

export interface TrainReport {
  epochs: {
    epoch: number;
    train_loss: number;
    val_loss: number;
    val_acc: number;
    val_macro_f1: number;
  }[];
  best_epoch: number;
  val_confusion: number[][];
  test_confusion: number[][] | null;
  class_distribution: Record<string, Record<string, number>>;
  metrics: Record<string, number>;
  n_features: number;
  n_train: number;
  n_val: number;
  n_test: number;
  elapsed_seconds: number;
  torch_version: string;
}

export interface TrainingResult {
  model_id: string;
  metrics: Record<string, number>;
  class_distribution: Record<string, Record<string, number>>;
  report: TrainReport;
}

// ---- tuning sweep (src/trader/service/tuning) ----

export type TuningBase = Partial<Omit<TrainingSpec, "train_end" | "val_end">> & {
  symbols: string[];
};

export interface CVConfigSpec {
  folds?: number;
  mode?: "rolling" | "anchored";
  train_days: number;
  val_days: number;
  test_days: number;
  step_days?: number | null;
  fee_bps?: number;
  spread_bps?: number;
  slippage_bps?: number;
  allocator?: string;
  leverage?: number;
  threshold?: number;
  on_no_signal?: "hold" | "flat";
}

export type GridValue = number | string | boolean;

export interface TuningSpec {
  base: TuningBase;
  cv: CVConfigSpec;
  /** "lstm" | "gbm" for a model sweep; any other registered strategy for a classical one. */
  strategy?: string;
  /** fixed constructor args for a classical-strategy sweep. */
  params?: Record<string, unknown>;
  grid: Record<string, GridValue[]>;
  top_k?: number;
  max_workers?: number;
}

export interface Stat {
  median: number | null;
  mean: number | null;
  std: number | null;
}

export interface FoldMetrics {
  fold: number;
  train_end: string;
  val_end: string;
  test_end: string | null;
  val_macro_f1: number | null;
  test_macro_f1: number | null;
  metrics: Metrics;
}

export interface TuningRow {
  config: Record<string, GridValue>;
  rank: number | null;
  error?: string;
  aggregate?: {
    n_folds: number;
    folds_sharpe_gt_0_5: number;
    worst_fold_sharpe: number;
    sharpe: Stat;
    total_return: Stat;
    turnover: Stat;
    hit_rate: Stat;
    profit_factor: Stat;
  };
  folds?: FoldMetrics[];
}

export interface TuningReport {
  measured_at: string;
  finished_at: string;
  strategy?: string;
  symbols: string[];
  horizon: number;
  feature_set: string | null;
  params?: Record<string, unknown> | null;
  grid: Record<string, GridValue[]>;
  n_configs: number;
  n_errored: number;
  cv: Record<string, unknown>;
  results: TuningRow[];
  top: TuningRow[];
  report_path?: string;
}
