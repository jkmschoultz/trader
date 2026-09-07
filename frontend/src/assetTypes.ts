// Saxo asset types the data layer is exercised on (trader.saxo.instruments),
// with friendlier labels for the UI. `value` is what the API expects.
export const ASSET_TYPES: { value: string; label: string }[] = [
  { value: "CfdOnStock", label: "Stock (CFD)" },
  { value: "Stock", label: "Stock (equity)" },
  { value: "Etf", label: "ETF" },
  { value: "CfdOnIndex", label: "CFD on index" },
  { value: "FxSpot", label: "FX spot" },
];

// The default selection: the leveraged / margin product.
export const DEFAULT_ASSET_TYPE = "CfdOnStock";
