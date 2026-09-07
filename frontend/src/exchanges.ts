// Exchange suffixes as they appear in a Saxo symbol (AAPL:xnas), with the
// venue name for display. `value` is sent to the API as the preferred exchange
// for resolving a bare ticker. NASDAQ is the default.
export const EXCHANGES: { value: string; label: string }[] = [
  { value: "xnas", label: "NASDAQ" },
  { value: "xnys", label: "NYSE" },
  { value: "arcx", label: "NYSE Arca" },
  { value: "xlon", label: "London (LSE)" },
  { value: "xetr", label: "Xetra" },
  { value: "xmil", label: "Borsa Italiana" },
  { value: "xpar", label: "Euronext Paris" },
  { value: "xams", label: "Euronext Amsterdam" },
  { value: "xbru", label: "Euronext Brussels" },
  { value: "xsto", label: "Nasdaq Stockholm" },
  { value: "xhel", label: "Nasdaq Helsinki" },
  { value: "xcse", label: "Nasdaq Copenhagen" },
  { value: "xswx", label: "SIX Swiss" },
  { value: "xtks", label: "Tokyo" },
  { value: "xhkg", label: "Hong Kong" },
  { value: "xasx", label: "ASX (Australia)" },
];

export const DEFAULT_EXCHANGE = "xnas";
