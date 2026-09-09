import { useState } from "react";

import { ActiveJobs } from "./components/ActiveJobs";
import { AuthBanner } from "./components/AuthBanner";
import { DataView } from "./routes/DataView";
import { BacktestView } from "./routes/BacktestView";
import { TrainView } from "./routes/TrainView";
import { TuningView } from "./routes/TuningView";

type Tab = "data" | "backtest" | "train" | "tune";

const TABS: { id: Tab; label: string }[] = [
  { id: "data", label: "Data" },
  { id: "backtest", label: "Backtest" },
  { id: "train", label: "Train" },
  { id: "tune", label: "Tune" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("backtest");

  return (
    <div className="mx-auto min-h-screen max-w-6xl">
      <AuthBanner />
      <header className="flex items-center gap-6 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <span className="font-semibold">trader</span>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`rounded px-3 py-1 text-sm ${
                tab === t.id
                  ? "bg-slate-200 font-medium dark:bg-slate-800"
                  : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <ActiveJobs />

      <main className="p-4">
        {tab === "data" && <DataView />}
        {tab === "backtest" && <BacktestView />}
        {tab === "train" && <TrainView />}
        {tab === "tune" && <TuningView />}
      </main>
    </div>
  );
}
