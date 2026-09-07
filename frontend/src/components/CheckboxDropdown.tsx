import { useEffect, useRef, useState } from "react";

export interface Option {
  value: string;
  label: string;
}

interface Props {
  options: Option[];
  selected: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  className?: string;
}

/** A button that opens a panel of checkboxes; closes on outside click or Escape. */
export function CheckboxDropdown({
  options,
  selected,
  onChange,
  placeholder = "Select…",
  className = "",
}: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const toggle = (value: string) => {
    onChange(
      selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value],
    );
  };

  const summary =
    selected.length === 0
      ? placeholder
      : options
          .filter((o) => selected.includes(o.value))
          .map((o) => o.label)
          .join(", ");

  return (
    <div ref={root} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 rounded border border-slate-300 bg-transparent px-2 py-1 text-left text-sm dark:border-slate-700"
      >
        <span className="truncate">{summary}</span>
        <span className="text-slate-400">▾</span>
      </button>
      {open && (
        <div className="absolute z-20 mt-1 max-h-64 w-max min-w-full overflow-auto rounded border border-slate-300 bg-white p-1 shadow-lg dark:border-slate-700 dark:bg-slate-900">
          {options.map((o) => (
            <label
              key={o.value}
              className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <input
                type="checkbox"
                checked={selected.includes(o.value)}
                onChange={() => toggle(o.value)}
              />
              {o.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
